package main

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"flag"
	"log"
	"math"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/go-ole/go-ole" // For CoInitialize/CoUninitialize
	"github.com/google/uuid"
	"github.com/jj11hh/opus"
	"github.com/pion/webrtc/v4"           // Updated to v4
	"github.com/pion/webrtc/v4/pkg/media" // Updated to v4
)

// AppConfig holds all command-line arguments
type AppConfig struct {
	SineWave              bool
	PreferredCodec        string
	AudioBitrate          int
	OpusMaxAverageBitrate int // 0 means not set
	OpusMaxPlaybackRate   int
	OpusCBR               bool
	OpusUseInbandFEC      bool
	OpusUseDTX            bool
	Port                  string
}

var appConfig AppConfig // Global application configuration
var opusEncoder *opus.Encoder

// TrackManager holds the list of active audio tracks and a mutex for synchronization.
type TrackManager struct {
	mu            sync.Mutex
	tracks        map[string]*webrtc.TrackLocalStaticSample
	captureConfig *AudioCaptureConfig // Configuration of the audio source (WASAPI or Sine)
}

// NewTrackManager creates a new TrackManager.
func NewTrackManager(config *AudioCaptureConfig) *TrackManager {
	return &TrackManager{
		tracks:        make(map[string]*webrtc.TrackLocalStaticSample),
		captureConfig: config,
	}
}

// AddTrack adds a new track to the manager.
func (tm *TrackManager) AddTrack(track *webrtc.TrackLocalStaticSample) string {
	tm.mu.Lock()
	defer tm.mu.Unlock()
	trackID := uuid.NewString()
	tm.tracks[trackID] = track
	log.Printf("Track added to manager. ID: %s. Total tracks: %d", trackID, len(tm.tracks))
	return trackID
}

// RemoveTrack removes a track from the manager by ID.
func (tm *TrackManager) RemoveTrack(id string) {
	tm.mu.Lock()
	defer tm.mu.Unlock()
	if _, ok := tm.tracks[id]; ok {
		delete(tm.tracks, id)
		log.Printf("Track removed from manager. ID: %s. Total tracks: %d", id, len(tm.tracks))
	}
}

// WriteSampleToAllTracks writes a media sample to all managed tracks.
// It removes tracks that return ErrClosedPipe.
func (tm *TrackManager) WriteSampleToAllTracks(sample media.Sample) {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	tracksToRemove := []string{}
	for id, track := range tm.tracks {
		if err := track.WriteSample(sample); err != nil {
			if strings.Contains(err.Error(), "use of closed network connection") ||
				strings.Contains(err.Error(), "srtp: RtpSender is closed") ||
				strings.Contains(err.Error(), "io: read/write on closed pipe") ||
				strings.Contains(err.Error(), "broken pipe") {
				tracksToRemove = append(tracksToRemove, id)
			} else {
				log.Printf("Error writing sample to track %s: %v", id, err)
			}
		}
	}

	for _, id := range tracksToRemove {
		delete(tm.tracks, id) // Already under lock
		log.Printf("Track removed due to connection closed/error. ID: %s. Total tracks: %d", id, len(tm.tracks))
	}
}

func getCodecCapability(codecName string, sourceSampleRate uint32, sourceChannels uint16, cliCfg AppConfig) webrtc.RTPCodecCapability {
	codecName = strings.ToLower(codecName)
	log.Printf("Determining codec capability for: %s, source SR: %d, source CH: %d", codecName, sourceSampleRate, sourceChannels)

	switch codecName {
	case "opus":
		opusFmtpParams := []string{}
		if cliCfg.OpusUseInbandFEC {
			opusFmtpParams = append(opusFmtpParams, "useinbandfec=1")
		}
		opusFmtpParams = append(opusFmtpParams, "minptime=10")

		if cliCfg.OpusMaxPlaybackRate > 0 {
			opusFmtpParams = append(opusFmtpParams, "maxplaybackrate="+strconv.Itoa(cliCfg.OpusMaxPlaybackRate))
		}
		if cliCfg.OpusMaxAverageBitrate > 0 {
			opusFmtpParams = append(opusFmtpParams, "maxaveragebitrate="+strconv.Itoa(cliCfg.OpusMaxAverageBitrate))
		}
		if cliCfg.OpusCBR {
			opusFmtpParams = append(opusFmtpParams, "cbr=1")
		}
		if cliCfg.OpusUseDTX {
			opusFmtpParams = append(opusFmtpParams, "usedtx=1")
		} else {
			opusFmtpParams = append(opusFmtpParams, "usedtx=0")
		}

		// Only add stereo parameters if the source is actually stereo.
		// The opus.Encoder will be initialized with the sourceChannels.
		// WebRTC expects Channels: 2 in RTPCodecCapability for Opus regardless of actual channels,
		// but SDP fmtp should reflect reality.
		if sourceChannels == 2 {
			opusFmtpParams = append(opusFmtpParams, "stereo=1")
			opusFmtpParams = append(opusFmtpParams, "sprop-stereo=1")
		}

		return webrtc.RTPCodecCapability{
			MimeType:    webrtc.MimeTypeOpus,
			ClockRate:   48000, // Opus always uses 48000 Hz clock rate in WebRTC
			Channels:    2,     // Per RFC7587, Opus RTP payload format SHOULD set channels to 2.
			SDPFmtpLine: strings.Join(opusFmtpParams, ";"),
		}
	case "pcmu":
		return webrtc.RTPCodecCapability{MimeType: webrtc.MimeTypePCMU, ClockRate: 8000, Channels: sourceChannels}
	case "pcma":
		return webrtc.RTPCodecCapability{MimeType: webrtc.MimeTypePCMA, ClockRate: 8000, Channels: sourceChannels}
	case "g722":
		return webrtc.RTPCodecCapability{MimeType: webrtc.MimeTypeG722, ClockRate: 8000, Channels: sourceChannels} // G.722 is typically mono, but can be stereo. Here using sourceChannels.
	default:
		log.Printf("Warning: Unknown preferred codec '%s', defaulting to Opus.", codecName)
		// Default to Opus, assuming mono if sourceChannels is not 2
		opusDefaultFmtp := "minptime=10;useinbandfec=1;usedtx=0"
		if sourceChannels == 2 {
			opusDefaultFmtp += ";stereo=1;sprop-stereo=1"
		}
		return webrtc.RTPCodecCapability{
			MimeType:    webrtc.MimeTypeOpus,
			ClockRate:   48000,
			Channels:    2,
			SDPFmtpLine: opusDefaultFmtp,
		}
	}
}

func startSineWaveGenerator(config *AudioCaptureConfig) <-chan []byte {
	audioChan := make(chan []byte, 100) // Buffered channel
	go func() {
		defer func() {
			close(audioChan)
			log.Println("Sine wave generator stopped.")
		}()

		frequency := 440.0
		amplitude := 0.3 * 32767.0 // For S16LE
		timeOffset := 0.0
		timeIncrementPerSample := 1.0 / float64(config.SampleRate)

		bufferDuration := time.Duration(config.FramesPerBuffer) * time.Second / time.Duration(config.SampleRate)
		if bufferDuration <= 0 {
			bufferDuration = 20 * time.Millisecond // Fallback
		}
		log.Printf("Sine wave generator: SR=%d, CH=%d, FramesPerBuffer=%d, BufferDuration=%v",
			config.SampleRate, config.Channels, config.FramesPerBuffer, bufferDuration)

		ticker := time.NewTicker(bufferDuration)
		defer ticker.Stop()

		for range ticker.C {
			numSamplesPerChannel := int(config.FramesPerBuffer)
			totalSamples := numSamplesPerChannel * int(config.Channels)
			pcmData := make([]int16, totalSamples)
			for i := 0; i < numSamplesPerChannel; i++ {
				t := timeOffset + float64(i)*timeIncrementPerSample
				sampleValue := amplitude * math.Sin(2*math.Pi*frequency*t)
				for ch := 0; ch < int(config.Channels); ch++ {
					pcmData[i*int(config.Channels)+ch] = int16(sampleValue)
				}
			}
			timeOffset += float64(numSamplesPerChannel) * timeIncrementPerSample

			byteData := make([]byte, totalSamples*2) // 2 bytes per int16
			for i, s := range pcmData {
				binary.LittleEndian.PutUint16(byteData[i*2:], uint16(s))
			}

			select {
			case audioChan <- byteData:
			default:
				log.Println("Sine wave audio channel full, discarding packet.")
			}
		}
	}()
	log.Println("Sine wave generator started.")
	return audioChan
}

// --- Refactored Audio Processing Functions ---

// convertToS16LE converts audio data from various source formats to S16LE.
// sourcePacket: The raw audio data from the source.
// sourceIsFloat: True if source data is float32.
// sourceBitDepth: Bit depth of the source (e.g., 16 or 32).
// sourceChannels: Number of channels in the source audio.
// Returns the S16LE packet and true if conversion was successful.
func convertToS16LE(sourcePacket []byte, sourceIsFloat bool, sourceBitDepth uint16, sourceChannels int) ([]byte, bool) {
	if sourceIsFloat && sourceBitDepth == 32 {
		numTotalSourceSampleValues := len(sourcePacket) / 4
		if len(sourcePacket)%4 != 0 {
			log.Printf("Warning: Float32 source packet length %d is not a multiple of 4. Skipping conversion.", len(sourcePacket))
			return nil, false
		}
		s16leSamples := make([]byte, numTotalSourceSampleValues*2)
		for i := 0; i < numTotalSourceSampleValues; i++ {
			floatSample := math.Float32frombits(binary.LittleEndian.Uint32(sourcePacket[i*4 : (i+1)*4]))
			scaledSample := float64(floatSample) * 32767.0
			var intSample int16
			if scaledSample > 32767.0 {
				intSample = 32767
			} else if scaledSample < -32768.0 {
				intSample = -32768
			} else {
				intSample = int16(scaledSample)
			}
			binary.LittleEndian.PutUint16(s16leSamples[i*2:(i+1)*2], uint16(intSample))
		}
		return s16leSamples, true
	} else if !sourceIsFloat && sourceBitDepth == 16 {
		// Already S16LE, no conversion needed
		return sourcePacket, true
	}

	log.Printf("Warning: Unhandled source format (float:%t, depth:%d) for S16LE target. Passing raw data.", sourceIsFloat, sourceBitDepth)
	return sourcePacket, false // Indicate conversion didn't happen or wasn't strictly S16LE
}

// encodeOpus encodes S16LE PCM data using the global Opus encoder.
// pcmS16LE: The S16LE PCM data, interleaved if stereo.
// opusEncoderChannels: Number of channels Opus encoder is configured for.
// Returns the encoded Opus packet or nil if encoding failed.
func encodeOpus(pcmS16LE []byte, opusEncoderChannels int) []byte {
	if opusEncoder == nil {
		log.Println("Error: Opus encoder is not initialized.")
		return nil
	}
	if len(pcmS16LE)%(opusEncoderChannels*2) != 0 && opusEncoderChannels > 0 {
		log.Printf("Error: Opus pcmS16LE length %d not multiple of frame size for S16LE (%d channels * 2 bytes). Skipping encode.", len(pcmS16LE), opusEncoderChannels)
		return nil
	}

	numPCM16SamplesTotal := len(pcmS16LE) / 2
	pcm16 := make([]int16, numPCM16SamplesTotal)
	for i := 0; i < numPCM16SamplesTotal; i++ {
		pcm16[i] = int16(binary.LittleEndian.Uint16(pcmS16LE[i*2 : (i+1)*2]))
	}

	opusDataBuffer := make([]byte, 4000) // Max opus frame size is smaller, but this is safe
	n, err := opusEncoder.Encode(pcm16, opusDataBuffer)
	if err != nil {
		log.Printf("Opus encoding failed: %v. PCM len: %d.", err, len(pcm16))
		return nil
	}
	return opusDataBuffer[:n]
}

// processAudioChunk handles a single chunk of audio data from the accumulator.
// It performs necessary conversions, encoding, and prepares the sample for WebRTC tracks.
func processAudioChunk(
	chunkFromAccumulator []byte,
	captureConfig *AudioCaptureConfig,
	iterationFrameDuration time.Duration,
	opusEncoderChannels int, // Channels Opus encoder expects (e.g., 1 for mono, 2 for stereo)
) *media.Sample {

	sourceIsFloat := captureConfig.IsFloat
	sourceBitDepth := captureConfig.BitDepth
	sourceChannels := int(captureConfig.Channels)
	sourceBytesPerFullFrame := 0
	bytesPerSourceSampleValue := 0

	if sourceIsFloat && sourceBitDepth == 32 {
		bytesPerSourceSampleValue = 4
	} else if !sourceIsFloat && sourceBitDepth == 16 {
		bytesPerSourceSampleValue = 2
	} else {
		log.Printf("Warning: Unsupported source audio format (float:%t, depth:%d) in processAudioChunk.", sourceIsFloat, sourceBitDepth)
		return nil
	}
	sourceBytesPerFullFrame = bytesPerSourceSampleValue * sourceChannels

	if sourceBytesPerFullFrame == 0 || len(chunkFromAccumulator)%sourceBytesPerFullFrame != 0 {
		log.Printf("Error: chunkFromAccumulator length (%d) is not multiple of source frame size (%d) or frame size is zero. Skipping chunk.", len(chunkFromAccumulator), sourceBytesPerFullFrame)
		return nil
	}

	// --- Convert source audio to S16LE if necessary for Opus or other codecs ---
	// Note: captureConfig.Codec.Channels refers to the WebRTC track's channel count,
	// which might be 2 for Opus even if the source and encoder are mono.
	// The crucial part is that the input to the Opus encoder must match its configured channels.
	s16lePacket := chunkFromAccumulator
	conversionSuccess := true

	// Many codecs, including Opus, expect S16LE input.
	if !(appConfig.PreferredCodec == "opus" && !sourceIsFloat && sourceBitDepth == 16 && int(captureConfig.Channels) == opusEncoderChannels) {
		// If not already in the exact format Opus encoder needs (S16LE, matching channels), convert.
		// This simplified condition assumes Opus encoder always gets S16LE.
		// For non-Opus, conversion to S16LE is generally safe.
		s16lePacket, conversionSuccess = convertToS16LE(chunkFromAccumulator, sourceIsFloat, sourceBitDepth, sourceChannels)
		if !conversionSuccess {
			log.Printf("Failed to convert chunk to S16LE. Skipping.")
			return nil
		}
	}
	// At this point, s16lePacket is S16LE with `sourceChannels`.

	finalPacketData := s16lePacket
	isOpusCodec := captureConfig.Codec.MimeType == webrtc.MimeTypeOpus && opusEncoder != nil

	if isOpusCodec {
		// Ensure `s16lePacket` has the channel count the opusEncoder was initialized with.
		// The `getCodecCapability` and Opus encoder initialization ensure `opusEncoderChannels`
		// reflects the actual channels (1 or 2) for encoding.
		// If sourceChannels != opusEncoderChannels, this is a problem (e.g. source is stereo, encoder mono).
		// This simple model assumes sourceChannels matches opusEncoderChannels for Opus input.
		if sourceChannels != opusEncoderChannels {
			log.Printf("Warning: Opus encoder configured for %d channels, but S16LE input has %d channels. This might lead to issues.", opusEncoderChannels, sourceChannels)
			// Potentially drop, or attempt to process if channels are compatible (e.g. taking only first channel if encoder is mono)
			// For now, we'll proceed, but this is a point of potential mismatch if not handled carefully upstream.
		}

		encodedOpusData := encodeOpus(s16lePacket, opusEncoderChannels)
		if encodedOpusData == nil {
			return nil
		}
		finalPacketData = encodedOpusData
	}

	if iterationFrameDuration > 0 {
		return &media.Sample{Data: finalPacketData, Duration: iterationFrameDuration}
	}
	log.Printf("Skipping sample generation due to zero/negative duration. Encoded size: %d", len(finalPacketData))
	return nil
}

// audioProcessingLoop reads from audioChan, accumulates data, and processes it in chunks.
func audioProcessingLoop(audioChan <-chan []byte, trackManager *TrackManager) {
	log.Println("Audio processing loop started.")
	defer log.Println("Audio processing loop stopped.")

	var audioAccumulator []byte
	opusFrameDuration := 20 * time.Millisecond // Target Opus frame duration

	for rawAudioData := range audioChan {
		if rawAudioData == nil {
			log.Println("Audio channel closed, stopping audio processing loop.")
			return
		}
		audioAccumulator = append(audioAccumulator, rawAudioData...)

		captureConfig := trackManager.captureConfig
		if captureConfig == nil {
			log.Println("Error: captureConfig is nil in audio processing loop.")
			audioAccumulator = []byte{} // Clear accumulator
			continue
		}

		sourceChannels := int(captureConfig.Channels)
		sourceSampleRate := captureConfig.SampleRate
		sourceIsFloat := captureConfig.IsFloat
		sourceBitDepth := captureConfig.BitDepth

		if sourceChannels == 0 || sourceSampleRate == 0 {
			log.Printf("Error critical: Source channels (%d) or sample rate (%d) is zero. Clearing accumulator.", sourceChannels, sourceSampleRate)
			audioAccumulator = []byte{}
			continue
		}

		bytesPerSourceSampleValue := 0
		if sourceIsFloat && sourceBitDepth == 32 {
			bytesPerSourceSampleValue = 4
		} else if !sourceIsFloat && sourceBitDepth == 16 {
			bytesPerSourceSampleValue = 2
		} else {
			log.Printf("Warning: Unsupported source audio format (float:%t, depth:%d). Clearing accumulator.", sourceIsFloat, sourceBitDepth)
			audioAccumulator = []byte{}
			continue
		}
		sourceBytesPerFullFrame := bytesPerSourceSampleValue * sourceChannels

		isOpusCodec := captureConfig.Codec.MimeType == webrtc.MimeTypeOpus && opusEncoder != nil
		// opusEncoderChannels will be the actual number of channels the Opus encoder was initialized with (1 or 2)
		opusEncoderChannels := 0
		if isOpusCodec {
			// This should reflect the channels used when opus.NewEncoder was called.
			// This is derived from captureConfig.Codec.Channels (which is 2 from RTPCodecCapability)
			// OR from sourceChannels if we decided Opus encoder should match source.
			// The current setup initializes Opus encoder with captureConfig.Codec.Channels from getCodecCapability,
			// which could be 1 or 2 based on source.
			opusEncCfgCh := int(captureConfig.Codec.Channels) // This is what was used for opus.NewEncoder if logic is consistent
			if appConfig.PreferredCodec == "opus" {           // If Opus is preferred, encoder matches source channels or 2 if source is stereo
				if captureConfig.Channels == 1 { // if source is mono
					opusEncCfgCh = 1
				} else {
					opusEncCfgCh = 2
				}
			}
			opusEncoderChannels = opusEncCfgCh

		}

		var bytesToTakeFromAccumulatorPerIteration int
		var iterationFrameDuration time.Duration

		if isOpusCodec {
			opusEncoderSampleRate := int(captureConfig.Codec.ClockRate) // Should be 48000 for Opus
			opusPcmSamplesPerChannel := int(float64(opusEncoderSampleRate) * opusFrameDuration.Seconds())

			// Bytes of SOURCE audio needed to produce one Opus frame's worth of samples
			// This assumes sourceSampleRate matches opusEncoderSampleRate or resampling happens before Opus.
			// Currently, no resampling.
			if sourceSampleRate != uint32(opusEncoderSampleRate) && captureConfig.SampleRate != 0 { // check captureConfig.SampleRate for 0
				log.Printf("Warning: Source SR (%d) != Opus Encoder SR (%d). Resampling not implemented. Audio quality/timing issues may occur.", sourceSampleRate, opusEncoderSampleRate)
			}
			bytesToTakeFromAccumulatorPerIteration = opusPcmSamplesPerChannel * sourceChannels * bytesPerSourceSampleValue
			iterationFrameDuration = opusFrameDuration
		} else {
			// For non-Opus codecs, frame based on ~20ms of source audio
			if sourceBytesPerFullFrame > 0 {
				desiredFramesPerChunk := int(float64(sourceSampleRate) * (20 * time.Millisecond).Seconds())
				bytesToTakeFromAccumulatorPerIteration = desiredFramesPerChunk * sourceBytesPerFullFrame
				iterationFrameDuration = 20 * time.Millisecond

				if len(audioAccumulator) < bytesToTakeFromAccumulatorPerIteration && len(audioAccumulator) > 0 && len(audioAccumulator)%sourceBytesPerFullFrame == 0 {
					bytesToTakeFromAccumulatorPerIteration = len(audioAccumulator) // Process what's available if clean frames
					iterationFrameDuration = time.Duration(len(audioAccumulator)/sourceBytesPerFullFrame) * time.Second / time.Duration(sourceSampleRate)
				} else if len(audioAccumulator) < bytesToTakeFromAccumulatorPerIteration {
					continue // Not enough for preferred chunk, wait for more
				}
			} else {
				log.Println("Error: sourceBytesPerFullFrame is zero for non-Opus. Clearing accumulator.")
				audioAccumulator = []byte{}
				continue
			}
		}

		if bytesToTakeFromAccumulatorPerIteration <= 0 {
			continue // Avoid issues if calculation results in zero or negative
		}

		for len(audioAccumulator) >= bytesToTakeFromAccumulatorPerIteration {
			chunkFromAccumulator := make([]byte, bytesToTakeFromAccumulatorPerIteration)
			copy(chunkFromAccumulator, audioAccumulator[:bytesToTakeFromAccumulatorPerIteration])
			audioAccumulator = audioAccumulator[bytesToTakeFromAccumulatorPerIteration:]

			sample := processAudioChunk(chunkFromAccumulator, captureConfig, iterationFrameDuration, opusEncoderChannels)
			if sample != nil {
				trackManager.WriteSampleToAllTracks(*sample)
			}
		}
	}
}

// --- End of Refactored Audio Processing Functions ---

func main() {
	flag.BoolVar(&appConfig.SineWave, "sine-wave", false, "Stream a 440Hz sine wave instead of system audio.")
	flag.StringVar(&appConfig.PreferredCodec, "preferred-codec", "opus", "Preferred audio codec (opus, pcmu, pcma, g722).")
	flag.IntVar(&appConfig.AudioBitrate, "audio-bitrate", 96000, "Target audio bitrate in bps.")
	flag.IntVar(&appConfig.OpusMaxAverageBitrate, "opus-maxaveragebitrate", 0, "Opus specific: Sets 'maxaveragebitrate' in bps. 0 means not set.")
	flag.IntVar(&appConfig.OpusMaxPlaybackRate, "opus-maxplaybackrate", 48000, "Opus specific: Sets 'maxplaybackrate'.")
	flag.BoolVar(&appConfig.OpusCBR, "opus-cbr", false, "Opus specific: Enable constant bitrate.")
	flag.BoolVar(&appConfig.OpusUseInbandFEC, "opus-useinbandfec", true, "Opus specific: Enable inband FEC.")
	flag.BoolVar(&appConfig.OpusUseDTX, "opus-usedtx", true, "Opus specific: Enable Discontinuous Transmission.")
	flag.StringVar(&appConfig.Port, "port", "8088", "Port for the HTTP server.")
	flag.Parse()

	log.Printf("Application Configuration: %+v", appConfig)

	var audioChan <-chan []byte
	var currentCaptureConfig *AudioCaptureConfig
	var err error

	if !appConfig.SineWave {
		if errOle := ole.CoInitialize(0); errOle != nil {
			log.Fatalf("Failed to initialize COM: %v", errOle)
		}
		defer ole.CoUninitialize()
		log.Println("COM Initialized for WASAPI.")

		log.Println("Real audio input mode (WASAPI).")
		audioChan, currentCaptureConfig, err = StartAudioCapture()
		if err != nil {
			log.Fatalf("Failed to start audio capture: %v", err)
		}
		log.Printf("Audio capture started. Initial Config: SampleRate=%d, Channels=%d, BitDepth=%d, IsFloat=%t",
			currentCaptureConfig.SampleRate, currentCaptureConfig.Channels, currentCaptureConfig.BitDepth, currentCaptureConfig.IsFloat)
		currentCaptureConfig.Codec = getCodecCapability(appConfig.PreferredCodec, currentCaptureConfig.SampleRate, currentCaptureConfig.Channels, appConfig)
		log.Printf("Capture config updated for preferred codec: %s, SR: %d, CH: %d (SDP), Fmtp: '%s'",
			currentCaptureConfig.Codec.MimeType, currentCaptureConfig.Codec.ClockRate, currentCaptureConfig.Codec.Channels, currentCaptureConfig.Codec.SDPFmtpLine)
	} else {
		log.Println("Sine wave mode enabled.")
		sineSR := uint32(48000)
		sineCH := uint16(1) // Sine wave is mono
		codecForSine := getCodecCapability(appConfig.PreferredCodec, sineSR, sineCH, appConfig)
		currentCaptureConfig = &AudioCaptureConfig{
			SampleRate:      sineSR,
			Channels:        sineCH,
			BitDepth:        16,
			IsFloat:         false,
			FramesPerBuffer: uint32(float64(sineSR) * 0.020), // 20ms
			Codec:           codecForSine,
		}
		audioChan = startSineWaveGenerator(currentCaptureConfig)
		log.Printf("Sine wave generator configured. Effective Track Config: Codec: %s, SR: %d, CH: %d (SDP), Fmtp: '%s'",
			currentCaptureConfig.Codec.MimeType, currentCaptureConfig.Codec.ClockRate, currentCaptureConfig.Codec.Channels, currentCaptureConfig.Codec.SDPFmtpLine)
	}

	if currentCaptureConfig.Codec.MimeType == webrtc.MimeTypeOpus {
		// Determine actual number of channels for Opus encoder based on source or explicit stereo preference
		opusEncoderChannels := int(currentCaptureConfig.Channels) // Default to source channels
		if appConfig.PreferredCodec == "opus" {                   // If opus is explicitly chosen
			// If source is mono, encoder is mono. If source is stereo, encoder is stereo.
			// The RTPCodecCapability.Channels is always 2 for Opus in WebRTC, but encoder itself matches source.
			// This was handled by getCodecCapability logic for fmtp line correctly reflecting source channels.
			// The encoder itself should use the actual number of channels it will receive.
		} else { // If Opus is a fallback, it might assume 2 channels if not specified.
			// Keep currentCaptureConfig.Channels which could be 1 or 2.
		}
		// If currentCaptureConfig.Channels (from source) is 1, opusEncoderChannels is 1.
		// If currentCaptureConfig.Channels (from source) is 2, opusEncoderChannels is 2.
		// This ensures encoder matches the actual audio stream characteristic.

		log.Printf("Initializing Opus encoder with ClockRate: %d, Channels: %d", currentCaptureConfig.Codec.ClockRate, opusEncoderChannels)
		app := opus.AppAudio
		var errOpusEncoder error
		opusEncoder, errOpusEncoder = opus.NewEncoder(
			int(currentCaptureConfig.Codec.ClockRate), // Should be 48000
			opusEncoderChannels,                       // Use actual source channels for encoder
			app,
		)
		if errOpusEncoder != nil {
			log.Fatalf("Failed to create Opus encoder: %v", errOpusEncoder)
		}

		var appString string
		switch app {
		case opus.AppVoIP:
			appString = "VoIP"
		case opus.AppAudio:
			appString = "Audio"
		case opus.AppRestrictedLowdelay:
			appString = "Restricted Low Delay"
		}

		log.Printf("Opus encoder initialized: SR=%d, Input CH=%d, App=%s",
			currentCaptureConfig.Codec.ClockRate,
			opusEncoderChannels,
			appString)

		if appConfig.OpusMaxAverageBitrate > 0 {
			if err := opusEncoder.SetBitrate(appConfig.OpusMaxAverageBitrate); err != nil {
				log.Printf("Warning: Failed to set Opus encoder bitrate (from opus-maxaveragebitrate %d): %v", appConfig.OpusMaxAverageBitrate, err)
			} else {
				log.Printf("Opus encoder bitrate (from opus-maxaveragebitrate) set to %d bps", appConfig.OpusMaxAverageBitrate)
			}
		} else if appConfig.AudioBitrate > 0 {
			if err := opusEncoder.SetBitrate(appConfig.AudioBitrate); err != nil {
				log.Printf("Warning: Failed to set Opus encoder bitrate (from audio-bitrate %d): %v", appConfig.AudioBitrate, err)
			} else {
				log.Printf("Opus encoder bitrate (from audio-bitrate) set to %d bps", appConfig.AudioBitrate)
			}
		}
		if err := opusEncoder.SetVBR(!appConfig.OpusCBR); err != nil {
			log.Printf("Warning: Failed to set Opus encoder VBR/CBR state (CBR: %t): %v", appConfig.OpusCBR, err)
		} else {
			log.Printf("Opus encoder VBR mode: %t (CBR: %t)", !appConfig.OpusCBR, appConfig.OpusCBR)
		}
		if err := opusEncoder.SetInBandFEC(appConfig.OpusUseInbandFEC); err != nil {
			log.Printf("Warning: Failed to set Opus encoder Inband FEC (%t): %v", appConfig.OpusUseInbandFEC, err)
		} else {
			log.Printf("Opus encoder Inband FEC: %t", appConfig.OpusUseInbandFEC)
		}
		if err := opusEncoder.SetDTX(appConfig.OpusUseDTX); err != nil {
			log.Printf("Warning: Failed to set Opus encoder DTX (%t): %v", appConfig.OpusUseDTX, err)
		} else {
			log.Printf("Opus encoder DTX: %t", appConfig.OpusUseDTX)
		}
	}

	trackManager := NewTrackManager(currentCaptureConfig)

	// Start the refactored audio processing loop
	go audioProcessingLoop(audioChan, trackManager)

	peerConnectionConfig := webrtc.Configuration{
		ICEServers: []webrtc.ICEServer{
			{URLs: []string{"stun:stun.l.google.com:19302"}},
		},
	}

	clientDirPath := "./client"

	if _, err := os.Stat(clientDirPath); err != nil {
		log.Printf("Failed to stat client directory %s, use ../client instead: %v", clientDirPath, err)
		clientDirPath = "../client"
	}

	if _, err := os.Stat(clientDirPath); err != nil {
		log.Printf("Failed to stat client directory %s: %v", clientDirPath, err)
	}

	fs := http.FileServer(http.Dir(clientDirPath))
	http.Handle("/", fs)
	log.Printf("Serving static files from %s at /", clientDirPath)

	http.HandleFunc("/offer", func(w http.ResponseWriter, r *http.Request) {
		log.Println("Received /offer request")
		peerConnection, errPC := webrtc.NewPeerConnection(peerConnectionConfig)
		if errPC != nil {
			log.Print("Failed to create new PeerConnection:", errPC)
			http.Error(w, "Failed to create PeerConnection", http.StatusInternalServerError)
			return
		}

		// Use the captureConfig from the trackManager for the client track's codec details
		clientTrackCodec := trackManager.captureConfig.Codec

		clientAudioTrack, errTrack := webrtc.NewTrackLocalStaticSample(
			clientTrackCodec, // This uses the RTPCodecCapability which might say Channels: 2 for Opus
			"audio-"+uuid.NewString(),
			"pion-stream",
		)
		if errTrack != nil {
			log.Printf("Failed to create audio track for client: %v", errTrack)
			http.Error(w, "Failed to create audio track", http.StatusInternalServerError)
			peerConnection.Close()
			return
		}

		rtpSender, errAddTrack := peerConnection.AddTrack(clientAudioTrack)
		if errAddTrack != nil {
			log.Printf("Failed to add track to PeerConnection: %v", errAddTrack)
			http.Error(w, "Failed to add track", http.StatusInternalServerError)
			peerConnection.Close()
			return
		}

		trackID := trackManager.AddTrack(clientAudioTrack)

		go func() {
			for {
				if pkts, _, rtcpErr := rtpSender.ReadRTCP(); rtcpErr != nil {
					trackManager.RemoveTrack(trackID) // Ensure track is removed if RTCP reader errors out
					return
				} else if len(pkts) > 0 {
				}
			}
		}()

		peerConnection.OnICECandidate(func(candidate *webrtc.ICECandidate) {
			if candidate == nil {
				return
			}
		})

		peerConnection.OnConnectionStateChange(func(s webrtc.PeerConnectionState) {
			log.Printf("Peer Connection State for track %s has changed: %s", trackID, s.String())
			if s == webrtc.PeerConnectionStateFailed || s == webrtc.PeerConnectionStateClosed || s == webrtc.PeerConnectionStateDisconnected {
				log.Printf("PeerConnection for track %s is %s. Removing track from manager.", trackID, s.String())
				trackManager.RemoveTrack(trackID)
				// Consider closing the peerConnection itself here if not already handled
			}
		})

		var offer webrtc.SessionDescription
		if errDecode := json.NewDecoder(r.Body).Decode(&offer); errDecode != nil {
			log.Print("Failed to decode offer:", errDecode)
			http.Error(w, "Failed to decode offer", http.StatusBadRequest)
			peerConnection.Close()
			return
		}

		if errSetRemote := peerConnection.SetRemoteDescription(offer); errSetRemote != nil {
			log.Printf("Failed to set remote description for track %s: %v", trackID, errSetRemote)
			http.Error(w, "Failed to set remote description", http.StatusInternalServerError)
			peerConnection.Close()
			return
		}

		answer, errCreateAnswer := peerConnection.CreateAnswer(nil)
		if errCreateAnswer != nil {
			log.Printf("Failed to create answer for track %s: %v", trackID, errCreateAnswer)
			http.Error(w, "Failed to create answer", http.StatusInternalServerError)
			peerConnection.Close()
			return
		}

		gatherComplete := webrtc.GatheringCompletePromise(peerConnection)
		if errSetLocal := peerConnection.SetLocalDescription(answer); errSetLocal != nil {
			log.Printf("Failed to set local description for track %s: %v", trackID, errSetLocal)
			http.Error(w, "Failed to set local description", http.StatusInternalServerError)
			peerConnection.Close()
			return
		}

		<-gatherComplete

		finalLocalDescription := *peerConnection.LocalDescription()
		response, errMarshal := json.Marshal(finalLocalDescription)
		if errMarshal != nil {
			log.Printf("Failed to marshal final local description for track %s: %v", trackID, errMarshal)
			http.Error(w, "Failed to marshal answer", http.StatusInternalServerError)
			peerConnection.Close()
			return
		}

		w.Header().Set("Content-Type", "application/json")
		if _, errWrite := w.Write(response); errWrite != nil {
			log.Printf("Failed to write response for track %s: %v", trackID, errWrite)
		}
	})

	sigs := make(chan os.Signal, 1)
	signal.Notify(sigs, syscall.SIGINT, syscall.SIGTERM)

	serverAddr := ":" + appConfig.Port
	httpServer := &http.Server{Addr: serverAddr}

	go func() {
		sig := <-sigs
		log.Printf("Received signal: %s, shutting down...", sig)

		// audioProcessingLoop will terminate when audioChan is closed.
		// audioChan (from StartAudioCapture or startSineWaveGenerator) should be closed by its producer.
		// For WASAPI, StartAudioCapture's goroutine should handle closing its channel.
		// For SineWave, its goroutine closes its channel.
		// Closing trackManager or peerConnections might be needed if not handled by connection state changes.

		ctxShutdown, cancelShutdown := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancelShutdown()

		log.Printf("Shutting down HTTP server at %s...", serverAddr)
		if err := httpServer.Shutdown(ctxShutdown); err != nil {
			log.Printf("HTTP server Shutdown error: %v", err)
		}
		log.Println("HTTP server shut down.")

		trackManager.mu.Lock()
		log.Printf("Clearing all %d managed tracks post HTTP server shutdown...", len(trackManager.tracks))
		// Tracks are typically removed on PeerConnection state changes (closed/failed)
		// Explicitly clearing here ensures cleanup if some linger.
		trackManager.tracks = make(map[string]*webrtc.TrackLocalStaticSample)
		trackManager.mu.Unlock()
		log.Println("Track manager cleared.")

		if !appConfig.SineWave {
			// If StartAudioCapture has a Stop method, call it here. Assuming it closes its channel.
			log.Println("Assuming WASAPI audio capture will stop as its channel reader terminates or it handles shutdown.")
		} else {
			log.Println("Sine wave generator will stop as its channel reader terminates.")
		}

		log.Println("Shutdown sequence complete.")
	}()

	log.Printf("Signaling and HTTP server starting on %s...", serverAddr)
	if err := httpServer.ListenAndServe(); err != http.ErrServerClosed {
		log.Fatalf("HTTP server ListenAndServe on %s: %v", serverAddr, err)
	}
	log.Println("HTTP server stopped.")
}
