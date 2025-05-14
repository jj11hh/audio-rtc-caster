package main

import (
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
	"github.com/pion/webrtc/v4"           // Updated to v4
	"github.com/pion/webrtc/v4/pkg/media" // Updated to v4
	"gopkg.in/hraban/opus.v2"
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
	// InputDevice int // Future: for device selection, if WASAPI allows easy enumeration by ID
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

	if len(tm.tracks) == 0 {
		// log.Println("TrackManager.WriteSampleToAllTracks: No tracks to write to.") // Can be verbose
		return
	}
	// log.Printf("TrackManager.WriteSampleToAllTracks: Writing sample (len: %d, dur: %v) to %d tracks", len(sample.Data), sample.Duration, len(tm.tracks))

	tracksToRemove := []string{}
	for id, track := range tm.tracks {
		log.Printf("TrackManager.WriteSampleToAllTracks: Attempting to write to track %s (Sample len: %d, dur: %v)", id, len(sample.Data), sample.Duration)
		if err := track.WriteSample(sample); err != nil {
			// media.ErrClosedPipe might not be exported or could have changed.
			// The string checks are more robust for identifying closed pipe scenarios.
			if strings.Contains(err.Error(), "use of closed network connection") ||
				strings.Contains(err.Error(), "srtp: RtpSender is closed") ||
				strings.Contains(err.Error(), "io: read/write on closed pipe") || // Common generic pipe error
				strings.Contains(err.Error(), "broken pipe") { // Another common one
				log.Printf("Track %s connection closed (err: %v), scheduling for removal.", id, err)
				tracksToRemove = append(tracksToRemove, id)
			} else {
				log.Printf("Error writing sample to track %s: %v", id, err)
			}
		} else {
			log.Printf("TrackManager.WriteSampleToAllTracks: Successfully wrote to track %s", id)
		}
	}

	for _, id := range tracksToRemove {
		delete(tm.tracks, id) // Already under lock
		log.Printf("Track removed due to connection closed/error. ID: %s. Total tracks: %d", id, len(tm.tracks))
	}
}

func getCodecCapability(codecName string, sourceSampleRate uint32, sourceChannels uint16, cliCfg AppConfig) webrtc.RTPCodecCapability {
	codecName = strings.ToLower(codecName)
	log.Printf("Determining codec capability for: %s, source SR: %d, source CH: %d, with CLI config: %+v", codecName, sourceSampleRate, sourceChannels, cliCfg)

	switch codecName {
	case "opus":
		opusFmtpParams := []string{}
		if cliCfg.OpusUseInbandFEC { // Default true in AppConfig
			opusFmtpParams = append(opusFmtpParams, "useinbandfec=1")
		}
		// minptime=10 is a common default, present in Python examples and original hardcoded value.
		opusFmtpParams = append(opusFmtpParams, "minptime=10")

		if cliCfg.OpusMaxPlaybackRate > 0 { // Default 48000 in AppConfig
			opusFmtpParams = append(opusFmtpParams, "maxplaybackrate="+strconv.Itoa(cliCfg.OpusMaxPlaybackRate))
		}
		if cliCfg.OpusMaxAverageBitrate > 0 { // If user sets this CLI flag
			opusFmtpParams = append(opusFmtpParams, "maxaveragebitrate="+strconv.Itoa(cliCfg.OpusMaxAverageBitrate))
		}

		if cliCfg.OpusCBR { // Default false in AppConfig
			opusFmtpParams = append(opusFmtpParams, "cbr=1")
		}
		if cliCfg.SineWave {
			opusFmtpParams = append(opusFmtpParams, "usedtx=1")
		} else {
			opusFmtpParams = append(opusFmtpParams, "usedtx=0") // Explicitly disable DTX
		}

		if sourceChannels == 2 {
			opusFmtpParams = append(opusFmtpParams, "stereo=1")
			opusFmtpParams = append(opusFmtpParams, "sprop-stereo=1") // Often paired with stereo=1
		}

		// opus/48000/2 is forced by RTC7587, We will return real channels with fmtp parameters
		return webrtc.RTPCodecCapability{
			MimeType:    webrtc.MimeTypeOpus,
			ClockRate:   48000,
			Channels:    2,
			SDPFmtpLine: strings.Join(opusFmtpParams, ";"),
		}
	case "pcmu":
		return webrtc.RTPCodecCapability{MimeType: webrtc.MimeTypePCMU, ClockRate: 8000, Channels: sourceChannels}
	case "pcma":
		return webrtc.RTPCodecCapability{MimeType: webrtc.MimeTypePCMA, ClockRate: 8000, Channels: sourceChannels}
	case "g722":
		return webrtc.RTPCodecCapability{MimeType: webrtc.MimeTypeG722, ClockRate: 8000, Channels: sourceChannels}
	default:
		return webrtc.RTPCodecCapability{MimeType: webrtc.MimeTypeOpus, ClockRate: 8000, Channels: sourceChannels}
	}
}

func startSineWaveGenerator(config *AudioCaptureConfig) <-chan []byte {
	audioChan := make(chan []byte, 100) // Buffered channel
	go func() {
		defer func() {
			close(audioChan)
			log.Println("Sine wave generator stopped.")
		}()

		// Python defaults: SineWaveTrack(frequency=440, sample_rate=48000, channels=1, samples_per_frame=int(48000 * 0.02))
		// We use config passed which should reflect these defaults if appConfig.SineWave is true.
		frequency := 440.0
		amplitude := 0.3 * 32767.0 // For S16LE
		timeOffset := 0.0
		timeIncrementPerSample := 1.0 / float64(config.SampleRate)

		// Duration of one buffer of audio data
		bufferDuration := time.Duration(config.FramesPerBuffer) * time.Second / time.Duration(config.SampleRate)
		if bufferDuration <= 0 {
			bufferDuration = 20 * time.Millisecond // Fallback, should be derived
		}
		log.Printf("Sine wave generator: SR=%d, CH=%d, FramesPerBuffer=%d, BufferDuration=%v",
			config.SampleRate, config.Channels, config.FramesPerBuffer, bufferDuration)
		log.Printf("Sine wave generator: Ticker set to %v", bufferDuration)

		ticker := time.NewTicker(bufferDuration)
		defer ticker.Stop()

		for {
			select {
			case <-ticker.C:
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
					// log.Printf("Sine wave generator: Sent %d bytes to audioChan", len(byteData)) // Can be very verbose
				default:
					log.Println("Sine wave audio channel full, discarding packet.")
				}
				// TODO: Add a way to stop this goroutine cleanly, e.g., via a quit channel.
				// For now, it stops when audioChan's reader (main audio processing goroutine) stops.
				// Or if DefaultAudioCapture.Stop() is adapted to signal this too for shutdown.
			}
		}
	}()
	log.Println("Sine wave generator started.")
	return audioChan
}

// SDP modification was removed as it's incompatible with pion/webrtc's SetLocalDescription requirements.
// The SDP from CreateAnswer must be used directly.

func main() {
	// Define CLI flags
	flag.BoolVar(&appConfig.SineWave, "sine-wave", false, "Stream a 440Hz sine wave instead of system audio.")
	flag.StringVar(&appConfig.PreferredCodec, "preferred-codec", "opus", "Preferred audio codec (opus, pcmu, pcma, g722).")
	flag.IntVar(&appConfig.AudioBitrate, "audio-bitrate", 96000, "Target audio bitrate in bps (e.g., 32000). Sets 'b=AS:' line in SDP.")
	flag.IntVar(&appConfig.OpusMaxAverageBitrate, "opus-maxaveragebitrate", 0, "Opus specific: Sets 'maxaveragebitrate' in bps (e.g., 20000). 0 means not set by user.")
	flag.IntVar(&appConfig.OpusMaxPlaybackRate, "opus-maxplaybackrate", 48000, "Opus specific: Sets 'maxplaybackrate' (e.g., 48000 for fullband default).")
	flag.BoolVar(&appConfig.OpusCBR, "opus-cbr", false, "Opus specific: Enable constant bitrate ('cbr=1').")
	flag.BoolVar(&appConfig.OpusUseInbandFEC, "opus-useinbandfec", true, "Opus specific: Enable inband Forward Error Correction ('useinbandfec=1').") // Default true in Python
	flag.BoolVar(&appConfig.OpusUseDTX, "opus-usedtx", true, "Opus specific: Enable Discontinuous Transmission ('usedtx=1').")                        // Default true in Python
	flag.StringVar(&appConfig.Port, "port", "8088", "Port for the HTTP server.")
	flag.Parse()

	log.Printf("Application Configuration: %+v", appConfig)

	var audioChan <-chan []byte
	var currentCaptureConfig *AudioCaptureConfig // Holds the config for the active audio source
	var err error

	if !appConfig.SineWave {
		// Initialize COM for WASAPI only if not using sine wave
		if err := ole.CoInitialize(0); err != nil { // Changed wca to ole
			log.Fatalf("Failed to initialize COM: %v", err)
		}
		defer ole.CoUninitialize() // Changed wca to ole
		log.Println("COM Initialized for WASAPI.")

		log.Println("Real audio input mode (WASAPI).")
		audioChan, currentCaptureConfig, err = StartAudioCapture() // StartAudioCapture sets its own Codec initially
		if err != nil {
			log.Fatalf("Failed to start audio capture: %v", err)
		}
		log.Printf("Audio capture started. Initial Config: %+v", *currentCaptureConfig)
		// Override codec based on CLI, using source's sample rate and channels, and appConfig
		currentCaptureConfig.Codec = getCodecCapability(appConfig.PreferredCodec, currentCaptureConfig.SampleRate, currentCaptureConfig.Channels, appConfig)
		log.Printf("Capture config updated for preferred codec: %+v", currentCaptureConfig.Codec)
	} else {
		log.Println("Sine wave mode enabled.")
		// For sine wave, we define the audio parameters. Python defaults: 48kHz, 1 channel, 20ms frames.
		// The Codec for the track itself will be based on appConfig.PreferredCodec.
		sineSR := uint32(48000)
		sineCH := uint16(1) // Python's SineWaveTrack is mono by default. Opus track will still be stereo if preferred codec is Opus.
		// getCodecCapability handles channel count for Opus (forces to 2) and takes appConfig.
		codecForSine := getCodecCapability(appConfig.PreferredCodec, sineSR, sineCH, appConfig)
		currentCaptureConfig = &AudioCaptureConfig{
			SampleRate:      sineSR,
			Channels:        sineCH, // Actual source channels for sine wave
			BitDepth:        16,     // Generating S16LE
			IsFloat:         false,
			FramesPerBuffer: uint32(float64(sineSR) * 0.020), // 20ms worth of frames
			Codec:           codecForSine,                    // Codec for the WebRTC track
		}
		audioChan = startSineWaveGenerator(currentCaptureConfig) // Generator uses currentCaptureConfig.SampleRate, .Channels, .FramesPerBuffer
		log.Printf("Sine wave generator configured. Effective Track Config: %+v", *currentCaptureConfig)
	}

	// Initialize Opus encoder if Opus is the preferred codec
	if currentCaptureConfig.Codec.MimeType == webrtc.MimeTypeOpus {
		var errOpusEncoder error
		opusEncoder, errOpusEncoder = opus.NewEncoder(
			int(currentCaptureConfig.Codec.ClockRate), // Should be 48000 for Opus
			int(currentCaptureConfig.Codec.Channels),  // Should be 2 for Opus
			opus.AppVoIP,
		)
		if errOpusEncoder != nil {
			log.Fatalf("Failed to create Opus encoder: %v", errOpusEncoder)
		}
		log.Printf("Opus encoder initialized: SR=%d, CH=%d, App=VoIP",
			currentCaptureConfig.Codec.ClockRate,
			currentCaptureConfig.Codec.Channels)

		// Configure Opus encoder based on appConfig
		if appConfig.OpusMaxAverageBitrate > 0 {
			if err := opusEncoder.SetBitrate(appConfig.OpusMaxAverageBitrate); err != nil {
				log.Printf("Warning: Failed to set Opus encoder bitrate (from opus-maxaveragebitrate %d): %v", appConfig.OpusMaxAverageBitrate, err)
			} else {
				log.Printf("Opus encoder bitrate (from opus-maxaveragebitrate) set to %d bps", appConfig.OpusMaxAverageBitrate)
			}
		} else if appConfig.AudioBitrate > 0 { // Fallback to general audio-bitrate
			if err := opusEncoder.SetBitrate(appConfig.AudioBitrate); err != nil {
				log.Printf("Warning: Failed to set Opus encoder bitrate (from audio-bitrate %d): %v", appConfig.AudioBitrate, err)
			} else {
				log.Printf("Opus encoder bitrate (from audio-bitrate) set to %d bps", appConfig.AudioBitrate)
			}
		}

		if err := opusEncoder.SetVariableBitrate(!appConfig.OpusCBR); err != nil { // VBR is !CBR
			log.Printf("Warning: Failed to set Opus encoder VBR/CBR state (CBR: %t): %v", appConfig.OpusCBR, err)
		} else {
			log.Printf("Opus encoder VBR mode: %t (CBR: %t)", !appConfig.OpusCBR, appConfig.OpusCBR)
		}

		if err := opusEncoder.SetInbandFec(appConfig.OpusUseInbandFEC); err != nil {
			log.Printf("Warning: Failed to set Opus encoder Inband FEC (%t): %v", appConfig.OpusUseInbandFEC, err)
		} else {
			log.Printf("Opus encoder Inband FEC: %t", appConfig.OpusUseInbandFEC)
		}

		if err := opusEncoder.SetDtx(appConfig.OpusUseDTX); err != nil {
			log.Printf("Warning: Failed to set Opus encoder DTX (%t): %v", appConfig.OpusUseDTX, err)
		} else {
			log.Printf("Opus encoder DTX: %t", appConfig.OpusUseDTX)
		}
	}

	trackManager := NewTrackManager(currentCaptureConfig)

	// Goroutine to read audio data and write to all managed tracks
	go func() {
		log.Println("Audio processing goroutine started.")

		// Determine the desired interval for pacing WriteSample calls
		var processingInterval time.Duration
		if trackManager.captureConfig != nil && trackManager.captureConfig.SampleRate > 0 && trackManager.captureConfig.FramesPerBuffer > 0 {
			processingInterval = time.Duration(trackManager.captureConfig.FramesPerBuffer) * time.Second / time.Duration(trackManager.captureConfig.SampleRate)
		} else {
			processingInterval = 20 * time.Millisecond // Fallback to 20ms
			log.Printf("Audio processing goroutine: captureConfig not fully available for interval calculation, defaulting to %v", processingInterval)
		}
		log.Printf("Audio processing goroutine: pacing WriteSample calls at interval: %v", processingInterval)

		// processingTicker := time.NewTicker(processingInterval) // Removed for simpler data-driven pacing
		// defer processingTicker.Stop() // Removed

		for rawAudioData := range audioChan {
			// log.Printf("Audio processing: Received %d bytes from audioChan", len(rawAudioData)) // Checked
			if rawAudioData == nil {
				log.Println("Audio channel closed, stopping audio write goroutine.")
				return
			}

			processedPacket := rawAudioData // Start with raw data

			// Determine properties of the source audio from currentCaptureConfig
			sourceIsFloat := currentCaptureConfig.IsFloat
			sourceBitDepth := currentCaptureConfig.BitDepth
			sourceChannels := int(currentCaptureConfig.Channels)
			sourceSampleRate := currentCaptureConfig.SampleRate

			if sourceChannels == 0 || sourceSampleRate == 0 {
				log.Printf("Error critical: Source channels (%d) or sample rate (%d) is zero. Skipping packet.", sourceChannels, sourceSampleRate)
				continue
			}

			// --- Calculate number of frames from source for duration ---
			var numFramesForDurationCalc int
			bytesPerSourceSampleValue := 0 // Bytes for a single sample value (e.g., float32 is 4, int16 is 2)
			if sourceIsFloat && sourceBitDepth == 32 {
				bytesPerSourceSampleValue = 4
			} else if !sourceIsFloat && sourceBitDepth == 16 {
				bytesPerSourceSampleValue = 2
			} else {
				log.Printf("Warning: Unsupported source audio format (float:%t, depth:%d). Cannot calculate frames. Skipping.", sourceIsFloat, sourceBitDepth)
				continue
			}

			// Total bytes for one frame across all source channels
			bytesPerSourceFrame := bytesPerSourceSampleValue * sourceChannels
			if bytesPerSourceFrame == 0 { // Should be caught by sourceChannels == 0 earlier, but defensive
				log.Printf("Error: Calculated bytesPerSourceFrame is zero. Skipping.")
				continue
			}
			if len(rawAudioData)%bytesPerSourceFrame != 0 {
				log.Printf("Error: rawAudioData length (%d) is not a multiple of expected source frame size (%d bytes/frame). Skipping.", len(rawAudioData), bytesPerSourceFrame)
				continue
			}
			numFramesForDurationCalc = len(rawAudioData) / bytesPerSourceFrame
			if numFramesForDurationCalc == 0 {
				// log.Println("Received empty audio data based on frame calculation. Skipping.")
				continue
			}

			// --- Determine if target track codec expects S16LE ---
			isTargetS16LE := false
			targetCodecChannels := int(currentCaptureConfig.Codec.Channels)
			switch currentCaptureConfig.Codec.MimeType {
			case webrtc.MimeTypeOpus, webrtc.MimeTypeG722, webrtc.MimeTypePCMA, webrtc.MimeTypePCMU:
				isTargetS16LE = true
			case "": // Default or unspecified for raw audio tracks might imply S16LE
				isTargetS16LE = true
			}

			// --- Convert source audio to S16LE if necessary ---
			if sourceIsFloat && sourceBitDepth == 32 && isTargetS16LE {
				// Source is float32, target is S16LE. Convert.
				// numTotalSourceSampleValues is total sample points (e.g. L,R,L,R for stereo float if sourceChannels=2)
				numTotalSourceSampleValues := len(rawAudioData) / 4        // 4 bytes per float32 sample value
				s16leSamples := make([]byte, numTotalSourceSampleValues*2) // 2 bytes per int16 sample value
				for i := 0; i < numTotalSourceSampleValues; i++ {
					floatSample := math.Float32frombits(binary.LittleEndian.Uint32(rawAudioData[i*4 : (i+1)*4]))
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
				processedPacket = s16leSamples // Now S16LE, channel count is still `sourceChannels`
			} else if !sourceIsFloat && sourceBitDepth == 16 && isTargetS16LE {
				// Source is already S16LE, target is S16LE. No conversion needed for sample format.
				// processedPacket remains rawAudioData
			} else if !isTargetS16LE {
				log.Printf("Warning: Target codec %s does not expect S16LE. Source (float:%t, depth:%d). Passing raw data.",
					currentCaptureConfig.Codec.MimeType, sourceIsFloat, sourceBitDepth)
				// processedPacket remains rawAudioData. This path might be problematic if codec expects something else.
			} else {
				// Source is not float32 or S16LE, but target is S16LE. Unhandled source format.
				log.Printf("Warning: Unhandled source format (float:%t, depth:%d) for S16LE target. Passing raw data.", sourceIsFloat, sourceBitDepth)
				// processedPacket remains rawAudioData
			}

			// --- Mono to Stereo conversion if track expects stereo but processedPacket (now S16LE) is mono ---
			if isTargetS16LE && targetCodecChannels == 2 && sourceChannels == 1 {
				// At this point, processedPacket is S16LE mono if conversions happened correctly.
				// numFramesForDurationCalc is the number of mono frames from the source.
				numMonoFrames := numFramesForDurationCalc

				// Validate that processedPacket actually contains S16LE mono data of expected length.
				expectedMonoS16LELength := numMonoFrames * 2 // 2 bytes per S16LE sample
				if len(processedPacket) != expectedMonoS16LELength {
					log.Printf("Error: Mismatch in S16LE mono packet size for mono-to-stereo conversion. Expected %d bytes for %d frames, got %d bytes. Skipping.",
						expectedMonoS16LELength, numMonoFrames, len(processedPacket))
					continue
				}

				stereoPacket := make([]byte, numMonoFrames*2*2) // Each mono frame (2 bytes) becomes a 2-channel stereo frame (4 bytes)
				for i := 0; i < numMonoFrames; i++ {
					monoSampleBytes := processedPacket[i*2 : (i+1)*2]
					// Left channel
					stereoPacket[i*4] = monoSampleBytes[0]
					stereoPacket[i*4+1] = monoSampleBytes[1]
					// Right channel (duplicate mono sample)
					stereoPacket[i*4+2] = monoSampleBytes[0]
					stereoPacket[i*4+3] = monoSampleBytes[1]
				}
				processedPacket = stereoPacket // Now S16LE stereo
			}

			// --- Calculate dynamic sample duration using frames from SOURCE data ---
			var dynamicSampleDuration time.Duration
			if numFramesForDurationCalc > 0 {
				dynamicSampleDuration = time.Duration(numFramesForDurationCalc) * time.Second / time.Duration(sourceSampleRate)
			} else {
				// This case should have been caught earlier by numFramesForDurationCalc == 0 check
				log.Println("Warning: numFramesForDurationCalc is zero after processing, cannot calculate duration. Skipping.")
				continue
			}

			if dynamicSampleDuration <= 0 {
				log.Printf("Warning: Calculated dynamicSampleDuration is zero or negative (%v). Frames: %d, SR: %d. Skipping.",
					dynamicSampleDuration, numFramesForDurationCalc, sourceSampleRate)
				continue
			}
			// log.Printf("Audio processing: Processed packet length: %d, Duration: %v. Source SR: %d, CH: %d. Target Codec: %s, CH: %d",
			// 	len(processedPacket), dynamicSampleDuration, sourceSampleRate, sourceChannels, currentCaptureConfig.Codec.MimeType, targetCodecChannels)

			// log.Printf("Audio Processing: rawDataLen=%d, srcSR=%d, srcCH=%d, srcDepth=%d, srcFloat=%t, bytesPerSrcFrame=%d, numFramesForDuration=%d",
			// 	len(rawAudioData), sourceSampleRate, sourceChannels, sourceBitDepth, sourceIsFloat, bytesPerSourceFrame, numFramesForDurationCalc)
			// log.Printf("Audio Processing: targetCodecMime=%s, targetCH=%d, isTargetS16LE=%t",
			// 	currentCaptureConfig.Codec.MimeType, targetCodecChannels, isTargetS16LE)
			// if isTargetS16LE && targetCodecChannels == 2 && sourceChannels == 1 {
			// 	log.Printf("Audio Processing: Mono-to-Stereo conversion was performed.")
			// }
			// log.Printf("Audio Processing: processedPacketLen=%d, dynamicSampleDuration=%s",
			// 	len(processedPacket), dynamicSampleDuration.String())

			encodedPacketData := processedPacket // Default to PCM data

			if currentCaptureConfig.Codec.MimeType == webrtc.MimeTypeOpus && opusEncoder != nil {
				// Convert S16LE []byte to []int16 for Opus encoder
				if len(processedPacket)%2 != 0 {
					log.Printf("Error: processedPacket length %d is not even for S16LE to []int16 conversion. Skipping Opus encoding.", len(processedPacket))
				} else {
					numPCM16Samples := len(processedPacket) / 2
					pcm16 := make([]int16, numPCM16Samples)
					for i := 0; i < numPCM16Samples; i++ {
						pcm16[i] = int16(binary.LittleEndian.Uint16(processedPacket[i*2 : (i+1)*2]))
					}

					// Prepare buffer for Opus encoded data.
					// A common Opus frame is 20ms. At 48kHz stereo, this is 960 samples/channel.
					// Max bitrate for Opus is high, but typical frame sizes are small.
					// A buffer of 4000 bytes should be sufficient for most Opus frames.
					opusDataBuffer := make([]byte, 4000)

					n, err := opusEncoder.Encode(pcm16, opusDataBuffer)
					if err != nil {
						log.Printf("Opus encoding failed: %v. Sending raw PCM instead.", err)
						// encodedPacketData remains processedPacket (PCM)
					} else {
						encodedPacketData = opusDataBuffer[:n]
						// log.Printf("Opus encoded %d PCM bytes to %d Opus bytes. Duration: %v", len(processedPacket), n, dynamicSampleDuration)
					}
				}
			}

			trackManager.WriteSampleToAllTracks(media.Sample{Data: encodedPacketData, Duration: dynamicSampleDuration})
			// log.Printf("DEBUG: Wrote sample to track manager. Packet size: %d, Duration: %v", len(encodedPacketData), dynamicSampleDuration) // Example debug line

			// Wait for the next tick to pace the WriteSample calls -- Removed processingTicker
			// <-processingTicker.C
		}
		log.Println("Exited audio track writing loop.")
	}()

	// Prepare the WebRTC peer connection configuration (globally for all connections)
	peerConnectionConfig := webrtc.Configuration{
		ICEServers: []webrtc.ICEServer{
			{
				URLs: []string{"stun:stun.l.google.com:19302"},
			},
		},
	}

	// Serve static files for the client
	// Determine the correct path to the 'client' directory.
	// If running `go run .` from `server-go` dir, `../client` is correct.
	clientDirPath := "../client"
	// For deployed binaries, this path might need to be configurable or discovered differently.
	fs := http.FileServer(http.Dir(clientDirPath))
	http.Handle("/", fs)
	log.Printf("Serving static files from %s at /", clientDirPath)

	http.HandleFunc("/offer", func(w http.ResponseWriter, r *http.Request) {
		log.Println("Received /offer request")
		peerConnection, err := webrtc.NewPeerConnection(peerConnectionConfig)
		if err != nil {
			log.Print("Failed to create new PeerConnection:", err)
			http.Error(w, "Failed to create PeerConnection", http.StatusInternalServerError)
			return
		}
		log.Println("New PeerConnection created.")

		// Use the captureConfig from the trackManager, which is based on CLI args / defaults
		clientTrackCodec := trackManager.captureConfig.Codec
		log.Printf("Client track will be created with codec: %s, SR: %d, CH: %d, Fmtp: '%s'",
			clientTrackCodec.MimeType, clientTrackCodec.ClockRate, clientTrackCodec.Channels, clientTrackCodec.SDPFmtpLine)

		clientAudioTrack, err := webrtc.NewTrackLocalStaticSample(
			clientTrackCodec,
			"audio-"+uuid.NewString(),
			"pion-stream",
		)
		if err != nil {
			log.Printf("Failed to create audio track for client: %v", err)
			http.Error(w, "Failed to create audio track", http.StatusInternalServerError)
			peerConnection.Close()
			return
		}
		log.Printf("Client audio track created successfully.")

		rtpSender, err := peerConnection.AddTrack(clientAudioTrack)
		if err != nil {
			log.Printf("Failed to add track to PeerConnection: %v", err)
			http.Error(w, "Failed to add track", http.StatusInternalServerError)
			peerConnection.Close()
			return
		}
		log.Println("Client audio track added to PeerConnection.")

		trackID := trackManager.AddTrack(clientAudioTrack)

		go func() {
			// rtcpBuf := make([]byte, 1500) // Not needed for ReadRTCP()
			for {
				if pkts, _, rtcpErr := rtpSender.ReadRTCP(); rtcpErr != nil {
					log.Printf("RTCP Read loop for track %s closed or error: %v.", trackID, rtcpErr)
					// Track removal from manager will be handled by OnConnectionStateChange or WriteSample error
					return
				} else if len(pkts) > 0 {
					log.Printf("Received %d RTCP packets for track %s", len(pkts), trackID)
					// Process RTCP packets if necessary (e.g., Sender Reports, Receiver Reports)
				}
			}
		}()

		// The PeerConnection should not be closed when the /offer handler finishes.
		// It needs to stay alive for the duration of the WebRTC session.
		// Closure is handled by OnConnectionStateChange or server shutdown.
		peerConnection.OnICECandidate(func(candidate *webrtc.ICECandidate) {
			if candidate == nil {
				log.Printf("ICE Candidate gathering finished for PC of track %s.", trackID)
				return
			}
			// Normally, these would be sent to the client via the signaling channel.
			log.Printf("ICE Candidate for PC of track %s: %s", trackID, candidate.ToJSON().Candidate)
		})

		peerConnection.OnConnectionStateChange(func(s webrtc.PeerConnectionState) {
			log.Printf("Peer Connection State for track %s has changed: %s\n", trackID, s.String())
			if s == webrtc.PeerConnectionStateFailed || s == webrtc.PeerConnectionStateClosed || s == webrtc.PeerConnectionStateDisconnected {
				log.Printf("PeerConnection for track %s is %s. Removing track from manager.", trackID, s.String())
				trackManager.RemoveTrack(trackID)
			}
		})

		var offer webrtc.SessionDescription
		if err := json.NewDecoder(r.Body).Decode(&offer); err != nil {
			log.Print("Failed to decode offer:", err)
			http.Error(w, "Failed to decode offer", http.StatusBadRequest)
			return
		}
		log.Printf("Offer received for track %s", trackID)
		log.Printf("Offer SDP for track %s:\n%s", trackID, offer.SDP)

		if err := peerConnection.SetRemoteDescription(offer); err != nil {
			log.Printf("Failed to set remote description for track %s: %v", trackID, err)
			http.Error(w, "Failed to set remote description", http.StatusInternalServerError)
			return
		}
		log.Printf("Remote description set for track %s", trackID)

		answer, err := peerConnection.CreateAnswer(nil)
		if err != nil {
			log.Printf("Failed to create answer for track %s: %v", trackID, err)
			http.Error(w, "Failed to create answer", http.StatusInternalServerError)
			return
		}
		log.Printf("Raw Answer created for track %s", trackID)
		log.Printf("Initial Answer SDP for track %s (before SetLocalDescription):\n%s", trackID, answer.SDP)

		// SDP modification was removed. Using answer from CreateAnswer directly.
		// log.Printf("Original Answer SDP before SetLocalDescription:\n%s", answer.SDP)

		gatherComplete := webrtc.GatheringCompletePromise(peerConnection)
		if err := peerConnection.SetLocalDescription(answer); err != nil {
			log.Printf("Failed to set local description for track %s (with modified SDP): %v", trackID, err)
			http.Error(w, "Failed to set local description", http.StatusInternalServerError)
			return
		}
		log.Printf("Local description set for track %s. Waiting for ICE gathering...", trackID)

		<-gatherComplete
		log.Printf("ICE Gathering Complete for track %s.", trackID)

		finalLocalDescription := *peerConnection.LocalDescription()
		log.Printf("Final Local Description SDP for track %s (to be sent to client):\n%s", trackID, finalLocalDescription.SDP)
		response, err := json.Marshal(finalLocalDescription)
		if err != nil {
			log.Printf("Failed to marshal final local description for track %s: %v", trackID, err)
			http.Error(w, "Failed to marshal answer", http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		if _, err := w.Write(response); err != nil {
			log.Printf("Failed to write response for track %s: %v", trackID, err)
		}
		log.Printf("Final Answer sent for track %s", trackID)
	})

	// Graceful shutdown handling
	sigs := make(chan os.Signal, 1)
	signal.Notify(sigs, syscall.SIGINT, syscall.SIGTERM)

	serverAddr := ":" + appConfig.Port
	httpServer := &http.Server{Addr: serverAddr}

	go func() {
		sig := <-sigs
		log.Printf("Received signal: %s, shutting down...", sig)

		// The audio capture goroutine (for WASAPI) and the sine wave generator goroutine
		// will stop when their respective audioChan is closed or the main audio processing
		// goroutine (which reads from audioChan) terminates.
		// The audioChan for WASAPI is closed by the capture goroutine itself upon its termination.
		// The audioChan for the sine wave generator is also closed by its own goroutine upon termination.
		// Thus, no explicit stop call is needed here for DefaultAudioCapture as it's no longer used globally.
		if !appConfig.SineWave {
			log.Println("WASAPI audio capture will stop as its channel reader terminates.")
		} else {
			log.Println("Sine wave generator will stop as its channel reader terminates.")
		}

		trackManager.mu.Lock()
		log.Printf("Closing all %d managed tracks...", len(trackManager.tracks))
		for id := range trackManager.tracks {
			// Tracks themselves don't have a Stop method in pion/webrtc/v3 TrackLocalStaticSample.
			// Closing the PeerConnection (handled by client disconnect or HTTP handler defer)
			// or errors in WriteSample will lead to their removal from the manager.
			log.Printf("Track %s will be implicitly closed/removed.", id)
		}
		trackManager.tracks = make(map[string]*webrtc.TrackLocalStaticSample)
		trackManager.mu.Unlock()
		log.Println("Track manager cleared.")

		log.Printf("Shutting down HTTP server at %s...", serverAddr)
		if err := httpServer.Shutdown(nil); err != nil {
			log.Printf("HTTP server Shutdown: %v", err)
		}
		log.Println("Shutdown sequence complete.")
	}()

	log.Printf("Signaling and HTTP server starting on %s...", serverAddr)
	if err := httpServer.ListenAndServe(); err != http.ErrServerClosed {
		log.Fatalf("HTTP server ListenAndServe on %s: %v", serverAddr, err)
	}
	log.Println("HTTP server stopped.")
}
