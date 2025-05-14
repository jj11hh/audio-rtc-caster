package main

import (
	"fmt"
	"log"
	"time"
	"unsafe" // Required for CoTaskMemFree
	"github.com/go-ole/go-ole"
	"github.com/moutend/go-wca/pkg/wca"
	"github.com/pion/webrtc/v4" // Updated to v4
)

// Standard Windows Wave Format Tags not explicitly defined in go-wca
const (
	WAVE_FORMAT_IEEE_FLOAT uint16 = 0x0003
	WAVE_FORMAT_EXTENSIBLE uint16 = 0xFFFE
)

// AudioCaptureConfig holds parameters of the captured audio.
type AudioCaptureConfig struct {
	SampleRate      uint32
	Channels        uint16
	BitDepth        uint16 // e.g., 16 for int16, 32 for float32
	IsFloat         bool
	FramesPerBuffer uint32 // Number of frames in each buffer from WASAPI
	NBlockAlign     uint16 // Store NBlockAlign to avoid race condition with CoTaskMemFree
	Codec           webrtc.RTPCodecCapability // For informing WebRTC track creation
}

// StartAudioCapture initializes WASAPI, selects the default loopback device,
// starts audio capture, and returns a channel for audio data, capture configuration, and any error.
// IMPORTANT: CoInitialize(0) must be called by the caller (e.g., in main) before this function,
// and CoUninitialize() deferred after this function's lifecycle.
func StartAudioCapture() (readOnlyAudioChan <-chan []byte, config *AudioCaptureConfig, err error) {
	log.Println("Initializing audio capture...")

	audioChan := make(chan []byte, 100) // Buffered channel for audio chunks

	var devenum *wca.IMMDeviceEnumerator
	err = wca.CoCreateInstance(wca.CLSID_MMDeviceEnumerator, 0, wca.CLSCTX_ALL, wca.IID_IMMDeviceEnumerator, &devenum)
	if err != nil {
		return nil, nil, fmt.Errorf("CoCreateInstance for IMMDeviceEnumerator failed: %w", err)
	}
	defer devenum.Release()

	var mmdevice *wca.IMMDevice
	// Use ERender for eDataFlow and DEVICE_STATE_ACTIVE for stateMask.
	// The original code used EConsole which is an ERole, incorrect for stateMask.
	err = devenum.GetDefaultAudioEndpoint(uint32(wca.ERender), wca.DEVICE_STATE_ACTIVE, &mmdevice)
	if err != nil {
		return nil, nil, fmt.Errorf("GetDefaultAudioEndpoint failed: %w", err)
	}
	defer mmdevice.Release()
	log.Println("Default audio endpoint (render) obtained.")

	var audioClient *wca.IAudioClient
	// The obj parameter for Activate must be a pointer to the interface variable.
	err = mmdevice.Activate(wca.IID_IAudioClient, wca.CLSCTX_ALL, nil, &audioClient)
	if err != nil {
		return nil, nil, fmt.Errorf("Activate IAudioClient failed: %w", err)
	}
	// audioClient will be released in the capture goroutine's defer or if an error occurs before goroutine starts

	var waveFormat *wca.WAVEFORMATEX
	// GetMixFormat expects a pointer to a pointer for the WAVEFORMATEX structure.
	err = audioClient.GetMixFormat(&waveFormat)
	if err != nil {
		audioClient.Release()
		return nil, nil, fmt.Errorf("GetMixFormat failed: %w", err)
	}
	if waveFormat == nil {
		audioClient.Release()
		return nil, nil, fmt.Errorf("GetMixFormat returned a nil waveFormat")
	}
	// CoTaskMemFree is in the 'ole' package.
	defer ole.CoTaskMemFree(uintptr(unsafe.Pointer(waveFormat))) // Free the WAVEFORMATEX structure

	log.Printf("Device Mix Format: SampleRate %d, Channels %d, BitsPerSample %d, FormatTag %d, BlockAlign %d",
		waveFormat.NSamplesPerSec, waveFormat.NChannels, waveFormat.WBitsPerSample, waveFormat.WFormatTag, waveFormat.NBlockAlign)

	isFloat := false
	// Use locally defined constants for wave format tags if not in wca.
	if waveFormat.WFormatTag == WAVE_FORMAT_IEEE_FLOAT {
		isFloat = true
		log.Println("Audio format is IEEE Float.")
	} else if waveFormat.WFormatTag == wca.WAVE_FORMAT_PCM {
		log.Println("Audio format is PCM.")
	} else if waveFormat.WFormatTag == WAVE_FORMAT_EXTENSIBLE {
		// For WAVE_FORMAT_EXTENSIBLE, the actual format is in a SubFormat GUID within WAVEFORMATEXTENSIBLE.
		// This simplified check assumes 32-bit extensible is float.
		if waveFormat.WBitsPerSample == 32 {
			isFloat = true // Common for extensible to be float32
			log.Println("Audio format is Extensible, assuming Float32 due to 32 bits/sample.")
		} else {
			log.Println("Audio format is Extensible, assuming PCM based on bits/sample != 32.")
		}
	} else {
		log.Printf("Unsupported audio format tag: %d. Proceeding with caution.", waveFormat.WFormatTag)
		// Potentially allow proceeding if WBitsPerSample makes sense for PCM/Float
		if waveFormat.WBitsPerSample == 32 {
			isFloat = true
			log.Println("Assuming Float for unknown 32-bit format.")
		} else if waveFormat.WBitsPerSample == 16 {
			isFloat = false
			log.Println("Assuming PCM for unknown 16-bit format.")
		} else {
			audioClient.Release()
			return nil, nil, fmt.Errorf("unsupported audio format tag: %d and ambiguous bit depth %d", waveFormat.WFormatTag, waveFormat.WBitsPerSample)
		}
	}

	captureConfig := &AudioCaptureConfig{
		SampleRate: waveFormat.NSamplesPerSec,
		Channels:   waveFormat.NChannels,
		BitDepth:   waveFormat.WBitsPerSample,
		IsFloat:    isFloat,
		NBlockAlign: waveFormat.NBlockAlign, // Store NBlockAlign here
		// Codec will be set in main.go after this function returns
	}
	// captureConfig.Codec = webrtc.RTPCodecCapability{MimeType: webrtc.MimeTypePCMS16LE, ClockRate: captureConfig.SampleRate, Channels: captureConfig.Channels} // Removed

	var hnsBufferDuration wca.REFERENCE_TIME = 20 * 10000 // 20 ms
	// Initialize expects format *WAVEFORMATEX and audioSessionGUID *ole.GUID (can be nil)
	err = audioClient.Initialize(wca.AUDCLNT_SHAREMODE_SHARED, wca.AUDCLNT_STREAMFLAGS_LOOPBACK, hnsBufferDuration, 0, waveFormat, nil)
	if err != nil {
		audioClient.Release()
		return nil, nil, fmt.Errorf("IAudioClient Initialize failed: %w", err)
	}
	log.Println("IAudioClient initialized for loopback capture.")

	var bufferFrameCount uint32 // Must be uint32 for GetBufferSize output
	err = audioClient.GetBufferSize(&bufferFrameCount)
	if err != nil {
		audioClient.Release()
		return nil, nil, fmt.Errorf("GetBufferSize failed: %w", err)
	}
	captureConfig.FramesPerBuffer = bufferFrameCount
	log.Printf("AudioClient buffer size: %d frames", bufferFrameCount)

	var captureClient *wca.IAudioCaptureClient
	// GetAudioCaptureClient is not standard; use GetService with IID_IAudioCaptureClient.
	err = audioClient.GetService(wca.IID_IAudioCaptureClient, &captureClient)
	if err != nil {
		audioClient.Release()
		return nil, nil, fmt.Errorf("GetService for IAudioCaptureClient failed: %w", err)
	}

	if err := audioClient.Start(); err != nil {
		captureClient.Release()
		audioClient.Release()
		return nil, nil, fmt.Errorf("AudioClient Start failed: %w", err)
	}
	log.Println("Audio capture started successfully.")

	go func() {
		defer func() {
			log.Println("Stopping audio client...")
			if err := audioClient.Stop(); err != nil {
				log.Printf("Failed to stop audio client: %v", err)
			}
			captureClient.Release()
			audioClient.Release()
			close(audioChan)
			log.Println("Audio capture goroutine finished and resources released.")
		}()

		// Use captureConfig.SampleRate which was safely copied from waveFormat.NSamplesPerSec
		// The bufferFrameCount is also part of captureConfig but it's also available directly.
		// No direct access to waveFormat needed here anymore for these values.
		if captureConfig.SampleRate == 0 {
			log.Println("captureConfig.SampleRate is 0 in capture goroutine, cannot calculate sleep duration. Exiting.")
			return
		}
		sleepDuration := time.Duration(float64(bufferFrameCount)/float64(captureConfig.SampleRate)*1000/2) * time.Millisecond
		if sleepDuration <= 0 {
			sleepDuration = 5 * time.Millisecond // Minimum sleep
		}
		log.Printf("Capture goroutine polling every %v", sleepDuration)

		for {
			time.Sleep(sleepDuration)

			var numFramesInNextPacket uint32
			err := captureClient.GetNextPacketSize(&numFramesInNextPacket)
			if err != nil {
				log.Printf("GetNextPacketSize failed: %v. Stopping capture.", err)
				return
			}

			if numFramesInNextPacket == 0 {
				continue
			}

			var pData *byte
			var numFramesAvailable uint32 = numFramesInNextPacket // This can be updated by GetBuffer
			var flags uint32
			var devicePosition, qpcPosition uint64 // Not used but required by API

			err = captureClient.GetBuffer(&pData, &numFramesAvailable, &flags, &devicePosition, &qpcPosition)
			if err != nil {
				log.Printf("IAudioCaptureClient.GetBuffer failed: %v. Stopping capture.", err)
				return
			}
			
			var audioData []byte
			if pData != nil && numFramesAvailable > 0 {
				// Calculate buffer size in bytes: numFramesAvailable * NBlockAlign
				// Use captureConfig.NBlockAlign which was safely copied.
				if captureConfig.NBlockAlign == 0 {
					log.Printf("captureConfig.NBlockAlign is 0. Skipping buffer processing.")
					captureClient.ReleaseBuffer(numFramesAvailable) // Release buffer even if not processed
					continue
				}
				bufferSizeInBytes := numFramesAvailable * uint32(captureConfig.NBlockAlign)
				audioData = unsafe.Slice(pData, int(bufferSizeInBytes))
			} else {
				// No data or pData is nil, but ReleaseBuffer still needs to be called
				captureClient.ReleaseBuffer(numFramesAvailable)
				continue
			}


			// Process data (e.g., convert if necessary, then send)
			if flags&wca.AUDCLNT_BUFFERFLAGS_SILENT != 0 {
				// log.Println("Silent packet received")
				// For silent packets, audioData might contain silence or be marked by flag.
				// If sending zeros, ensure it's the correct length.
			}

			chunkToSend := make([]byte, len(audioData))
			copy(chunkToSend, audioData)

			err = captureClient.ReleaseBuffer(numFramesAvailable) // framesRead for ReleaseBuffer
			if err != nil {
				log.Printf("ReleaseBuffer failed: %v. Stopping capture.", err)
				return
			}

			select {
			case audioChan <- chunkToSend:
			default:
				log.Println("Audio channel full, discarding packet.")
			}
		}
	}()

	return audioChan, captureConfig, nil
}