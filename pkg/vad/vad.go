package vad

import (
	"fmt"

	"zen-sep/pkg/audio"

	ort "github.com/yalue/onnxruntime_go"
)

// Segment represents a speech activity interval.
type Segment struct {
	StartSec float64 `json:"start_sec"`
	EndSec   float64 `json:"end_sec"`
}

// Detector wraps the Silero VAD ONNX model session.
type Detector struct {
	session *ort.DynamicAdvancedSession
	opts    *ort.SessionOptions
}

// NewDetector initializes a new Silero VAD detector.
func NewDetector(modelPath string, libPath string) (*Detector, error) {
	ort.SetSharedLibraryPath(libPath)
	_ = ort.InitializeEnvironment()

	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("failed to create VAD session options: %w", err)
	}

	_ = opts.SetIntraOpNumThreads(1)
	_ = opts.SetInterOpNumThreads(1)

	inputs, outputs, err := ort.GetInputOutputInfo(modelPath)
	if err != nil {
		opts.Destroy()
		return nil, fmt.Errorf("failed to get VAD model info: %w", err)
	}

	inputNames := make([]string, len(inputs))
	for i, in := range inputs {
		inputNames[i] = in.Name
	}
	outputNames := make([]string, len(outputs))
	for i, out := range outputs {
		outputNames[i] = out.Name
	}

	session, err := ort.NewDynamicAdvancedSession(modelPath, inputNames, outputNames, opts)
	if err != nil {
		opts.Destroy()
		return nil, fmt.Errorf("failed to create VAD session: %w", err)
	}

	return &Detector{
		session: session,
		opts:    opts,
	}, nil
}

// Close cleans up VAD model session resources.
func (d *Detector) Close() {
	if d.session != nil {
		d.session.Destroy()
	}
	if d.opts != nil {
		d.opts.Destroy()
	}
}

// Detect segments the audio data based on speech activity detection (VAD).
// Silero VAD operates at 16000 Hz. If the input rate is different, it resamples.
func (d *Detector) Detect(pcm []float32, originalRate int, channels int) ([]Segment, error) {
	// 1. Downmix to Mono if stereo/multichannel
	var mono []float32
	if channels > 1 {
		mono = make([]float32, len(pcm)/channels)
		for i := 0; i < len(mono); i++ {
			var sum float64
			for c := 0; c < channels; c++ {
				sum += float64(pcm[i*channels+c])
			}
			mono[i] = float32(sum / float64(channels))
		}
	} else {
		mono = pcm
	}

	// 2. Resample to 16000 Hz if necessary
	targetRate := 16000
	var samples16k []float32
	if originalRate != targetRate {
		samples16k = audio.Resample(mono, originalRate, targetRate)
	} else {
		samples16k = mono
	}

	// Silero VAD expects a window size of 512, 1024, or 1536 samples at 16000 Hz.
	// 512 samples corresponds to 32 ms.
	windowSize := 512
	totalSamples := len(samples16k)
	
	// Create required GRU state tensor: shape [2, 1, 64], initialized to 0
	stateShape := ort.NewShape(2, 1, 64)
	stateBuf := make([]float32, 2*1*64)
	stateTensor, err := ort.NewTensor(stateShape, stateBuf)
	if err != nil {
		return nil, fmt.Errorf("failed to create VAD state tensor: %w", err)
	}
	defer stateTensor.Destroy()

	// Create sample rate tensor: shape [1], value [16000]
	srShape := ort.NewShape(1)
	srTensor, err := ort.NewTensor(srShape, []int64{16000})
	if err != nil {
		return nil, fmt.Errorf("failed to create VAD sample rate tensor: %w", err)
	}
	defer srTensor.Destroy()

	threshold := float32(0.5)
	minSpeechDurSec := 0.250 // min duration of speech segment to register (250ms)
	minSilenceDurSec := 0.350 // speech segment ends after 350ms of silence
	
	minSpeechSamples := int(minSpeechDurSec * float64(targetRate))
	minSilenceSamples := int(minSilenceDurSec * float64(targetRate))

	var segments []Segment
	isSpeech := false
	speechStartSample := -1
	silenceStartSample := -1

	// Iterate audio in windows of size 512
	for start := 0; start <= totalSamples-windowSize; start += windowSize {
		windowBuf := samples16k[start : start+windowSize]
		
		inputShape := ort.NewShape(1, int64(windowSize))
		inputTensor, err := ort.NewTensor(inputShape, windowBuf)
		if err != nil {
			return nil, fmt.Errorf("failed to create VAD input tensor: %w", err)
		}

		// Prepare output tensors
		probShape := ort.NewShape(1, 1)
		probTensor, err := ort.NewEmptyTensor[float32](probShape)
		if err != nil {
			inputTensor.Destroy()
			return nil, fmt.Errorf("failed to create VAD prob tensor: %w", err)
		}

		nextStateTensor, err := ort.NewEmptyTensor[float32](stateShape)
		if err != nil {
			inputTensor.Destroy()
			probTensor.Destroy()
			return nil, fmt.Errorf("failed to create VAD next state tensor: %w", err)
		}

		err = d.session.Run(
			[]ort.Value{inputTensor, srTensor, stateTensor},
			[]ort.Value{probTensor, nextStateTensor},
		)
		inputTensor.Destroy()

		if err != nil {
			probTensor.Destroy()
			nextStateTensor.Destroy()
			return nil, fmt.Errorf("VAD inference failed: %w", err)
		}

		prob := probTensor.GetData()[0]
		probTensor.Destroy()

		// Update state for next iteration
		copy(stateBuf, nextStateTensor.GetData())
		nextStateTensor.Destroy()

		// Speech detection logic
		if prob >= threshold {
			if !isSpeech {
				isSpeech = true
				speechStartSample = start
			}
			silenceStartSample = -1
		} else {
			if isSpeech {
				if silenceStartSample == -1 {
					silenceStartSample = start
				}
				// If silent for longer than minSilenceDurSec, close the segment
				if start-silenceStartSample >= minSilenceSamples {
					durationSamples := silenceStartSample - speechStartSample
					if durationSamples >= minSpeechSamples {
						segments = append(segments, Segment{
							StartSec: float64(speechStartSample) / float64(targetRate),
							EndSec:   float64(silenceStartSample) / float64(targetRate),
						})
					}
					isSpeech = false
					speechStartSample = -1
					silenceStartSample = -1
				}
			}
		}
	}

	// Handle case where audio ends during active speech
	if isSpeech && speechStartSample != -1 {
		endSample := totalSamples
		if silenceStartSample != -1 {
			endSample = silenceStartSample
		}
		durationSamples := endSample - speechStartSample
		if durationSamples >= minSpeechSamples {
			segments = append(segments, Segment{
				StartSec: float64(speechStartSample) / float64(targetRate),
				EndSec:   float64(endSample) / float64(targetRate),
			})
		}
	}

	return segments, nil
}
