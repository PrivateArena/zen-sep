package separator

import (
	"fmt"
	"runtime"

	ort "github.com/yalue/onnxruntime_go"
)

// ModelType defines the architecture of the separation model
type ModelType string

const (
	ModelTypeMDXNet   ModelType = "mdxnet"
	ModelTypeDemucs   ModelType = "demucs"
	ModelTypeSpleeter ModelType = "spleeter"
)

type Config struct {
	Type       ModelType `json:"type"`
	SampleRate int       `json:"sample_rate"`
	HopSize    int       `json:"hop_size"`
	N_FFT      int       `json:"n_fft"`
	ChunkSize  int       `json:"chunk_size"`
	FreqBins   int       `json:"freq_bins"`
	Overlap    float32   `json:"overlap"`
	Stems      []string  `json:"stems"`
}

type Separator struct {
	// Fix #3: use DynamicAdvancedSession so SessionOptions are actually applied.
	Session *ort.DynamicAdvancedSession
	Config  Config
	Inputs  []ort.InputOutputInfo
	Outputs []ort.InputOutputInfo
	Opts    *ort.SessionOptions
}

func NewSeparator(modelPath string, libPath string, config Config) (*Separator, error) {
	ort.SetSharedLibraryPath(libPath)
	err := ort.InitializeEnvironment()
	if err != nil {
		// Environment might already be initialized in some contexts
	}

	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("failed to create session options: %w", err)
	}

	// Use all logical CPUs for intra-op parallelism (matrix math inside a single op).
	// Cap at 8 to avoid scheduler overhead on very wide CPUs.
	numThreads := runtime.NumCPU()
	if numThreads > 8 {
		numThreads = 8
	}
	_ = opts.SetIntraOpNumThreads(numThreads)
	// Inter-op parallelism: 1 is usually best unless the model has independent branches.
	_ = opts.SetInterOpNumThreads(1)

	inputs, outputs, err := ort.GetInputOutputInfo(modelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to get model info: %w", err)
	}

	inputNames := make([]string, len(inputs))
	for i, in := range inputs {
		inputNames[i] = in.Name
	}
	outputNames := make([]string, len(outputs))
	for i, out := range outputs {
		outputNames[i] = out.Name
	}

	// Fix #3: NewDynamicAdvancedSession forwards opts to the C OrtSession,
	// so thread-count settings are actually applied during inference.
	session, err := ort.NewDynamicAdvancedSession(modelPath, inputNames, outputNames, opts)
	if err != nil {
		return nil, fmt.Errorf("failed to create session: %w", err)
	}

	return &Separator{
		Session: session,
		Config:  config,
		Inputs:  inputs,
		Outputs: outputs,
		Opts:    opts,
	}, nil
}

// RunInference runs the ONNX model. Accepts and returns ort.Value slices so
// callers can pass *Tensor[float32] directly (it implements ort.Value).
func (s *Separator) RunInference(inputs []ort.Value, outputs []ort.Value) error {
	return s.Session.Run(inputs, outputs)
}
