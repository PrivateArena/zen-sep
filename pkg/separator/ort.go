package separator

import (
	"fmt"
	ort "github.com/yalue/onnxruntime_go"
)

// ModelType defines the architecture of the separation model
type ModelType string

const (
	ModelTypeMDXNet ModelType = "mdxnet"
	ModelTypeDemucs ModelType = "demucs"
	ModelTypeSpleeter ModelType = "spleeter"
)

type Config struct {
	Type         ModelType `json:"type"`
	SampleRate   int       `json:"sample_rate"`
	HopSize      int       `json:"hop_size"`
	N_FFT        int       `json:"n_fft"`
	ChunkSize    int       `json:"chunk_size"`
	FreqBins     int       `json:"freq_bins"`
	Overlap      float32   `json:"overlap"`
	Stems        []string  `json:"stems"`
}

type Separator struct {
	Session *ort.DynamicSession[float32, float32]
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
	_ = opts.SetIntraOpNumThreads(4)
	_ = opts.SetInterOpNumThreads(4)

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

	session, err := ort.NewDynamicSession[float32, float32](modelPath, inputNames, outputNames)
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

func (s *Separator) RunInference(inputs []*ort.Tensor[float32], outputs []*ort.Tensor[float32]) error {
	return s.Session.Run(inputs, outputs)
}
