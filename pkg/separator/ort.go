package separator

import (
	"fmt"
	ort "github.com/yalue/onnxruntime_go"
)

type Separator struct {
	Session *ort.DynamicSession[float32, float32]
	Inputs  []ort.InputOutputInfo
	Outputs []ort.InputOutputInfo
	Shape   []int64
	Opts    *ort.SessionOptions
}

func NewSeparator(modelPath string, libPath string) (*Separator, error) {
	ort.SetSharedLibraryPath(libPath)
	err := ort.InitializeEnvironment()
	if err != nil {
		return nil, fmt.Errorf("failed to init ORT: %w", err)
	}

	// Performance optimization: Multi-threading and Intra-Op optimization
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("failed to create session options: %w", err)
	}
	// Let ORT use half of the available CPUs to prevent 100% saturation and leave room for audio I/O
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

	shape := inputs[0].Dimensions
	for i, v := range shape {
		if v <= 0 {
			if i == 0 { shape[i] = 1 }
			if i == 1 { shape[i] = 2 } // standard
			if i == 3 { shape[i] = 256 }
		}
	}

	return &Separator{
		Session: session,
		Inputs:  inputs,
		Outputs: outputs,
		Shape:   shape,
		Opts:    opts,
	}, nil
}

func (s *Separator) RunInference(inputData []float32) ([]float32, error) {
	inputShape := ort.NewShape(s.Shape...)
	inputTensor, err := ort.NewTensor(inputShape, inputData)
	if err != nil {
		return nil, fmt.Errorf("failed to create input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	outputShape := ort.NewShape(s.Shape...)
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return nil, fmt.Errorf("failed to create output tensor: %w", err)
	}
	defer outputTensor.Destroy()

	err = s.Session.Run([]*ort.Tensor[float32]{inputTensor}, []*ort.Tensor[float32]{outputTensor})
	if err != nil {
		return nil, fmt.Errorf("inference failed: %w", err)
	}

	return outputTensor.GetData(), nil
}
