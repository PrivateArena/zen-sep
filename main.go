package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"runtime/pprof"

	"zen-separator/pkg/audio"
	"zen-separator/pkg/separator"
)

func main() {
	pprofEnabled := flag.Bool("pprof", false, "Enable CPU profiling")
	modelType := flag.String("type", "mdxnet", "Model type: mdxnet, demucs, or spleeter")
	flag.Usage = func() {
		fmt.Printf("Usage: %s [options] <input.wav> <model.onnx>\n", os.Args[0])
		flag.PrintDefaults()
	}
	flag.Parse()

	args := flag.Args()
	if len(args) < 2 {
		flag.Usage()
		return
	}

	inputFile := args[0]
	modelFile := args[1]

	if *pprofEnabled {
		fmt.Println("🚀 CPU Profiling Enabled...")
		f, err := os.Create("cpu.prof")
		if err != nil {
			log.Fatal("could not create CPU profile: ", err)
		}
		defer f.Close()
		if err := pprof.StartCPUProfile(f); err != nil {
			log.Fatal("could not start CPU profile: ", err)
		}
		defer pprof.StopCPUProfile()
	}

	libPath := "/media/jang/home/Deve/transcribe-models/onnxruntime-linux-x64-1.24.2/lib/libonnxruntime.so.1.24.2"

	// Default MDX-Net Config
	config := separator.Config{
		Type:      separator.ModelTypeMDXNet,
		N_FFT:     5120,
		HopSize:   1280,
		ChunkSize: 256,
		FreqBins:  2561, // (n_fft / 2) + 1
		Stems:     []string{"vocals", "instrumental"},
	}

	if *modelType == "demucs" {
		config = separator.Config{
			Type:       separator.ModelTypeDemucs,
			SampleRate: 44100,
			ChunkSize:  343980,
			Stems:      []string{"drums", "bass", "other", "vocals"},
		}
	} else if *modelType == "spleeter" {
		config = separator.Config{
			Type:       separator.ModelTypeSpleeter,
			SampleRate: 44100,
			N_FFT:      4096,
			HopSize:    1024,
			ChunkSize:  1024, // Frames
			Stems:      []string{"vocals"},
		}
	}

	fmt.Printf("Loading model (%s): %s\n", config.Type, modelFile)
	sep, err := separator.NewSeparator(modelFile, libPath, config)
	if err != nil {
		log.Fatalf("Failed to init separator: %v", err)
	}
	defer sep.Session.Destroy()

	fmt.Printf("Processing file: %s\n", inputFile)
	data, channels, err := audio.LoadWav(inputFile)
	if err != nil {
		log.Fatalf("Failed to load wav: %v", err)
	}

	if channels != 2 {
		log.Fatal("Model expects stereo (2 channels) audio")
	}

	if err := sep.Separate(data, inputFile); err != nil {
		log.Fatalf("Failed to separate audio: %v", err)
	}
}
