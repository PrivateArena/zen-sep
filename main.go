package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"runtime/pprof"

	"zen-sep/pkg/audio"
	"zen-sep/pkg/config"
	"zen-sep/pkg/separator"
)

func main() {
	pprofEnabled := flag.Bool("pprof", false, "Enable CPU profiling")
	modelType := flag.String("type", "mdxnet", "Model type: mdxnet, demucs, or spleeter")
	libPathFlag := flag.String("lib", "", "Path to libonnxruntime.so (overrides ONNXRUNTIME_LIB environment variable)")
	outputDir := flag.String("output-dir", "", "Directory to save output stems")
	configFile := flag.String("config", "", "Path to YAML configuration file")
	also16k := flag.Bool("also-16k", false, "Save a downsampled 16 kHz mono copy alongside outputs")
	residual := flag.Bool("residual", false, "Save the residual complement stem (input - separated)")
	device := flag.String("device", "cpu", "Execution provider device: cpu or cuda")
	targetLUFS := flag.Float64("target-lufs", 0, "Target EBU R128 integrated loudness in LUFS (e.g. -23)")
	vadModel := flag.String("vad-model", "", "Path to Silero VAD ONNX model for voice activity segmentation")
	inputDir := flag.String("input-dir", "", "Directory of WAV files for batch processing")

	flag.Usage = func() {
		fmt.Printf("Usage: %s [options] <input.wav/directory> <model.onnx>\n", os.Args[0])
		flag.PrintDefaults()
	}
	flag.Parse()

	args := flag.Args()
	// Model file is required as the last arg if not in input-dir mode, or generally we need at least 1 or 2 args.
	if *inputDir == "" && len(args) < 2 {
		flag.Usage()
		return
	} else if *inputDir != "" && len(args) < 1 {
		fmt.Println("Error: Model file (.onnx) must be specified when using --input-dir")
		flag.Usage()
		return
	}

	var inputFile string
	var modelFile string
	if *inputDir != "" {
		modelFile = args[0]
	} else {
		inputFile = args[0]
		modelFile = args[1]
	}

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

	// 1.1 Expose and detect libonnxruntime path
	libPath := *libPathFlag
	if libPath == "" {
		libPath = os.Getenv("ONNXRUNTIME_LIB")
	}
	if libPath == "" {
		libPath = "/usr/lib/libonnxruntime.so" // Reasonable default
	}

	// 3.1 YAML Config support
	var cfg *separator.Config
	if *configFile != "" {
		var err error
		cfg, err = config.Load(*configFile)
		if err != nil {
			log.Fatalf("Failed to load YAML config: %v", err)
		}
	} else {
		// Fallback to hard-coded defaults
		switch *modelType {
		case "demucs":
			cfg = &separator.Config{
				Type:       separator.ModelTypeDemucs,
				SampleRate: 44100,
				ChunkSize:  343980,
				Stems:      []string{"drums", "bass", "other", "vocals"},
			}
		case "spleeter":
			cfg = &separator.Config{
				Type:       separator.ModelTypeSpleeter,
				SampleRate: 44100,
				N_FFT:      4096,
				HopSize:    1024,
				ChunkSize:  1024,
				Stems:      []string{"vocals"},
			}
		default:
			cfg = &separator.Config{
				Type:      separator.ModelTypeMDXNet,
				N_FFT:     5120,
				HopSize:   1280,
				ChunkSize: 256,
				FreqBins:  2561,
				Stems:     []string{"vocals", "instrumental"},
			}
		}
	}

	// Override config with command-line flags if provided
	if *outputDir != "" {
		cfg.OutputDir = *outputDir
	}
	if *also16k {
		cfg.Also16k = true
	}
	if *residual {
		cfg.Residual = true
	}
	if *device != "" {
		cfg.Device = *device
	}
	if *targetLUFS != 0 {
		cfg.TargetLUFS = *targetLUFS
	}
	if *vadModel != "" {
		cfg.VADModel = *vadModel
	}

	fmt.Printf("Loading model (%s): %s\n", cfg.Type, modelFile)
	sep, err := separator.NewSeparator(modelFile, libPath, *cfg)
	if err != nil {
		log.Fatalf("Failed to init separator: %v", err)
	}
	defer sep.Session.Destroy()

	// 3.2 Batch processing mode
	var files []string
	if *inputDir != "" {
		entries, err := os.ReadDir(*inputDir)
		if err != nil {
			log.Fatalf("Failed to read input directory: %v", err)
		}
		for _, entry := range entries {
			if !entry.IsDir() && filepath.Ext(entry.Name()) == ".wav" {
				files = append(files, filepath.Join(*inputDir, entry.Name()))
			}
		}
		if len(files) == 0 {
			log.Fatalf("No WAV files found in directory: %s", *inputDir)
		}
		fmt.Printf("Batch processing %d files from directory: %s\n", len(files), *inputDir)
	} else {
		files = []string{inputFile}
	}

	for idx, file := range files {
		fmt.Printf("[%d/%d] Processing file: %s\n", idx+1, len(files), file)
		data, channels, sr, err := audio.LoadWav(file)
		if err != nil {
			log.Printf("Failed to load wav %s: %v", file, err)
			continue
		}

		if channels != 2 {
			log.Printf("Skipping %s: Model expects stereo (2 channels) audio", file)
			continue
		}

		// Update separator config dynamically with true source sample rate
		sep.Config.SampleRate = sr

		if err := sep.Separate(data, file); err != nil {
			log.Printf("Failed to separate audio %s: %v", file, err)
		}
	}
}
