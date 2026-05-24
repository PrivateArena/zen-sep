package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"math/cmplx"
	"os"
	"path/filepath"
	"runtime/pprof"
	"sync"
	"sync/atomic"

	"zen-separator/pkg/audio"
	"zen-separator/pkg/separator"
	"github.com/schollz/progressbar/v3"
	ort "github.com/yalue/onnxruntime_go"
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

	if config.Type == separator.ModelTypeMDXNet {
		processMDXNet(sep, data, inputFile)
	} else if config.Type == separator.ModelTypeDemucs {
		processDemucs(sep, data, inputFile)
	} else if config.Type == separator.ModelTypeSpleeter {
		processSpleeter(sep, data, inputFile)
	}
}

func processMDXNet(sep *separator.Separator, data []float32, inputFile string) {
	n_fft := sep.Config.N_FFT
	hopSize := sep.Config.HopSize
	freqBins := sep.Config.FreqBins
	chunkSize := sep.Config.ChunkSize
	numSamples := len(data) / 2

	// De-interleave
	leftChannel := make([]float32, numSamples)
	rightChannel := make([]float32, numSamples)
	for i := 0; i < numSamples; i++ {
		leftChannel[i] = data[i*2]
		rightChannel[i] = data[i*2+1]
	}

	fmt.Println("Getting STFT (Parallel)...")
	var stftLeft, stftRight [][]complex128
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		stftLeft = audio.GetSTFT(leftChannel, n_fft, hopSize)
	}()
	go func() {
		defer wg.Done()
		stftRight = audio.GetSTFT(rightChannel, n_fft, hopSize)
	}()
	wg.Wait()

	numFrames := len(stftLeft)
	outputStftLeft := make([][]complex128, numFrames)
	outputStftRight := make([][]complex128, numFrames)
	for i := range outputStftLeft {
		// Fix #7: allocate freqBins, not n_fft (was 2x too large)
		outputStftLeft[i] = make([]complex128, freqBins)
		outputStftRight[i] = make([]complex128, freqBins)
	}

	sem := make(chan struct{}, 4)
	var progress int32

	for startFrame := 0; startFrame < numFrames; startFrame += chunkSize / 2 {
		wg.Add(1)
		go func(frame int) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			endFrame := frame + chunkSize
			if endFrame > numFrames {
				return
			}

			inputBuf := make([]float32, 1*2*freqBins*chunkSize)
			for t := 0; t < chunkSize; t++ {
				fIdx := frame + t
				for f := 0; f < freqBins; f++ {
					inputBuf[0*freqBins*chunkSize+f*chunkSize+t] = float32(cmplx.Abs(stftLeft[fIdx][f]))
					inputBuf[1*freqBins*chunkSize+f*chunkSize+t] = float32(cmplx.Abs(stftRight[fIdx][f]))
				}
			}

			inputShape := ort.NewShape(1, 2, int64(freqBins), int64(chunkSize))
			inputTensor, _ := ort.NewTensor(inputShape, inputBuf)
			defer inputTensor.Destroy()

			outputBuf := make([]float32, 1*2*freqBins*chunkSize)
			outputTensor, _ := ort.NewEmptyTensor[float32](inputShape)
			defer outputTensor.Destroy()

			err := sep.RunInference(
				[]ort.Value{inputTensor},
				[]ort.Value{outputTensor},
			)
			if err != nil {
				log.Printf("Inference failed: %v", err)
				return
			}
			outputBuf = outputTensor.GetData()

			for t := 0; t < chunkSize; t++ {
				fIdx := frame + t
				for f := 0; f < freqBins; f++ {
					maskL := outputBuf[0*freqBins*chunkSize+f*chunkSize+t]
					maskR := outputBuf[1*freqBins*chunkSize+f*chunkSize+t]
					outputStftLeft[fIdx][f] = cmplx.Rect(float64(maskL), cmplx.Phase(stftLeft[fIdx][f]))
					outputStftRight[fIdx][f] = cmplx.Rect(float64(maskR), cmplx.Phase(stftRight[fIdx][f]))
				}
			}
			atomic.AddInt32(&progress, int32(chunkSize/2))
		}(startFrame)
	}
	wg.Wait()

	fmt.Println("Reconstructing audio...")
	outLeft := audio.GetISTFT(outputStftLeft, n_fft, hopSize, numSamples)
	outRight := audio.GetISTFT(outputStftRight, n_fft, hopSize, numSamples)

	saveStereo(outLeft, outRight, inputFile, "instrumental")
}

func processDemucs(sep *separator.Separator, data []float32, inputFile string) {
	chunkSize := sep.Config.ChunkSize // 343980 for htdemucs_ort_v1
	numSamples := len(data) / 2
	numStems := len(sep.Config.Stems)

	// htdemucs_ort_v1 needs "input" [1, 2, 343980] and "x" [1, 4, 2048, 336]
	// "x" is the STFT of the input.
	demucs_n_fft := 4096
	demucs_hop := 1024

	// Fix #2: De-interleave full track once and compute STFT before the loop.
	// Previously GetSTFT was called inside the sliding-window loop, causing
	// 4× redundant computation due to 75% overlap between consecutive chunks.
	leftFull := make([]float32, numSamples)
	rightFull := make([]float32, numSamples)
	for i := 0; i < numSamples; i++ {
		leftFull[i] = data[i*2]
		rightFull[i] = data[i*2+1]
	}
	fmt.Println("Getting STFT (Demucs, full track)...")
	stftFullL := audio.GetSTFT(leftFull, demucs_n_fft, demucs_hop)
	stftFullR := audio.GetSTFT(rightFull, demucs_n_fft, demucs_hop)

	// Sliding window with 25% overlap
	stepSize := chunkSize * 3 / 4
	numChunks := (numSamples + stepSize - 1) / stepSize

	// Output buffers for each stem
	stemOutputs := make([][]float32, numStems)
	stemWeights := make([]float32, numSamples)
	for i := range stemOutputs {
		stemOutputs[i] = make([]float32, numSamples*2)
	}

	xFreqBins := 2048
	xFrames := 336

	bar := progressbar.Default(int64(numChunks), "Separating (Demucs)")

	for start := 0; start < numSamples; start += stepSize {
		// Prepare waveform input
		inputWav := make([]float32, 2*chunkSize)
		for i := 0; i < chunkSize; i++ {
			idx := start + i
			if idx < numSamples {
				inputWav[i] = data[idx*2]
				inputWav[chunkSize+i] = data[idx*2+1]
			}
		}

		inputWavShape := ort.NewShape(1, 2, int64(chunkSize))
		inputWavTensor, _ := ort.NewTensor(inputWavShape, inputWav)

		// Fix #2: slice the precomputed full-track STFT for this chunk's frame range.
		inputX := make([]float32, 4*xFreqBins*xFrames)
		frameStart := start / demucs_hop
		for t := 0; t < xFrames; t++ {
			fIdx := frameStart + t
			if fIdx >= len(stftFullL) {
				break
			}
			for f := 0; f < xFreqBins; f++ {
				inputX[0*xFreqBins*xFrames+f*xFrames+t] = float32(real(stftFullL[fIdx][f]))
				inputX[1*xFreqBins*xFrames+f*xFrames+t] = float32(imag(stftFullL[fIdx][f]))
				inputX[2*xFreqBins*xFrames+f*xFrames+t] = float32(real(stftFullR[fIdx][f]))
				inputX[3*xFreqBins*xFrames+f*xFrames+t] = float32(imag(stftFullR[fIdx][f]))
			}
		}

		inputXShape := ort.NewShape(1, 4, int64(xFreqBins), int64(xFrames))
		inputXTensor, _ := ort.NewTensor(inputXShape, inputX)

		// htdemucs_ort might have 2 outputs: [1, 4, 2, 343980] and some other metadata
		outputTensors := make([]*ort.Tensor[float32], len(sep.Outputs))
		for i, out := range sep.Outputs {
			outShape := ort.NewShape(out.Dimensions...)
			outputTensors[i], _ = ort.NewEmptyTensor[float32](outShape)
		}

		inputVals := []ort.Value{inputWavTensor, inputXTensor}
		outputVals := make([]ort.Value, len(outputTensors))
		for i, t := range outputTensors {
			outputVals[i] = t
		}
		err := sep.RunInference(inputVals, outputVals)

		// Fix #4: explicit Destroy at end of loop body instead of defer.
		inputWavTensor.Destroy()
		inputXTensor.Destroy()

		if err != nil {
			log.Printf("Inference error at %d: %v", start, err)
			for _, t := range outputTensors {
				t.Destroy()
			}
			bar.Add(1)
			continue
		}

		// Use the first output (primary stems)
		outData := outputTensors[0].GetData()
		for i := 0; i < chunkSize; i++ {
			idx := start + i
			if idx >= numSamples {
				break
			}

			// Triangular weight
			distEdge := float32(i)
			if float32(chunkSize-i) < distEdge {
				distEdge = float32(chunkSize - i)
			}
			w := distEdge / float32(chunkSize/8)
			if w > 1.0 {
				w = 1.0
			}
			if w < 1e-4 {
				w = 1e-4
			}

			for s := 0; s < numStems; s++ {
				lVal := outData[s*2*chunkSize+0*chunkSize+i]
				rVal := outData[s*2*chunkSize+1*chunkSize+i]
				stemOutputs[s][idx*2] += lVal * w
				stemOutputs[s][idx*2+1] += rVal * w
			}
			stemWeights[idx] += w
		}
		for _, t := range outputTensors {
			t.Destroy()
		}
		bar.Add(1)
	}

	for s, stemName := range sep.Config.Stems {
		out := make([]float32, numSamples*2)
		for i := 0; i < numSamples; i++ {
			if stemWeights[i] > 1e-5 {
				out[i*2] = stemOutputs[s][i*2] / stemWeights[i]
				out[i*2+1] = stemOutputs[s][i*2+1] / stemWeights[i]
			}
		}
		saveStereoFull(out, inputFile, stemName)
	}
}

func processSpleeter(sep *separator.Separator, data []float32, inputFile string) {
	n_fft := 4096
	hopSize := 1024
	numSamples := len(data) / 2

	// De-interleave
	left := make([]float32, numSamples)
	right := make([]float32, numSamples)
	for i := 0; i < numSamples; i++ {
		left[i] = data[i*2]
		right[i] = data[i*2+1]
	}

	fmt.Println("Getting STFT (Parallel)...")
	var stftL, stftR [][]complex128
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		stftL = audio.GetSTFT(left, n_fft, hopSize)
	}()
	go func() {
		defer wg.Done()
		stftR = audio.GetSTFT(right, n_fft, hopSize)
	}()
	wg.Wait()

	numFrames := len(stftL)
	// Spleeter (Sherpa-ONNX) expects [2, T, 512, 1024]
	// T is num_splits, 512 is freq bins (first 511 + 1?), 1024 is frames per split?
	// Actually, let's look at the shape again: [2, num_splits, 512, 1024]
	// If 1024 is frames per split, and num_splits = numFrames / 1024.
	// 512 is likely the first 512 bins of the 2049 bins (from 4096 FFT).
	
	splitSize := 1024
	numSplits := (numFrames + splitSize - 1) / splitSize
	freqBins := 512

	inputX := make([]float32, 2*numSplits*freqBins*splitSize)
	for split := 0; split < numSplits; split++ {
		for t := 0; t < splitSize; t++ {
			fIdx := split*splitSize + t
			if fIdx >= numFrames { break }
			for f := 0; f < freqBins; f++ {
				// Spleeter models typically use magnitude STFT
				inputX[0*numSplits*freqBins*splitSize + split*freqBins*splitSize + f*splitSize + t] = float32(cmplx.Abs(stftL[fIdx][f]))
				inputX[1*numSplits*freqBins*splitSize + split*freqBins*splitSize + f*splitSize + t] = float32(cmplx.Abs(stftR[fIdx][f]))
			}
		}
	}

	inputShape := ort.NewShape(2, int64(numSplits), int64(freqBins), int64(splitSize))
	inputTensor, _ := ort.NewTensor(inputShape, inputX)
	defer inputTensor.Destroy()

	outputTensors := make([]*ort.Tensor[float32], len(sep.Outputs))
	for i := range sep.Outputs {
		// Output shape [2, num_splits, 512, 1024]
		outShape := ort.NewShape(2, int64(numSplits), int64(freqBins), int64(splitSize))
		outputTensors[i], _ = ort.NewEmptyTensor[float32](outShape)
		defer outputTensors[i].Destroy()
	}

	fmt.Println("Running Spleeter Inference...")
	inputVals := []ort.Value{inputTensor}
	outputVals := make([]ort.Value, len(outputTensors))
	for i, t := range outputTensors {
		outputVals[i] = t
	}
	err := sep.RunInference(inputVals, outputVals)
	if err != nil {
		log.Fatalf("Spleeter inference failed: %v", err)
	}

	outData := outputTensors[0].GetData()
	outputStftL := make([][]complex128, numFrames)
	outputStftR := make([][]complex128, numFrames)
	for i := range outputStftL {
		outputStftL[i] = make([]complex128, n_fft/2+1)
		outputStftR[i] = make([]complex128, n_fft/2+1)
	}

	for split := 0; split < numSplits; split++ {
		for t := 0; t < splitSize; t++ {
			fIdx := split*splitSize + t
			if fIdx >= numFrames { break }
			for f := 0; f < freqBins; f++ {
				maskL := outData[0*numSplits*freqBins*splitSize + split*freqBins*splitSize + f*splitSize + t]
				maskR := outData[1*numSplits*freqBins*splitSize + split*freqBins*splitSize + f*splitSize + t]
				
				// Apply mask to original complex STFT
				outputStftL[fIdx][f] = cmplx.Rect(float64(maskL), cmplx.Phase(stftL[fIdx][f]))
				outputStftR[fIdx][f] = cmplx.Rect(float64(maskR), cmplx.Phase(stftR[fIdx][f]))
			}
			// Bins beyond 512 are zeroed out (standard Spleeter behavior for 11kHz cutoff)
		}
	}

	fmt.Println("Reconstructing Spleeter audio...")
	outLeft := audio.GetISTFT(outputStftL, n_fft, hopSize, numSamples)
	outRight := audio.GetISTFT(outputStftR, n_fft, hopSize, numSamples)

	saveStereo(outLeft, outRight, inputFile, "vocals")
}

func saveStereoFull(data []float32, originalPath, suffix string) {
	outputFile := filepath.Join(filepath.Dir(originalPath), suffix+"_"+filepath.Base(originalPath))
	_ = audio.SaveWav(outputFile, data, 44100, 2)
	fmt.Printf("\rSaved: %-40s\n", outputFile)
}

func saveStereo(left, right []float32, originalPath, suffix string) {
	numSamples := len(left)
	outInterleaved := make([]float32, numSamples*2)
	maxPeak := float32(0.0)
	for i := 0; i < numSamples; i++ {
		outInterleaved[i*2] = left[i]
		outInterleaved[i*2+1] = right[i]
		if math.Abs(float64(left[i])) > float64(maxPeak) { maxPeak = float32(math.Abs(float64(left[i]))) }
		if math.Abs(float64(right[i])) > float64(maxPeak) { maxPeak = float32(math.Abs(float64(right[i]))) }
	}

	gain := float32(1.0)
	if maxPeak > 0 { gain = 0.99 / maxPeak }
	for i := range outInterleaved { outInterleaved[i] *= gain }

	outputFile := filepath.Join(filepath.Dir(originalPath), suffix+"_"+filepath.Base(originalPath))
	_ = audio.SaveWav(outputFile, outInterleaved, 44100, 2)
	fmt.Printf("Saved: %s\n", outputFile)
}
