package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"math/cmplx"
	"os"
	"path/filepath"
	"runtime"
	"runtime/pprof"
	"sync"
	"sync/atomic"

	"zen-separator/pkg/audio"
	"zen-separator/pkg/separator"
)

func main() {
	pprofEnabled := flag.Bool("pprof", false, "Enable CPU profiling")
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

	fmt.Printf("Loading model: %s\n", modelFile)
	sep, err := separator.NewSeparator(modelFile, libPath)
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

	// Align with UVR5 MDX-Net defaults for 2560-bin models
	chns := int(sep.Shape[1])
	freqBins := int(sep.Shape[2])
	chunkSize := int(sep.Shape[3])

	// N_FFT = (dim_f - 1) * 2 or exactly 5120 for many HQ models
	n_fft := 5120
	hopSize := n_fft / 4 // 1280

	fmt.Printf("Engine Config: n_fft=%d, hop=%d, channels=%d, chunkSize=%d\n", n_fft, hopSize, chns, chunkSize)

	// Interleaved to de-interleaved
	numSamples := len(data) / 2
	leftChannel := make([]float32, numSamples)
	rightChannel := make([]float32, numSamples)
	for i := 0; i < numSamples; i++ {
		leftChannel[i] = data[i*2]
		rightChannel[i] = data[i*2+1]
	}

	var stftLeft, stftRight [][]complex128
	var outputStftLeft, outputStftRight [][]complex128

	fmt.Println("Getting STFT (Parallel)...")
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
	outputStftLeft = make([][]complex128, numFrames)
	outputStftRight = make([][]complex128, numFrames)
	for i := range outputStftLeft {
		outputStftLeft[i] = make([]complex128, n_fft)
		outputStftRight[i] = make([]complex128, n_fft)
	}

	fmt.Printf("Processing %d frames (Parallel Workers: %d)...\n", numFrames, runtime.NumCPU())

	// Worker pool for Inference
	// We limit to 4 parallel inference calls at once because each ORT call is internally multi-threaded
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

			inputBuf := make([]float32, 1*chns*freqBins*chunkSize)
			for t := 0; t < chunkSize; t++ {
				fIdx := frame + t
				for f := 0; f < freqBins; f++ {
					if chns == 4 {
						inputBuf[0*freqBins*chunkSize+f*chunkSize+t] = float32(real(stftLeft[fIdx][f]))
						inputBuf[1*freqBins*chunkSize+f*chunkSize+t] = float32(imag(stftLeft[fIdx][f]))
						inputBuf[2*freqBins*chunkSize+f*chunkSize+t] = float32(real(stftRight[fIdx][f]))
						inputBuf[3*freqBins*chunkSize+f*chunkSize+t] = float32(imag(stftRight[fIdx][f]))
					} else {
						inputBuf[0*freqBins*chunkSize+f*chunkSize+t] = float32(cmplx.Abs(stftLeft[fIdx][f]))
						inputBuf[1*freqBins*chunkSize+f*chunkSize+t] = float32(cmplx.Abs(stftRight[fIdx][f]))
					}
				}
			}

			outputTensors, err := sep.RunInference(inputBuf)
			if err != nil {
				log.Printf("Warning: Inference failed at frame %d: %v", frame, err)
				return
			}

			for t := 0; t < chunkSize; t++ {
				fIdx := frame + t
				for f := 0; f < freqBins; f++ {
					if chns == 4 {
						re_l := outputTensors[0*freqBins*chunkSize+f*chunkSize+t]
						im_l := outputTensors[1*freqBins*chunkSize+f*chunkSize+t]
						re_r := outputTensors[2*freqBins*chunkSize+f*chunkSize+t]
						im_r := outputTensors[3*freqBins*chunkSize+f*chunkSize+t]
						outputStftLeft[fIdx][f] = complex(float64(re_l), float64(im_l))
						outputStftRight[fIdx][f] = complex(float64(re_r), float64(im_r))
					} else {
						maskL := outputTensors[0*freqBins*chunkSize+f*chunkSize+t]
						maskR := outputTensors[1*freqBins*chunkSize+f*chunkSize+t]
						outputStftLeft[fIdx][f] = cmplx.Rect(float64(maskL), cmplx.Phase(stftLeft[fIdx][f]))
						outputStftRight[fIdx][f] = cmplx.Rect(float64(maskR), cmplx.Phase(stftRight[fIdx][f]))
					}
				}
			}

			newProg := atomic.AddInt32(&progress, int32(chunkSize/2))
			if frame%1024 == 0 {
				fmt.Printf("\rProgress: %.1f%%", float64(newProg)/float64(numFrames)*100)
			}
		}(startFrame)
	}
	wg.Wait()

	fmt.Println("\nReconstructing audio (Parallel ISTFT)...")
	var outLeft, outRight []float32
	wg.Add(2)
	go func() {
		defer wg.Done()
		outLeft = audio.GetISTFT(outputStftLeft, n_fft, hopSize, numSamples)
	}()
	go func() {
		defer wg.Done()
		outRight = audio.GetISTFT(outputStftRight, n_fft, hopSize, numSamples)
	}()
	wg.Wait()

	// Final Mastering
	fmt.Println("Final Processing: Peak Normalization & Boost (Parallel)...")
	
	numWorkers := runtime.NumCPU()
	chunkSizeMaster := (numSamples + numWorkers - 1) / numWorkers
	
	// Phase 1: Parallel Peak Detection
	peaks := make([]float32, numWorkers)
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func(id, start, end int) {
			defer wg.Done()
			currMax := float32(0.0)
			for i := start; i < end && i < numSamples; i++ {
				absL := float32(math.Abs(float64(outLeft[i])))
				absR := float32(math.Abs(float64(outRight[i])))
				if absL > currMax { currMax = absL }
				if absR > currMax { currMax = absR }
			}
			peaks[id] = currMax
		}(w, w*chunkSizeMaster, (w+1)*chunkSizeMaster)
	}
	wg.Wait()

	maxPeak := float32(0.0)
	for _, p := range peaks {
		if p > maxPeak { maxPeak = p }
	}
	
	gain := float32(1.0)
	if maxPeak > 1e-10 {
		gain = 0.99 / maxPeak
	}
	
	// Phase 2: Parallel Interleaving & Gain Application
	outInterleaved := make([]float32, numSamples*2)
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			for i := start; i < end && i < numSamples; i++ {
				outInterleaved[i*2] = outLeft[i] * gain
				outInterleaved[i*2+1] = outRight[i] * gain
			}
		}(w*chunkSizeMaster, (w+1)*chunkSizeMaster)
	}
	wg.Wait()

	outputFile := filepath.Join(filepath.Dir(inputFile), "separated_"+filepath.Base(inputFile))
	_ = audio.SaveWav(outputFile, outInterleaved, 44100, 2)

	fmt.Printf("Done! Saved to: %s (Gain Applied: %.4fx)\n", outputFile, gain)
}
