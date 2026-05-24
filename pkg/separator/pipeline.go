package separator

import (
	"fmt"
	"log"
	"math"
	"math/cmplx"
	"path/filepath"
	"sync"
	"sync/atomic"

	"zen-separator/pkg/audio"
	"github.com/schollz/progressbar/v3"
	ort "github.com/yalue/onnxruntime_go"
)

// Separate routes the separation to the correct pipeline based on config.
func (s *Separator) Separate(data []float32, inputFile string) error {
	switch s.Config.Type {
	case ModelTypeMDXNet:
		s.processMDXNet(data, inputFile)
	case ModelTypeDemucs:
		s.processDemucs(data, inputFile)
	case ModelTypeSpleeter:
		s.processSpleeter(data, inputFile)
	default:
		return fmt.Errorf("unsupported model type: %s", s.Config.Type)
	}
	return nil
}

func (s *Separator) processMDXNet(data []float32, inputFile string) {
	n_fft := s.Config.N_FFT
	hopSize := s.Config.HopSize
	freqBins := s.Config.FreqBins
	chunkSize := s.Config.ChunkSize
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

			err := s.RunInference(
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

	s.saveStereo(outLeft, outRight, inputFile, "instrumental")
}

func (s *Separator) processDemucs(data []float32, inputFile string) {
	chunkSize := s.Config.ChunkSize
	numSamples := len(data) / 2
	numStems := len(s.Config.Stems)

	demucs_n_fft := 4096
	demucs_hop := 1024

	leftFull := make([]float32, numSamples)
	rightFull := make([]float32, numSamples)
	for i := 0; i < numSamples; i++ {
		leftFull[i] = data[i*2]
		rightFull[i] = data[i*2+1]
	}
	fmt.Println("Getting STFT (Demucs, full track)...")
	stftFullL := audio.GetSTFT(leftFull, demucs_n_fft, demucs_hop)
	stftFullR := audio.GetSTFT(rightFull, demucs_n_fft, demucs_hop)

	stepSize := chunkSize * 3 / 4
	numChunks := (numSamples + stepSize - 1) / stepSize

	stemOutputs := make([][]float32, numStems)
	stemWeights := make([]float32, numSamples)
	for i := range stemOutputs {
		stemOutputs[i] = make([]float32, numSamples*2)
	}

	xFreqBins := 2048
	xFrames := 336

	bar := progressbar.Default(int64(numChunks), "Separating (Demucs)")

	for start := 0; start < numSamples; start += stepSize {
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

		outputTensors := make([]*ort.Tensor[float32], len(s.Outputs))
		for i, out := range s.Outputs {
			outShape := ort.NewShape(out.Dimensions...)
			outputTensors[i], _ = ort.NewEmptyTensor[float32](outShape)
		}

		inputVals := []ort.Value{inputWavTensor, inputXTensor}
		outputVals := make([]ort.Value, len(outputTensors))
		for i, t := range outputTensors {
			outputVals[i] = t
		}
		err := s.RunInference(inputVals, outputVals)

		inputWavTensor.Destroy()
		inputXTensor.Destroy()

		if err != nil {
			log.Printf("Inference error at %d: %v", start, err)
			for _, t := range outputTensors {
				t.Destroy()
			}
			_ = bar.Add(1)
			continue
		}

		outData := outputTensors[0].GetData()
		for i := 0; i < chunkSize; i++ {
			idx := start + i
			if idx >= numSamples {
				break
			}

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

			for sIdx := 0; sIdx < numStems; sIdx++ {
				lVal := outData[sIdx*2*chunkSize+0*chunkSize+i]
				rVal := outData[sIdx*2*chunkSize+1*chunkSize+i]
				stemOutputs[sIdx][idx*2] += lVal * w
				stemOutputs[sIdx][idx*2+1] += rVal * w
			}
			stemWeights[idx] += w
		}
		for _, t := range outputTensors {
			t.Destroy()
		}
		_ = bar.Add(1)
	}

	for sIdx, stemName := range s.Config.Stems {
		out := make([]float32, numSamples*2)
		for i := 0; i < numSamples; i++ {
			if stemWeights[i] > 1e-5 {
				out[i*2] = stemOutputs[sIdx][i*2] / stemWeights[i]
				out[i*2+1] = stemOutputs[sIdx][i*2+1] / stemWeights[i]
			}
		}
		s.saveStereoFull(out, inputFile, stemName)
	}
}

func (s *Separator) processSpleeter(data []float32, inputFile string) {
	n_fft := 4096
	hopSize := 1024
	numSamples := len(data) / 2

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
	splitSize := 1024
	numSplits := (numFrames + splitSize - 1) / splitSize
	freqBins := 512

	inputX := make([]float32, 2*numSplits*freqBins*splitSize)
	for split := 0; split < numSplits; split++ {
		for t := 0; t < splitSize; t++ {
			fIdx := split*splitSize + t
			if fIdx >= numFrames {
				break
			}
			for f := 0; f < freqBins; f++ {
				inputX[0*numSplits*freqBins*splitSize+split*freqBins*splitSize+f*splitSize+t] = float32(cmplx.Abs(stftL[fIdx][f]))
				inputX[1*numSplits*freqBins*splitSize+split*freqBins*splitSize+f*splitSize+t] = float32(cmplx.Abs(stftR[fIdx][f]))
			}
		}
	}

	inputShape := ort.NewShape(2, int64(numSplits), int64(freqBins), int64(splitSize))
	inputTensor, _ := ort.NewTensor(inputShape, inputX)
	defer inputTensor.Destroy()

	outputTensors := make([]*ort.Tensor[float32], len(s.Outputs))
	for i := range s.Outputs {
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
	err := s.RunInference(inputVals, outputVals)
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
			if fIdx >= numFrames {
				break
			}
			for f := 0; f < freqBins; f++ {
				maskL := outData[0*numSplits*freqBins*splitSize+split*freqBins*splitSize+f*splitSize+t]
				maskR := outData[1*numSplits*freqBins*splitSize+split*freqBins*splitSize+f*splitSize+t]

				outputStftL[fIdx][f] = cmplx.Rect(float64(maskL), cmplx.Phase(stftL[fIdx][f]))
				outputStftR[fIdx][f] = cmplx.Rect(float64(maskR), cmplx.Phase(stftR[fIdx][f]))
			}
		}
	}

	fmt.Println("Reconstructing Spleeter audio...")
	outLeft := audio.GetISTFT(outputStftL, n_fft, hopSize, numSamples)
	outRight := audio.GetISTFT(outputStftR, n_fft, hopSize, numSamples)

	s.saveStereo(outLeft, outRight, inputFile, "vocals")
}

func (s *Separator) saveStereoFull(data []float32, originalPath, suffix string) {
	outputFile := filepath.Join(filepath.Dir(originalPath), suffix+"_"+filepath.Base(originalPath))
	_ = audio.SaveWav(outputFile, data, 44100, 2)
	fmt.Printf("\rSaved: %-40s\n", outputFile)
}

func (s *Separator) saveStereo(left, right []float32, originalPath, suffix string) {
	numSamples := len(left)
	outInterleaved := make([]float32, numSamples*2)
	maxPeak := float32(0.0)
	for i := 0; i < numSamples; i++ {
		outInterleaved[i*2] = left[i]
		outInterleaved[i*2+1] = right[i]
		if math.Abs(float64(left[i])) > float64(maxPeak) {
			maxPeak = float32(math.Abs(float64(left[i])))
		}
		if math.Abs(float64(right[i])) > float64(maxPeak) {
			maxPeak = float32(math.Abs(float64(right[i])))
		}
	}

	gain := float32(1.0)
	if maxPeak > 0 {
		gain = 0.99 / maxPeak
	}
	for i := range outInterleaved {
		outInterleaved[i] *= gain
	}

	outputFile := filepath.Join(filepath.Dir(originalPath), suffix+"_"+filepath.Base(originalPath))
	_ = audio.SaveWav(outputFile, outInterleaved, 44100, 2)
	fmt.Printf("Saved: %s\n", outputFile)
}
