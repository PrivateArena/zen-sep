package separator

import (
	"fmt"
	"log"
	"math"
	"math/cmplx"
	"os"
	"path/filepath"
	"runtime"
	"sync"

	"zen-sep/pkg/audio"
	"zen-sep/pkg/vad"

	"github.com/schollz/progressbar/v3"
	ort "github.com/yalue/onnxruntime_go"
)

// Separate routes the separation to the correct pipeline based on config,
// applies residual stem calculation, sample-rate downsampling, VAD, and saves outputs.
func (s *Separator) Separate(data []float32, inputFile string) error {
	var stems map[string][]float32
	var err error

	sampleRate := s.Config.SampleRate
	if sampleRate == 0 {
		sampleRate = 44100
	}

	switch s.Config.Type {
	case ModelTypeMDXNet:
		stems, err = s.processMDXNet(data)
	case ModelTypeDemucs:
		stems, err = s.processDemucs(data)
	case ModelTypeSpleeter:
		stems, err = s.processSpleeter(data)
	default:
		return fmt.Errorf("unsupported model type: %s", s.Config.Type)
	}

	if err != nil {
		return fmt.Errorf("separation processing failed: %w", err)
	}

	// 3.6 Residual stem generation (input minus separated stem)
	if s.Config.Residual && len(stems) == 1 {
		var stemName string
		var stemData []float32
		for k, v := range stems {
			stemName = k
			stemData = v
		}
		if len(stemData) == len(data) {
			residualData := make([]float32, len(data))
			for i := range data {
				residualData[i] = data[i] - stemData[i]
			}
			residualName := "residual"
			if stemName == "vocals" {
				residualName = "instrumental"
			} else if stemName == "instrumental" {
				residualName = "vocals"
			}
			stems[residualName] = residualData
		}
	}

	// Save all stems (applying TargetLUFS loudness normalization or peak normalization)
	for stemName, stemData := range stems {
		err = s.saveStereoFull(stemData, inputFile, stemName, sampleRate)
		if err != nil {
			return fmt.Errorf("failed to save stem %s: %w", stemName, err)
		}
	}

	// 3.5 Resample to 16 kHz Mono
	if s.Config.Also16k {
		for stemName, stemData := range stems {
			mono := make([]float32, len(stemData)/2)
			for i := 0; i < len(mono); i++ {
				mono[i] = (stemData[i*2] + stemData[i*2+1]) / 2.0
			}
			samples16k := audio.Resample(mono, sampleRate, 16000)

			// Apply target LUFS normalization to 16k output if specified
			if s.Config.TargetLUFS != 0 {
				samples16k = audio.NormalizeLUFS(samples16k, 16000, 1, s.Config.TargetLUFS)
			} else {
				peakNormalize(samples16k)
			}

			outputFile := s.getOutputPath(inputFile, stemName+"_16k")
			err = audio.SaveWav(outputFile, samples16k, 16000, 1)
			if err != nil {
				return fmt.Errorf("failed to save 16k mono stem %s: %w", stemName, err)
			}
			fmt.Printf("Saved 16k mono: %s\n", outputFile)
		}
	}

	// 3.3 Voice Activity Detection (VAD) post-processing
	if s.Config.VADModel != "" {
		vadData, ok := stems["vocals"]
		if !ok {
			for _, v := range stems {
				vadData = v
				break
			}
		}
		if len(vadData) > 0 {
			libPath := os.Getenv("ONNXRUNTIME_LIB")
			if libPath == "" {
				libPath = "/usr/lib/libonnxruntime.so"
			}
			detector, err := vad.NewDetector(s.Config.VADModel, libPath)
			if err == nil {
				defer detector.Close()
				segments, err := detector.Detect(vadData, sampleRate, 2)
				if err == nil {
					fmt.Printf("VAD speech detection found %d segments.\n", len(segments))
					for idx, seg := range segments {
						startIdx := int(seg.StartSec * float64(sampleRate)) * 2
						endIdx := int(seg.EndSec * float64(sampleRate)) * 2
						if endIdx > len(vadData) {
							endIdx = len(vadData)
						}
						segData := vadData[startIdx:endIdx]

						segDir := filepath.Join(s.Config.OutputDir, "segments")
						if s.Config.OutputDir == "" {
							segDir = filepath.Join(filepath.Dir(inputFile), "segments")
						}
						_ = os.MkdirAll(segDir, 0755)

						segFile := filepath.Join(segDir, fmt.Sprintf("%s_segment_%d.wav", filepath.Base(inputFile), idx))
						_ = audio.SaveWav(segFile, segData, sampleRate, 2)
					}
				} else {
					log.Printf("VAD segment detection failed: %v", err)
				}
			} else {
				log.Printf("Failed to initialize VAD detector: %v", err)
			}
		}
	}

	return nil
}

func (s *Separator) processMDXNet(data []float32) (map[string][]float32, error) {
	n_fft := s.Config.N_FFT
	hopSize := s.Config.HopSize
	freqBins := s.Config.FreqBins
	chunkSize := s.Config.ChunkSize
	numSamples := len(data) / 2

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

	// 2.3 Pre-compute magnitude array to avoid per-chunk cmplx.Abs
	fmt.Println("Pre-computing spectral magnitudes...")
	magLeft := make([][]float32, numFrames)
	magRight := make([][]float32, numFrames)
	for i := 0; i < numFrames; i++ {
		magLeft[i] = make([]float32, freqBins)
		magRight[i] = make([]float32, freqBins)
		for f := 0; f < freqBins; f++ {
			rL := float32(real(stftLeft[i][f]))
			iL := float32(imag(stftLeft[i][f]))
			magLeft[i][f] = float32(math.Sqrt(float64(rL*rL + iL*iL)))

			rR := float32(real(stftRight[i][f]))
			iR := float32(imag(stftRight[i][f]))
			magRight[i][f] = float32(math.Sqrt(float64(rR*rR + iR*iR)))
		}
	}

	// 2.2 Concurrency workers based on runtime CPUs / intraOp
	intraOp := s.Config.IntraOpThreads
	if intraOp == 0 {
		intraOp = 4
	}
	workers := runtime.NumCPU() / intraOp
	if workers < 1 {
		workers = 1
	}
	sem := make(chan struct{}, workers)

	totalChunks := (numFrames + (chunkSize / 2) - 1) / (chunkSize / 2)
	bar := progressbar.Default(int64(totalChunks), "Separating (MDX)")

	var errMutex sync.Mutex
	var firstErr error
	setErr := func(e error) {
		errMutex.Lock()
		defer errMutex.Unlock()
		if firstErr == nil {
			firstErr = e
		}
	}

	// 2.2 Reuse chunk buffers with a local sync.Pool
	var chunkBufPool = sync.Pool{
		New: func() any {
			buf := make([]float32, 1*2*freqBins*chunkSize)
			return &buf
		},
	}

	for startFrame := 0; startFrame < numFrames; startFrame += chunkSize / 2 {
		if firstErr != nil {
			break
		}
		wg.Add(1)
		go func(frame int) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			actualFrames := chunkSize
			if frame+chunkSize > numFrames {
				actualFrames = numFrames - frame
			}
			if actualFrames <= 0 {
				return
			}

			inputBufPtr := chunkBufPool.Get().(*[]float32)
			inputBuf := *inputBufPtr
			// Clear buffer since it is reused
			for i := range inputBuf {
				inputBuf[i] = 0
			}

			// Fill magnitude data (rest will remain zero-padded)
			for t := 0; t < actualFrames; t++ {
				fIdx := frame + t
				for f := 0; f < freqBins; f++ {
					inputBuf[0*freqBins*chunkSize+f*chunkSize+t] = magLeft[fIdx][f]
					inputBuf[1*freqBins*chunkSize+f*chunkSize+t] = magRight[fIdx][f]
				}
			}

			inputShape := ort.NewShape(1, 2, int64(freqBins), int64(chunkSize))
			inputTensor, err := ort.NewTensor(inputShape, inputBuf)
			if err != nil {
				setErr(fmt.Errorf("failed to create input tensor: %w", err))
				chunkBufPool.Put(inputBufPtr)
				return
			}
			defer inputTensor.Destroy()

			outputTensor, err := ort.NewEmptyTensor[float32](inputShape)
			if err != nil {
				setErr(fmt.Errorf("failed to create empty output tensor: %w", err))
				chunkBufPool.Put(inputBufPtr)
				return
			}
			defer outputTensor.Destroy()

			err = s.RunInference(
				[]ort.Value{inputTensor},
				[]ort.Value{outputTensor},
			)
			if err != nil {
				setErr(fmt.Errorf("inference failed: %w", err))
				chunkBufPool.Put(inputBufPtr)
				return
			}

			outputBuf := outputTensor.GetData()
			for t := 0; t < actualFrames; t++ {
				fIdx := frame + t
				for f := 0; f < freqBins; f++ {
					maskL := outputBuf[0*freqBins*chunkSize+f*chunkSize+t]
					maskR := outputBuf[1*freqBins*chunkSize+f*chunkSize+t]
					outputStftLeft[fIdx][f] = cmplx.Rect(float64(maskL), cmplx.Phase(stftLeft[fIdx][f]))
					outputStftRight[fIdx][f] = cmplx.Rect(float64(maskR), cmplx.Phase(stftRight[fIdx][f]))
				}
			}

			chunkBufPool.Put(inputBufPtr)
			_ = bar.Add(1)
		}(startFrame)
	}
	wg.Wait()

	if firstErr != nil {
		return nil, firstErr
	}

	fmt.Println("Reconstructing audio...")
	outLeft := audio.GetISTFT(outputStftLeft, n_fft, hopSize, numSamples)
	outRight := audio.GetISTFT(outputStftRight, n_fft, hopSize, numSamples)

	// Return separated stem
	numSamplesTotal := len(outLeft)
	outInterleaved := make([]float32, numSamplesTotal*2)
	for i := 0; i < numSamplesTotal; i++ {
		outInterleaved[i*2] = outLeft[i]
		outInterleaved[i*2+1] = outRight[i]
	}

	return map[string][]float32{
		"instrumental": outInterleaved,
	}, nil
}

func (s *Separator) processDemucs(data []float32) (map[string][]float32, error) {
	chunkSize := s.Config.ChunkSize
	numSamples := len(data) / 2
	numStems := len(s.Config.Stems)

	demucs_n_fft := s.Config.N_FFT
	if demucs_n_fft == 0 {
		demucs_n_fft = 4096
	}
	demucs_hop := s.Config.HopSize
	if demucs_hop == 0 {
		demucs_hop = 1024
	}

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
		inputWavTensor, err := ort.NewTensor(inputWavShape, inputWav)
		if err != nil {
			return nil, fmt.Errorf("failed to create Demucs wav tensor: %w", err)
		}

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
		inputXTensor, err := ort.NewTensor(inputXShape, inputX)
		if err != nil {
			inputWavTensor.Destroy()
			return nil, fmt.Errorf("failed to create Demucs spectrogram tensor: %w", err)
		}

		outputTensors := make([]*ort.Tensor[float32], len(s.Outputs))
		for i, out := range s.Outputs {
			outShape := ort.NewShape(out.Dimensions...)
			outputTensors[i], err = ort.NewEmptyTensor[float32](outShape)
			if err != nil {
				inputWavTensor.Destroy()
				inputXTensor.Destroy()
				for _, t := range outputTensors[:i] {
					t.Destroy()
				}
				return nil, fmt.Errorf("failed to create empty Demucs output tensor: %w", err)
			}
		}

		inputVals := []ort.Value{inputWavTensor, inputXTensor}
		outputVals := make([]ort.Value, len(outputTensors))
		for i, t := range outputTensors {
			outputVals[i] = t
		}
		err = s.RunInference(inputVals, outputVals)

		inputWavTensor.Destroy()
		inputXTensor.Destroy()

		if err != nil {
			for _, t := range outputTensors {
				t.Destroy()
			}
			return nil, fmt.Errorf("Demucs inference failed at %d: %w", start, err)
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

	stems := make(map[string][]float32)
	for sIdx, stemName := range s.Config.Stems {
		out := make([]float32, numSamples*2)
		for i := 0; i < numSamples; i++ {
			if stemWeights[i] > 1e-5 {
				out[i*2] = stemOutputs[sIdx][i*2] / stemWeights[i]
				out[i*2+1] = stemOutputs[sIdx][i*2+1] / stemWeights[i]
			}
		}
		stems[stemName] = out
	}

	return stems, nil
}

func (s *Separator) processSpleeter(data []float32) (map[string][]float32, error) {
	n_fft := s.Config.N_FFT
	if n_fft == 0 {
		n_fft = 4096
	}
	hopSize := s.Config.HopSize
	if hopSize == 0 {
		hopSize = 1024
	}
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

	// 1.8 Chunking large Spleeter files in dynamic split batches to avoid OOMs
	batchSize := 16
	bar := progressbar.Default(int64(numSplits), "Separating (Spleeter)")
	outData := make([]float32, 2*numSplits*freqBins*splitSize)

	for splitStart := 0; splitStart < numSplits; splitStart += batchSize {
		currentSplits := batchSize
		if splitStart+batchSize > numSplits {
			currentSplits = numSplits - splitStart
		}

		inputX := make([]float32, 2*currentSplits*freqBins*splitSize)
		for split := 0; split < currentSplits; split++ {
			globalSplit := splitStart + split
			for t := 0; t < splitSize; t++ {
				fIdx := globalSplit*splitSize + t
				if fIdx >= numFrames {
					break
				}
				for f := 0; f < freqBins; f++ {
					inputX[0*currentSplits*freqBins*splitSize+split*freqBins*splitSize+f*splitSize+t] = float32(cmplx.Abs(stftL[fIdx][f]))
					inputX[1*currentSplits*freqBins*splitSize+split*freqBins*splitSize+f*splitSize+t] = float32(cmplx.Abs(stftR[fIdx][f]))
				}
			}
		}

		inputShape := ort.NewShape(2, int64(currentSplits), int64(freqBins), int64(splitSize))
		inputTensor, err := ort.NewTensor(inputShape, inputX)
		if err != nil {
			return nil, fmt.Errorf("failed to create Spleeter input tensor: %w", err)
		}

		outputTensors := make([]*ort.Tensor[float32], len(s.Outputs))
		for i := range s.Outputs {
			outputTensors[i], err = ort.NewEmptyTensor[float32](inputShape)
			if err != nil {
				inputTensor.Destroy()
				for _, t := range outputTensors[:i] {
					t.Destroy()
				}
				return nil, fmt.Errorf("failed to create empty Spleeter output tensor: %w", err)
			}
		}

		inputVals := []ort.Value{inputTensor}
		outputVals := make([]ort.Value, len(outputTensors))
		for i, t := range outputTensors {
			outputVals[i] = t
		}

		err = s.RunInference(inputVals, outputVals)
		inputTensor.Destroy()

		if err != nil {
			for _, t := range outputTensors {
				t.Destroy()
			}
			return nil, fmt.Errorf("Spleeter inference failed: %w", err)
		}

		batchOutData := outputTensors[0].GetData()
		copy(outData[2*splitStart*freqBins*splitSize:], batchOutData)

		for _, t := range outputTensors {
			t.Destroy()
		}
		_ = bar.Add(int(currentSplits))
	}

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

	numSamplesTotal := len(outLeft)
	outInterleaved := make([]float32, numSamplesTotal*2)
	for i := 0; i < numSamplesTotal; i++ {
		outInterleaved[i*2] = outLeft[i]
		outInterleaved[i*2+1] = outRight[i]
	}

	return map[string][]float32{
		"vocals": outInterleaved,
	}, nil
}

func (s *Separator) getOutputPath(originalPath, suffix string) string {
	baseName := suffix + "_" + filepath.Base(originalPath)
	if s.Config.OutputDir != "" {
		_ = os.MkdirAll(s.Config.OutputDir, 0755)
		return filepath.Join(s.Config.OutputDir, baseName)
	}
	return filepath.Join(filepath.Dir(originalPath), baseName)
}

func (s *Separator) saveStereoFull(data []float32, originalPath, suffix string, sampleRate int) error {
	if s.Config.TargetLUFS != 0 {
		data = audio.NormalizeLUFS(data, sampleRate, 2, s.Config.TargetLUFS)
	} else {
		peakNormalize(data)
	}

	outputFile := s.getOutputPath(originalPath, suffix)
	err := audio.SaveWav(outputFile, data, sampleRate, 2)
	if err != nil {
		return err
	}
	fmt.Printf("Saved: %s\n", outputFile)
	return nil
}

func peakNormalize(data []float32) {
	maxPeak := float32(0.0)
	for _, v := range data {
		if math.Abs(float64(v)) > float64(maxPeak) {
			maxPeak = float32(math.Abs(float64(v)))
		}
	}
	if maxPeak > 0 {
		gain := 0.99 / maxPeak
		for i := range data {
			data[i] *= gain
		}
	}
}
