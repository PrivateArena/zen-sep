package audio

import (
	"math"
)

// Biquad represents a standard Direct Form I biquad filter.
type biquad struct {
	b0, b1, b2, a1, a2 float64
	x1, x2, y1, y2     float64
}

func (b *biquad) Process(x float64) float64 {
	y := b.b0*x + b.b1*b.x1 + b.b2*b.x2 - b.a1*b.y1 - b.a2*b.y2
	b.x2 = b.x1
	b.x1 = x
	b.y2 = b.y1
	b.y1 = y
	return y
}

// GetKWeightingFilters returns the two biquad stages for K-weighting at a given sample rate.
// Specifically configured for 44100 Hz and 48000 Hz. Defaults to 44100 Hz coefficients.
func getKWeightingFilters(sampleRate int) ([]*biquad, []*biquad) {
	// Stage 1: High shelf pre-filter
	// Stage 2: High pass RLB filter
	var s1L, s1R, s2L, s2R biquad

	if sampleRate == 48000 {
		// Stage 1 (48 kHz)
		s1L.b0, s1L.b1, s1L.b2 = 1.53090908, -2.65116903, 1.16916679
		s1L.a1, s1L.a2 = -1.66375011, 0.71265695

		// Stage 2 (48 kHz)
		s2L.b0, s2L.b1, s2L.b2 = 1.0, -2.0, 1.0
		s2L.a1, s2L.a2 = -1.99004745, 0.99007225
	} else {
		// Default to 44100 Hz coefficients
		s1L.b0, s1L.b1, s1L.b2 = 1.53512486, -2.69169619, 1.19839281
		s1L.a1, s1L.a2 = -1.69065929, 0.73248117

		s2L.b0, s2L.b1, s2L.b2 = 1.0, -2.0, 1.0
		s2L.a1, s2L.a2 = -1.99004745, 0.99007225
	}

	// Copy left channel coefficients to right channel
	s1R = s1L
	s2R = s2L

	return []*biquad{&s1L, &s2L}, []*biquad{&s1R, &s2R}
}

// MeasureLUFS computes the EBU R128 integrated loudness in LUFS for a stereo or mono signal.
func MeasureLUFS(data []float32, sampleRate int, channels int) float64 {
	if len(data) == 0 {
		return -70.0
	}

	numSamples := len(data) / channels
	leftFiltered := make([]float64, numSamples)
	rightFiltered := make([]float64, numSamples)

	sL, sR := getKWeightingFilters(sampleRate)

	if channels == 1 {
		for i := 0; i < numSamples; i++ {
			v := float64(data[i])
			v = sL[0].Process(v)
			v = sL[1].Process(v)
			leftFiltered[i] = v
			rightFiltered[i] = v // duplicate for calculations
		}
	} else {
		for i := 0; i < numSamples; i++ {
			l := float64(data[i*2])
			l = sL[0].Process(l)
			l = sL[1].Process(l)
			leftFiltered[i] = l

			r := float64(data[i*2+1])
			r = sR[0].Process(r)
			r = sR[1].Process(r)
			rightFiltered[i] = r
		}
	}

	// 400ms block size
	blockSize := int(0.400 * float64(sampleRate))
	// 75% overlap => 100ms step
	stepSize := int(0.100 * float64(sampleRate))

	if numSamples < blockSize {
		return -70.0
	}

	var energies []float64
	for start := 0; start <= numSamples-blockSize; start += stepSize {
		var sumL, sumR float64
		for i := 0; i < blockSize; i++ {
			sumL += leftFiltered[start+i] * leftFiltered[start+i]
			sumR += rightFiltered[start+i] * rightFiltered[start+i]
		}
		meanL := sumL / float64(blockSize)
		meanR := sumR / float64(blockSize)
		
		// Stereo weighting: w_L = 1.0, w_R = 1.0 (BS.1770-4)
		z := meanL + meanR
		energies = append(energies, z)
	}

	// Absolute threshold gate (-70 LUFS)
	// LUFS = -0.691 + 10 * log10(z)
	var absGated []float64
	var absSum float64
	for _, z := range energies {
		if z > 0 {
			lufs := -0.691 + 10.0*math.Log10(z)
			if lufs >= -70.0 {
				absGated = append(absGated, z)
				absSum += z
			}
		}
	}

	if len(absGated) == 0 {
		return -70.0
	}

	// Relative gate threshold: average of absolute gated minus 10 dB
	absAvg := absSum / float64(len(absGated))
	relThresholdLUFS := -0.691 + 10.0*math.Log10(absAvg) - 10.0

	// Relative gating pass
	var relGated []float64
	var relSum float64
	for _, z := range absGated {
		lufs := -0.691 + 10.0*math.Log10(z)
		if lufs >= relThresholdLUFS {
			relGated = append(relGated, z)
			relSum += z
		}
	}

	if len(relGated) == 0 {
		return -70.0
	}

	finalAvg := relSum / float64(len(relGated))
	return -0.691 + 10.0*math.Log10(finalAvg)
}

// NormalizeLUFS adjusts the gain of the audio buffer to hit a targeted integrated LUFS value.
func NormalizeLUFS(data []float32, sampleRate int, channels int, targetLUFS float64) []float32 {
	currentLUFS := MeasureLUFS(data, sampleRate, channels)
	if currentLUFS <= -70.0 {
		return data // Silence
	}

	diffDB := targetLUFS - currentLUFS
	gain := math.Pow(10.0, diffDB/20.0)
	
	// Prevent massive blow-up gain on nearly silent audio
	if gain > 10.0 {
		gain = 10.0
	}

	out := make([]float32, len(data))
	for i, v := range data {
		out[i] = float32(float64(v) * gain)
	}
	return out
}
