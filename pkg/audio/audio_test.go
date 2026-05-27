package audio

import (
	"math"
	"testing"
)

// Helper to generate a sine wave for testing
func generateSineWave(freq float64, sampleRate int, durationSec float64, amplitude float64) []float32 {
	numSamples := int(durationSec * float64(sampleRate))
	data := make([]float32, numSamples*2) // Stereo
	for i := 0; i < numSamples; i++ {
		val := float32(amplitude * math.Sin(2.0*math.Pi*freq*float64(i)/float64(sampleRate)))
		data[i*2] = val
		data[i*2+1] = val
	}
	return data
}

func TestResample(t *testing.T) {
	inRate := 44100
	outRate := 16000
	duration := 1.5 // seconds
	
	input := generateSineWave(440, inRate, duration, 0.5)
	
	// Separate L/R channels to test mono resampler
	inputMono := make([]float32, len(input)/2)
	for i := 0; i < len(inputMono); i++ {
		inputMono[i] = input[i*2]
	}

	resampled := Resample(inputMono, inRate, outRate)
	
	expectedLen := int(math.Floor(float64(len(inputMono)) * float64(outRate) / float64(inRate)))
	if math.Abs(float64(len(resampled)-expectedLen)) > 5 {
		t.Errorf("Expected resampled length around %d, got %d", expectedLen, len(resampled))
	}
}

func TestMeasureLUFS(t *testing.T) {
	sampleRate := 44100
	duration := 2.0 // seconds
	
	// Generate a louder sine wave
	input := generateSineWave(1000, sampleRate, duration, 0.8)
	lufs := MeasureLUFS(input, sampleRate, 2)
	
	if lufs <= -70.0 {
		t.Errorf("Loudness measure returned silent value: %f LUFS", lufs)
	}
	
	t.Logf("Measured loudness: %.2f LUFS", lufs)
}

func TestNormalizeLUFS(t *testing.T) {
	sampleRate := 44100
	duration := 2.0
	target := -23.0

	input := generateSineWave(1000, sampleRate, duration, 0.3)
	normalized := NormalizeLUFS(input, sampleRate, 2, target)
	
	newLufs := MeasureLUFS(normalized, sampleRate, 2)
	
	// Verify that the normalized audio is within 0.2 LUFS of the target
	if math.Abs(newLufs-target) > 0.2 {
		t.Errorf("Expected loudness near %f, got %f", target, newLufs)
	}
	
	t.Logf("Loudness after normalization: %.2f LUFS", newLufs)
}
