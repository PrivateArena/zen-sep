package config

import (
	"os"
	"testing"
	"zen-sep/pkg/separator"
)

func TestLoadConfig(t *testing.T) {
	yamlContent := `
type: mdxnet
sample_rate: 48000
n_fft: 5120
hop_size: 1280
chunk_size: 256
freq_bins: 2561
stems:
  - vocals
  - instrumental
intra_op_threads: 4
output_dir: "/tmp/out"
device: cuda
target_lufs: -23.0
residual: true
also_16k: true
`
	tmpFile, err := os.CreateTemp("", "config_*.yaml")
	if err != nil {
		t.Fatalf("failed to create temp file: %v", err)
	}
	defer os.Remove(tmpFile.Name())

	if _, err := tmpFile.Write([]byte(yamlContent)); err != nil {
		t.Fatalf("failed to write to temp file: %v", err)
	}
	tmpFile.Close()

	cfg, err := Load(tmpFile.Name())
	if err != nil {
		t.Fatalf("failed to load config: %v", err)
	}

	if cfg.Type != separator.ModelTypeMDXNet {
		t.Errorf("expected ModelTypeMDXNet, got %s", cfg.Type)
	}
	if cfg.SampleRate != 48000 {
		t.Errorf("expected SampleRate 48000, got %d", cfg.SampleRate)
	}
	if cfg.IntraOpThreads != 4 {
		t.Errorf("expected IntraOpThreads 4, got %d", cfg.IntraOpThreads)
	}
	if cfg.OutputDir != "/tmp/out" {
		t.Errorf("expected OutputDir /tmp/out, got %s", cfg.OutputDir)
	}
	if cfg.Device != "cuda" {
		t.Errorf("expected Device cuda, got %s", cfg.Device)
	}
	if cfg.TargetLUFS != -23.0 {
		t.Errorf("expected TargetLUFS -23.0, got %f", cfg.TargetLUFS)
	}
	if !cfg.Residual {
		t.Errorf("expected Residual true, got false")
	}
	if !cfg.Also16k {
		t.Errorf("expected Also16k true, got false")
	}
}
