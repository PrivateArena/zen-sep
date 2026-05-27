package config

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v3"
	"zen-sep/pkg/separator"
)

// Load reads a YAML configuration file and parses it into a separator.Config.
func Load(path string) (*separator.Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	var cfg separator.Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("failed to unmarshal yaml: %w", err)
	}

	// Apply reasonable defaults if they are missing
	if cfg.SampleRate == 0 {
		cfg.SampleRate = 44100
	}
	if cfg.Type == "" {
		cfg.Type = separator.ModelTypeMDXNet
	}

	return &cfg, nil
}
