package main

import (
	"fmt"
	"os"
	"strings"
)

// Role defines an agent's specialization: which files it owns,
// what system prompt prefix it uses, and its identifier for AGENCY.md.
type Role struct {
	Name        string
	SkillPath   string   // path to SKILL.md relative to repo root
	WatchPaths  []string // files/dirs this agent cares about
	Description string
}

var roles = map[string]Role{
	"model-builder": {
		Name:      "model-builder",
		SkillPath: "agents/roles/model-builder/SKILL.md",
		WatchPaths: []string{
			"api/v1alpha1/types.go",
			"internal/webhook/model_registry.go",
		},
		Description: "Model architecture definitions and registry management",
	},
	"model-trainer": {
		Name:      "model-trainer",
		SkillPath: "agents/roles/model-trainer/SKILL.md",
		WatchPaths: []string{
			"internal/webhook/auto_config.go",
			"internal/webhook/trainjob_mutator.go",
		},
		Description: "Training parallelism configuration and auto-config advisor",
	},
	"infra": {
		Name:      "infra",
		SkillPath: "agents/roles/infra/SKILL.md",
		WatchPaths: []string{
			"api/v1alpha1/",
			"internal/controller/workers.go",
			"internal/controller/prolog.go",
			"internal/controller/checkpoint.go",
		},
		Description: "CRDs, StatefulSet/Job builders, Kueue integration",
	},
	"ops": {
		Name:      "ops",
		SkillPath: "agents/roles/ops/SKILL.md",
		WatchPaths: []string{
			"internal/controller/trainjob_controller.go",
			"internal/controller/checkpoint.go",
			"internal/webhook/",
		},
		Description: "Reconciler state machine, webhook logic, checkpoint management",
	},
	"sre": {
		Name:      "sre",
		SkillPath: "agents/roles/sre/SKILL.md",
		WatchPaths: []string{
			"internal/controller/workers.go",
			"internal/controller/prolog.go",
		},
		Description: "Monitoring, observability, GPU health, sidecar",
	},
	"cicd": {
		Name:      "cicd",
		SkillPath: "agents/roles/cicd/SKILL.md",
		WatchPaths: []string{
			"Makefile",
			"Dockerfile",
			"go.mod",
			".github/workflows/",
			"internal/controller/trainjob_controller_test.go",
			"internal/controller/suite_test.go",
		},
		Description: "Build, test, release, CI pipelines",
	},
	"model-eval": {
		Name:      "model-eval",
		SkillPath: "agents/roles/model-eval/SKILL.md",
		WatchPaths: []string{
			"internal/controller/checkpoint.go",
			"internal/controller/trainjob_controller.go",
			"api/v1alpha1/types.go",
			"internal/webhook/model_registry.go",
		},
		Description: "Post-training evaluation, benchmarking, and regression detection",
	},
	"model-install": {
		Name:      "model-install",
		SkillPath: "agents/roles/model-install/SKILL.md",
		WatchPaths: []string{
			"api/v1alpha1/types.go",
			"internal/controller/trainjob_controller.go",
			"internal/controller/checkpoint.go",
			"agents/manifests/vllm-deployment.yaml",
		},
		Description: "Checkpoint conversion, registry push, and serving config generation",
	},
}

// Config holds runtime configuration for an agent session.
type Config struct {
	Role       Role
	RepoURL    string
	RepoBranch string
	WorkDir    string
	VLLMEndpoint string
	VLLMModel    string
	MaxTurns     int
	DryRun       bool
}

func loadConfig() (*Config, error) {
	roleName := envOrDefault("AGENT_ROLE", "")
	if roleName == "" {
		return nil, fmt.Errorf("AGENT_ROLE is required (one of: %s)", strings.Join(roleNames(), ", "))
	}
	role, ok := roles[roleName]
	if !ok {
		return nil, fmt.Errorf("unknown role %q, valid roles: %s", roleName, strings.Join(roleNames(), ", "))
	}

	cfg := &Config{
		Role:         role,
		RepoURL:      envOrDefault("AGENT_REPO_URL", ""),
		RepoBranch:   envOrDefault("AGENT_REPO_BRANCH", "main"),
		WorkDir:      envOrDefault("AGENT_WORKDIR", "/workspace"),
		VLLMEndpoint: envOrDefault("VLLM_ENDPOINT", "http://vllm:8000/v1"),
		VLLMModel:    envOrDefault("VLLM_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct"),
		MaxTurns:     envIntOrDefault("AGENT_MAX_TURNS", 20),
		DryRun:       envOrDefault("AGENT_DRY_RUN", "false") == "true",
	}

	if cfg.RepoURL == "" {
		return nil, fmt.Errorf("AGENT_REPO_URL is required")
	}

	return cfg, nil
}

func roleNames() []string {
	names := make([]string, 0, len(roles))
	for k := range roles {
		names = append(names, k)
	}
	return names
}

func envOrDefault(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func envIntOrDefault(key string, def int) int {
	v := os.Getenv(key)
	if v == "" {
		return def
	}
	var n int
	if _, err := fmt.Sscanf(v, "%d", &n); err != nil {
		return def
	}
	return n
}
