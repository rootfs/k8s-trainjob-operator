---
project: trainjob-operator
status: active
last_cortex_run: null
---

# Project State

## Components

| Component | Owner | Status | Branch | Description |
|-----------|-------|--------|--------|-------------|
| initial-model-registry | null | pending | - | Populate model registry with community model architectures |
| auto-config-test-coverage | null | pending | - | Add unit tests for auto-parallelism search edge cases |
| prolog-ib-detection | null | pending | - | Improve InfiniBand port detection in prolog health check |
| ci-github-actions | null | pending | - | Set up GitHub Actions CI pipeline (lint, vet, test, build) |
| sidecar-dcgm-metrics | null | pending | - | Validate DCGM exporter metric names against Prometheus conventions |
| kueue-e2e-test | null | pending | - | Add integration test for Kueue suspend/unsuspend flow |
| eval-job-builder | null | pending | - | Create eval Job builder with benchmark suites and regression detection |
| eval-phase-reconciler | null | pending | - | Add PhaseEvaluating to reconciler state machine with eval Job lifecycle |
| install-conversion-job | null | pending | - | Build checkpoint-to-safetensors conversion Job builder (internal/controller/install.go) |
| install-registry-push | null | pending | - | Add OCI/HF/S3 registry push step after conversion completes |
| install-serving-config | null | pending | - | Generate vLLM serving config and semantic router routing table updates |

## Dependencies

```
initial-model-registry
  └── blocks: auto-config-test-coverage (need models to test against)

ci-github-actions
  └── blocks: nothing (independent)

prolog-ib-detection
  └── related: sidecar-dcgm-metrics (both touch GPU hardware detection)

eval-job-builder
  └── blocks: eval-phase-reconciler (need the Job builder before wiring into reconciler)

eval-phase-reconciler
  └── blocks: install-conversion-job (eval must pass before install starts)

install-conversion-job
  └── blocks: install-registry-push (need converted model before pushing)

install-registry-push
  └── blocks: install-serving-config (need registry artifact before generating serving config)
```

## Agent Notes

_No agent activity yet. Notes will be appended here as agents claim and complete tasks._

## Protocol

### Claiming Work

1. Pull latest `main` and `AGENCY.md`
2. Find a component with `Owner: null` and `Status: pending` that matches your role
3. Set `Owner` to your agent identifier, `Status` to `in_progress`, `Branch` to `agent/<role>/<component>`
4. Commit and push AGENCY.md to your branch
5. Begin work

### Completing Work

1. Push all changes to your branch
2. Set `Status` to `proposed` in AGENCY.md
3. Add a note in Agent Notes describing what was done
4. Release ownership (`Owner: null`) only after human merges the PR

### Conflict Resolution

- If `git push` fails due to AGENCY.md conflict, pull, re-check ownership, retry
- Never override another agent's claim
- If blocked by a dependency, add a note and move to the next available task
