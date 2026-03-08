# TrainJob Operator

A Kubernetes operator for distributed GPU training, built around the idea that **AI agents should manage the infrastructure that trains and serves AI models** — and that the training pipeline, the serving engine, and the agents that operate them can form a self-improving loop.

The project has two parts:

1. **The operator** — a K8s controller that manages the full training lifecycle: admission-time validation, auto-parallelism configuration, pre-training GPU health checks, worker orchestration, checkpoint management, and runtime monitoring. Built with [controller-runtime](https://github.com/kubernetes-sigs/controller-runtime).

2. **The agent system** — 8 specialized AI agents that develop, evaluate, and operate the operator itself. They coordinate through git, run on vLLM (the same inference engine the trained models deploy to), and include a harness engineering pipeline that automatically tightens agent instructions based on observed failures.

The original motivation was training multimodal embedding models for [vLLM Semantic Router](https://github.com/vllm-project/semantic-router) — fine-tuning CLIP, SigLIP, or BGE-style encoders for the routing layer. The system that trains these models is operated by agents that run on the same vLLM infrastructure the models deploy to. That circularity — agents improving the infrastructure that powers the agents — is the central design idea.

> **Disclaimer**: This is a hobby / exploration project. It is **not** production-ready and is not affiliated with any company. There are known gaps (see [Limitations](docs/comparison.md#limitations)). Use at your own risk.

---

## The Loop: Agent ↔ Infra ↔ Model Co-Design

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   ┌──────────────────┐     ┌──────────────────┐                    │
│   │   AI Agents       │     │   TrainJob        │                    │
│   │                   │────►│   Operator        │                    │
│   │ model-builder     │     │                   │                    │
│   │ model-trainer     │     │ Webhooks:         │                    │
│   │ infra             │     │  auto-parallelism │                    │
│   │ ops               │     │  validation       │                    │
│   │ sre               │     │  NCCL injection   │                    │
│   │ cicd              │     │                   │                    │
│   │ model-eval        │     │ Reconciler:       │                    │
│   │ model-install     │     │  prolog → train   │                    │
│   │                   │     │  → checkpoint     │                    │
│   └──────┬───────────┘     │  → eval → install │                    │
│          │                  └────────┬─────────┘                    │
│          │                           │                               │
│          │  agents develop           │  operator trains              │
│          │  and improve the          │  embedding models             │
│          │  operator code            │                               │
│          │                           ▼                               │
│          │                  ┌──────────────────┐                    │
│          │                  │  vLLM Serving     │                    │
│          │                  │                   │                    │
│          └──────────────────│  Semantic Router  │                    │
│             agents run on   │  serves trained   │                    │
│             vLLM for        │  models for       │                    │
│             reasoning       │  production       │                    │
│                             └──────────────────┘                    │
│                                                                     │
│  The harness engineering loop:                                      │
│  agents run → traces observed → eval scores → iterate improves     │
│  SKILL.md → agents get better → operator improves → models improve │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

The three layers reinforce each other:

- **Agents** develop and operate the operator (code changes, monitoring, CI/CD). They run on vLLM with tool-calling support.
- **Operator** trains embedding models on GPU clusters. It uses model architecture knowledge to auto-configure parallelism — the model and infrastructure are co-designed at admission time.
- **vLLM** serves the trained models for production inference and powers the agents' reasoning. When agents improve the operator, the operator trains better models, which deploy to the same vLLM that makes the agents smarter.

This is agent-infra-model co-design: the agent system, the training infrastructure, and the models are not separate concerns — they're a single feedback loop where each layer improves the others.

---

## What Each Layer Does

### Operator (training infrastructure)

- **Auto-parallelism** — matches model architecture (hidden dim, layers, heads) to GPU hardware (memory, bandwidth, FP8 support) at admission time. Hierarchical search over TP/PP/CP/FSDP mirroring the NVLink→IB interconnect hierarchy.
- **Prolog** — pre-flight GPU health check (hardware diagnostics, kernel validation, interconnect bandwidth) before committing GPU-hours.
- **Sidecar monitoring** — per-pod DCGM metrics and straggler detection built into the StatefulSet template. No cluster-wide webhook needed.
- **Checkpoint management** — periodic save, validation, retention, and protection on deletion.
- **Kueue integration** — suspend-based admission control with child resource isolation.

### Agents (development and operations)

- **8 specialized roles** — model-builder, model-trainer, infra, ops, sre, cicd, model-eval, model-install. Each has a SKILL.md with domain instructions and scoped file ownership (WatchPaths).
- **Git-based coordination** — AGENCY.md is the task board. No message bus, no custom CRDs for orchestration.
- **Harness engineering pipeline** — run → observe (structured traces) → eval (7 deterministic scorers + LLM-as-judge) → iterate (failure pattern extraction → SKILL.md improvement proposals). The harness tightens automatically over time.
- **Model-install agent** — converts checkpoints to serving format, pushes to registry, generates vLLM deployment patches and semantic router config updates. Deployment artifacts go to a branch for human/GitOps review — agents never `kubectl apply` to production.

---

## Documentation

| Document | What it covers |
|---|---|
| [CRD Reference](docs/crd.md) | TrainJob spec, ModelArchSpec, CheckpointSpec, status fields, YAML examples |
| [Design](docs/design.md) | Workflow (admission → prolog → workers → monitoring → deletion), auto-parallelism advisor (hierarchical search, pruning, caching), webhook efficiency |
| [Kueue Integration](docs/kueue.md) | Suspend-based admission, child resource isolation, standalone mode |
| [Comparison & Limitations](docs/comparison.md) | vs. Kubeflow, Kueue, Volcano, NeMo, TAS; known limitations; production gaps |
| [Agent System](agents/README.md) | 8 specialized agents, interaction diagrams, coordination protocol, model-install agent, run→observe→eval→iterate harness engineering pipeline |

---

## Building

Requires Go 1.24+.

```bash
make          # fmt, vet, build → bin/trainjob-operator
make test     # run envtest-based controller tests
make run      # run locally against current kubeconfig
```

Docker:

```bash
make docker-build IMG=ghcr.io/rootfs/trainjob-operator:latest
make docker-push  IMG=ghcr.io/rootfs/trainjob-operator:latest
```

Agents:

```bash
make agent-build                   # build agent binary
make agent-run ROLE=model-builder  # dry-run an agent locally
make agent-eval                    # evaluate agent traces
make agent-pipeline ROLE=sre       # full run → eval → iterate
```

Run `make help` to see all targets.

---

## Project Structure

```
api/v1alpha1/
  types.go                       CRD types (TrainJobSpec, TrainJobStatus, phases)
  groupversion_info.go           GroupVersion, SchemeBuilder, AddToScheme
  zz_generated.deepcopy.go       DeepCopy implementations for all CRD types

cmd/
  main.go                        Entry point: manager, controller, webhooks, health checks

internal/controller/
  trainjob_controller.go         Reconciler with state-machine phase transitions
  trainjob_controller_test.go    envtest-based integration tests
  suite_test.go                  Envtest suite setup (k8sClient, manager bootstrap)
  prolog.go                      Prolog Job builder (3-phase GPU health check)
  workers.go                     Headless Service + worker StatefulSet + GPU sidecar builder
  checkpoint.go                  Checkpoint validation Job builder

internal/webhook/
  trainjob_validator.go          Validating webhook (~10 rules)
  trainjob_mutator.go            Mutating webhook (auto-parallelism, NCCL env, defaults, Kueue suspend)
  auto_config.go                 Auto-parallelism search engine (hierarchical, cached)
  model_registry.go              Built-in model architectures and GPU specs

docs/
  crd.md                         CRD reference and examples
  design.md                      Workflow, auto-parallelism, webhook efficiency
  kueue.md                       Kueue integration details
  comparison.md                  Framework comparison and limitations

agents/                          Multi-agent system (see agents/README.md)
examples/                        Sample TrainJob CRs

Makefile                         Build, test, lint, docker, agent targets
Dockerfile                       Multi-stage build (distroless runtime)
go.mod / go.sum                  Go module (github.com/rootfs/trainjob-operator)
```

---

## License

This project is provided as-is for educational and exploratory purposes. No warranty, no support guarantees.
