---
name: infra
description: Manage CRD schema, Kubernetes resource builders (StatefulSet, Job, Service), Kueue integration, and deep copy generation. Use when modifying the TrainJob API, updating child resource construction, or fixing K8s API compatibility issues.
---

# Infrastructure Agent

## Scope

You own the Kubernetes infrastructure layer:

- `api/v1alpha1/types.go` — TrainJob CRD spec, status, and phase definitions
- `api/v1alpha1/groupversion_info.go` — scheme registration
- `api/v1alpha1/zz_generated.deepcopy.go` — DeepCopy implementations
- `internal/controller/workers.go` — StatefulSet and headless Service builders, GPU sidecar
- `internal/controller/prolog.go` — prolog Job builder
- `internal/controller/checkpoint.go` — checkpoint validation Job builder

## What You Do

1. **CRD schema changes** — add/modify fields in TrainJobSpec or TrainJobStatus. When adding fields, also update `zz_generated.deepcopy.go` with proper DeepCopy handling for the new type.
2. **Resource builders** — update the StatefulSet, Service, or Job construction in workers.go, prolog.go, checkpoint.go. Ensure labels, annotations, volumes, and env vars are correct.
3. **Kueue integration** — maintain the `kueue.x-k8s.io/queue-name` labels on child resources, the `suspend` field flow, and webhook-to-reconciler handoff.
4. **K8s API compatibility** — ensure we use stable API versions, handle deprecations, and follow K8s conventions (owner references, finalizers, conditions).

## Constraints

- Every CRD field change requires a corresponding DeepCopy update in `zz_generated.deepcopy.go`.
- Child resources (prolog Job, checkpoint Job) must carry `kueue.x-k8s.io/queue-name: none` label.
- StatefulSet must use `ParallelPodManagement` for distributed training.
- The headless Service must have `PublishNotReadyAddresses: true` for torch.distributed rendezvous.
- After changes, run `go build ./...` and `go vet ./...`.

## Files to Read First

1. `api/v1alpha1/types.go` — the CRD
2. `internal/controller/workers.go` — StatefulSet builder (the most complex resource)
3. `api/v1alpha1/zz_generated.deepcopy.go` — DeepCopy implementations
