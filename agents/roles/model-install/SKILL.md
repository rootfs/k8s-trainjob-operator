---
name: model-install
description: Deploy trained models from checkpoints to inference engines. Handles checkpoint-to-serving format conversion, model registry push, vLLM serving config generation, and semantic router routing table updates. Use when building the deployment pipeline between training completion and production serving.
---

# Model Install Agent

## Scope

You own the last mile between training and serving:

- `internal/controller/install.go` (new) — conversion Job builder and registry push logic
- `api/v1alpha1/types.go` — InstallSpec and install-related status fields
- Deployment manifests for vLLM serving updates
- Semantic router config generation

## What You Do

### 1. Checkpoint Conversion

Training produces sharded PyTorch checkpoint files (one per rank). Inference engines
need consolidated weights in a standard format. Build a `buildConversionJob()` function
that creates a K8s Job to:

- Consolidate distributed checkpoint shards into a single model
- Convert to HuggingFace `safetensors` format (the vLLM default)
- Optionally quantize (GPTQ, AWQ, FP8) for faster inference
- Write the output to a PVC or object storage path

The conversion Job needs GPU access (1 GPU minimum) to load and resave the model.

```go
type InstallSpec struct {
    Enabled       bool   `json:"enabled"`
    TargetFormat  string `json:"targetFormat,omitempty"`  // "safetensors" (default), "gguf", "tensorrt-llm"
    Quantization  string `json:"quantization,omitempty"`  // "", "gptq-4bit", "awq-4bit", "fp8"
    RegistryPath  string `json:"registryPath,omitempty"`  // e.g., "ghcr.io/rootfs/models/clip-vit-l"
    RegistryType  string `json:"registryType,omitempty"`  // "oci", "huggingface", "s3"
    ServingConfig *ServingConfigSpec `json:"servingConfig,omitempty"`
}

type ServingConfigSpec struct {
    VLLMDeployment    string `json:"vllmDeployment,omitempty"`    // name of the vLLM Deployment to update
    VLLMNamespace     string `json:"vllmNamespace,omitempty"`     // namespace of the vLLM Deployment
    SemanticRouterMap string `json:"semanticRouterMap,omitempty"` // ConfigMap name for routing table
    MaxReplicas       int32  `json:"maxReplicas,omitempty"`       // for canary: max replicas on new version
    CanaryPercent     int32  `json:"canaryPercent,omitempty"`     // traffic percentage for canary (0 = full rollout)
}
```

### 2. Registry Push

After conversion, push the model to a registry:

- **OCI/ORAS**: Push as an OCI artifact to ghcr.io, DockerHub, or any OCI registry.
  Use `oras push` in the Job container. This is the most K8s-native option.
- **HuggingFace Hub**: Push via `huggingface-cli upload`. Good for community models.
- **S3/GCS**: Simple `aws s3 cp` or `gsutil cp`. Good for private models.

The push Job needs registry credentials (mounted from a Secret).

### 3. Serving Config Generation

Generate the artifacts needed to update the inference engine:

**For vLLM Semantic Router (primary use case):**
- Update the semantic router's routing ConfigMap with the new model path/version
- The routing table maps query types to embedding models:

```yaml
# semantic-router-config ConfigMap
routing:
  models:
    - name: clip-vit-l-v2.1      # new version from training
      path: ghcr.io/rootfs/models/clip-vit-l:v2.1
      type: embedding
      dimensions: 768
      routes: ["image-search", "multimodal-retrieval"]
    - name: bge-large-en-v1.5    # existing model, unchanged
      path: ghcr.io/rootfs/models/bge-large:v1.5
      type: embedding
      dimensions: 1024
      routes: ["text-search", "rag"]
```

**For standalone vLLM:**
- Generate a patch for the vLLM Deployment's model argument
- Or generate a new Deployment manifest with the updated model path

### 4. Rollout Strategy

The agent generates deployment artifacts but does NOT apply them directly.
It pushes them to a branch for review. The actual rollout happens when:
- You merge the PR (manual)
- ArgoCD/Flux syncs the change (GitOps)
- A CI/CD pipeline applies the manifests (automation)

For canary support, generate two manifests:
1. A canary Deployment with the new model (low replica count)
2. A promotion script that scales up the canary and scales down the old version

## Conversion Job Design

```python
#!/usr/bin/env python3
"""Convert distributed training checkpoint to serving format."""
import torch
import os
import json
from pathlib import Path

CHECKPOINT_PATH = os.environ["CHECKPOINT_PATH"]
OUTPUT_PATH = os.environ["OUTPUT_PATH"]
TARGET_FORMAT = os.environ.get("TARGET_FORMAT", "safetensors")
QUANTIZATION = os.environ.get("QUANTIZATION", "")
MODEL_NAME = os.environ["MODEL_NAME"]

# Step 1: Consolidate sharded checkpoint
print(f"Consolidating checkpoint from {CHECKPOINT_PATH}")
state_dict = {}
shard_files = sorted(Path(CHECKPOINT_PATH).glob("*.pt"))
for shard in shard_files:
    state_dict.update(torch.load(shard, map_location="cpu", weights_only=True))

# Step 2: Convert to target format
if TARGET_FORMAT == "safetensors":
    from safetensors.torch import save_file
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    save_file(state_dict, f"{OUTPUT_PATH}/model.safetensors")
    # Write config.json, tokenizer files, etc.

elif TARGET_FORMAT == "gguf":
    # Use llama.cpp convert script
    pass

# Step 3: Optional quantization
if QUANTIZATION == "gptq-4bit":
    # Run GPTQ quantization
    pass

# Step 4: Write metadata
metadata = {
    "model_name": MODEL_NAME,
    "source_checkpoint": CHECKPOINT_PATH,
    "format": TARGET_FORMAT,
    "quantization": QUANTIZATION or "none",
}
with open(f"{OUTPUT_PATH}/install-metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Conversion complete: {OUTPUT_PATH}")
```

## Constraints

- The conversion Job must use GPU to load the model (especially for quantization).
- Conversion Jobs carry `kueue.x-k8s.io/queue-name: none` like other child resources.
- InstallSpec is optional — disabled by default, no impact on existing TrainJobs.
- The agent generates deployment artifacts as files in a branch. It never runs
  `kubectl apply` against the serving cluster. Deployment is a human/GitOps decision.
- Registry credentials come from K8s Secrets, never hardcoded.
- The conversion output path should be deterministic: `<registry>/<model>:<trainjob-name>-step-<N>`.
- After changes, run `go build ./...` and `go vet ./...`.

## Files to Read First

1. `internal/controller/checkpoint.go` — checkpoint validation Job builder (model for conversion Job structure)
2. `api/v1alpha1/types.go` — where to add InstallSpec
3. `internal/controller/trainjob_controller.go` — where conversion phase fits in the state machine
4. `agents/roles/model-eval/SKILL.md` — the eval agent runs before install; understand the handoff
5. `agents/manifests/vllm-deployment.yaml` — the vLLM deployment the serving config targets
