package webhook

import aiv1 "github.com/rootfs/trainjob-operator/api/v1alpha1"

// ModelRegistry holds known model architectures for auto-configuration.
// In production, this would be backed by a ConfigMap or external service
// so admins can add models without recompiling the operator.
var ModelRegistry = map[string]aiv1.ModelArchSpec{
	"llama-3-405b": {
		ParamsBillions:          405,
		HiddenDim:               16384,
		NumLayers:               126,
		NumHeads:                128,
		NumKVHeads:              8,
		SeqLen:                  8192,
		MicroBatchSize:          1,
		ActivationCheckpointing: true,
	},
	"llama-3-70b": {
		ParamsBillions:          70,
		HiddenDim:               8192,
		NumLayers:               80,
		NumHeads:                64,
		NumKVHeads:              8,
		SeqLen:                  8192,
		MicroBatchSize:          2,
		ActivationCheckpointing: true,
	},
	"llama-3-8b": {
		ParamsBillions:          8,
		HiddenDim:               4096,
		NumLayers:               32,
		NumHeads:                32,
		NumKVHeads:              8,
		SeqLen:                  8192,
		MicroBatchSize:          4,
		ActivationCheckpointing: false,
	},
	"llama-3-1b": {
		ParamsBillions:          1.3,
		HiddenDim:               2048,
		NumLayers:               16,
		NumHeads:                16,
		NumKVHeads:              8,
		SeqLen:                  8192,
		MicroBatchSize:          8,
		ActivationCheckpointing: false,
	},
	"mixtral-8x7b": {
		ParamsBillions:          46.7,
		HiddenDim:               4096,
		NumLayers:               32,
		NumHeads:                32,
		NumKVHeads:              8,
		SeqLen:                  32768,
		MicroBatchSize:          1,
		ActivationCheckpointing: true,
	},
	"mistral-7b": {
		ParamsBillions:          7.3,
		HiddenDim:               4096,
		NumLayers:               32,
		NumHeads:                32,
		NumKVHeads:              8,
		SeqLen:                  8192,
		MicroBatchSize:          4,
		ActivationCheckpointing: false,
	},
	"gpt-4-scale": {
		ParamsBillions:          175,
		HiddenDim:               12288,
		NumLayers:               96,
		NumHeads:                96,
		NumKVHeads:              96,
		SeqLen:                  8192,
		MicroBatchSize:          1,
		ActivationCheckpointing: true,
	},
}

// GPUSpec describes a GPU's capabilities for the auto-configurator.
type GPUSpec struct {
	MemoryGB     float64
	NVLinkBWGBps float64 // intra-node NVLink bandwidth
	IBBWGBps     float64 // inter-node InfiniBand bandwidth
	SupportsFP8  bool
}

var GPURegistry = map[string]GPUSpec{
	"NVIDIA-H100-SXM5-80GB": {
		MemoryGB:     72.0, // 80 - ~8 overhead
		NVLinkBWGBps: 900.0,
		IBBWGBps:     50.0,
		SupportsFP8:  true,
	},
	"NVIDIA-H200-SXM-141GB": {
		MemoryGB:     133.0,
		NVLinkBWGBps: 900.0,
		IBBWGBps:     50.0,
		SupportsFP8:  true,
	},
	"NVIDIA-A100-SXM4-80GB": {
		MemoryGB:     72.0,
		NVLinkBWGBps: 600.0,
		IBBWGBps:     25.0,
		SupportsFP8:  false,
	},
	"NVIDIA-A100-SXM4-40GB": {
		MemoryGB:     34.0,
		NVLinkBWGBps: 600.0,
		IBBWGBps:     25.0,
		SupportsFP8:  false,
	},
	"NVIDIA-L40S-48GB": {
		MemoryGB:     42.0,
		NVLinkBWGBps: 0, // no NVLink
		IBBWGBps:     12.5,
		SupportsFP8:  true,
	},
}

func lookupGPU(gpuType string) GPUSpec {
	if spec, ok := GPURegistry[gpuType]; ok {
		return spec
	}
	return GPUSpec{MemoryGB: 72.0, NVLinkBWGBps: 600.0, IBBWGBps: 25.0, SupportsFP8: false}
}
