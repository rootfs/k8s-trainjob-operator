IMG ?= ghcr.io/rootfs/trainjob-operator:latest
AGENT_IMG ?= ghcr.io/rootfs/trainjob-agent:latest
BINARY ?= bin/trainjob-operator
AGENT_BINARY ?= bin/trainjob-agent
ROLE ?= model-builder

.PHONY: all build test lint fmt vet clean docker-build docker-push run help \
	agent-build agent-docker-build agent-docker-push agent-run agent-deploy agent-clean

all: build

# ── Operator ──

build: fmt vet ## Build the operator binary
	go build -o $(BINARY) ./cmd/main.go

run: fmt vet ## Run the operator locally (requires kubeconfig)
	go run ./cmd/main.go

test: fmt vet ## Run unit and envtest tests
	go test ./... -coverprofile cover.out

fmt: ## Run go fmt
	go fmt ./...

vet: ## Run go vet
	go vet ./...

lint: ## Run golangci-lint (must be installed)
	golangci-lint run ./...

clean: ## Remove build artifacts
	rm -rf bin/ cover.out

docker-build: ## Build the operator Docker image
	docker build -t $(IMG) .

docker-push: ## Push the operator Docker image
	docker push $(IMG)

tidy: ## Run go mod tidy
	go mod tidy

# ── Agents ──

agent-build: ## Build the agent binary
	cd agents && go build -o ../$(AGENT_BINARY) .

agent-docker-build: ## Build the agent Docker image
	docker build -t $(AGENT_IMG) -f agents/Dockerfile .

agent-docker-push: ## Push the agent Docker image
	docker push $(AGENT_IMG)

agent-run: ## Run an agent locally (set ROLE=, AGENT_REPO_URL=, VLLM_ENDPOINT=)
	cd agents && AGENT_ROLE=$(ROLE) AGENT_DRY_RUN=true go run .

agent-deploy: ## Deploy agent infrastructure to K8s (namespace, RBAC, vLLM)
	kubectl apply -f agents/manifests/rbac.yaml
	kubectl apply -f agents/manifests/git-secret.yaml
	kubectl apply -f agents/manifests/vllm-deployment.yaml
	kubectl apply -f agents/manifests/agent-cronjob.yaml

agent-job: ## Launch a one-shot agent Job on K8s (set ROLE=model-builder)
	sed 's/AGENT_ROLE_PLACEHOLDER/$(ROLE)/g' agents/manifests/agent-job-template.yaml | kubectl apply -f -

agent-clean: ## Remove agent resources from K8s
	kubectl delete namespace trainjob-agents --ignore-not-found

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
