IMG ?= ghcr.io/rootfs/trainjob-operator:latest
BINARY ?= bin/trainjob-operator

.PHONY: all build test lint fmt vet clean docker-build docker-push run help

all: build

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

docker-build: ## Build the Docker image
	docker build -t $(IMG) .

docker-push: ## Push the Docker image
	docker push $(IMG)

tidy: ## Run go mod tidy
	go mod tidy

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'
