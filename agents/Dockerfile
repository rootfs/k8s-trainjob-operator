FROM golang:1.24 AS builder
WORKDIR /build
COPY agents/go.mod agents/go.sum ./
RUN go mod download
COPY agents/*.go ./
RUN CGO_ENABLED=0 GOOS=linux go build -a -o agent .

FROM alpine:3.20
RUN apk add --no-cache git bash grep ca-certificates openssh-client && \
    adduser -D -u 65532 agent
WORKDIR /workspace
COPY --from=builder /build/agent /usr/local/bin/agent
USER 65532
ENTRYPOINT ["agent"]
