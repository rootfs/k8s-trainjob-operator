package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

// AgentTrace captures the full execution trace of a single agent run.
// Written to agents/traces/<run_id>.json on completion.
type AgentTrace struct {
	RunID        string       `json:"run_id"`
	Role         string       `json:"role"`
	Task         string       `json:"task,omitempty"`
	Branch       string       `json:"branch"`
	Model        string       `json:"model"`
	StartTime    time.Time    `json:"start_time"`
	EndTime      time.Time    `json:"end_time"`
	Turns        int          `json:"turns"`
	MaxTurns     int          `json:"max_turns"`
	ToolCalls    []ToolRecord `json:"tool_calls"`
	FilesRead    []string     `json:"files_read"`
	FilesChanged []string     `json:"files_changed"`
	ExitReason   string       `json:"exit_reason"` // "done", "max_turns", "error", "stop"
	ExitSummary  string       `json:"exit_summary,omitempty"`
	BuildPassed  *bool        `json:"build_passed,omitempty"`
	VetPassed    *bool        `json:"vet_passed,omitempty"`
	DiffStat     string       `json:"diff_stat,omitempty"`
	ErrorMessages []string    `json:"error_messages,omitempty"`
	DryRun       bool         `json:"dry_run"`
}

// ToolRecord captures a single tool invocation with timing and outcome.
type ToolRecord struct {
	Turn       int    `json:"turn"`
	ToolCallID string `json:"tool_call_id"`
	Tool       string `json:"tool"`
	Arguments  string `json:"arguments"`
	ResultLen  int    `json:"result_len"`
	Error      bool   `json:"error"`
	ErrorMsg   string `json:"error_msg,omitempty"`
	DurationMs int64  `json:"duration_ms"`
}

// TraceCollector accumulates trace data during an agent run.
type TraceCollector struct {
	trace     AgentTrace
	filesRead map[string]bool
	filesEdit map[string]bool
}

func newTraceCollector(runID, role, branch, model string, maxTurns int, dryRun bool) *TraceCollector {
	return &TraceCollector{
		trace: AgentTrace{
			RunID:     runID,
			Role:      role,
			Branch:    branch,
			Model:     model,
			StartTime: time.Now(),
			MaxTurns:  maxTurns,
			DryRun:    dryRun,
		},
		filesRead: make(map[string]bool),
		filesEdit: make(map[string]bool),
	}
}

func (tc *TraceCollector) recordToolCall(turn int, call ToolCall, result string, err error, duration time.Duration) {
	rec := ToolRecord{
		Turn:       turn,
		ToolCallID: call.ID,
		Tool:       call.Function.Name,
		Arguments:  call.Function.Arguments,
		ResultLen:  len(result),
		DurationMs: duration.Milliseconds(),
	}
	if err != nil {
		rec.Error = true
		rec.ErrorMsg = err.Error()
		tc.trace.ErrorMessages = append(tc.trace.ErrorMessages, fmt.Sprintf("turn %d: %s: %s", turn, call.Function.Name, err.Error()))
	}

	// Track file access patterns from arguments
	var args map[string]interface{}
	if json.Unmarshal([]byte(call.Function.Arguments), &args) == nil {
		if path, ok := args["path"].(string); ok && path != "" {
			switch call.Function.Name {
			case "read_file":
				tc.filesRead[path] = true
			case "edit_file", "write_file":
				tc.filesEdit[path] = true
			}
		}
	}

	tc.trace.ToolCalls = append(tc.trace.ToolCalls, rec)
}

func (tc *TraceCollector) finish(exitReason, exitSummary string, turn int) {
	tc.trace.EndTime = time.Now()
	tc.trace.Turns = turn
	tc.trace.ExitReason = exitReason
	tc.trace.ExitSummary = exitSummary

	tc.trace.FilesRead = make([]string, 0, len(tc.filesRead))
	for f := range tc.filesRead {
		tc.trace.FilesRead = append(tc.trace.FilesRead, f)
	}
	tc.trace.FilesChanged = make([]string, 0, len(tc.filesEdit))
	for f := range tc.filesEdit {
		tc.trace.FilesChanged = append(tc.trace.FilesChanged, f)
	}
}

func (tc *TraceCollector) setTask(task string)     { tc.trace.Task = task }
func (tc *TraceCollector) setBuild(passed bool)     { tc.trace.BuildPassed = &passed }
func (tc *TraceCollector) setVet(passed bool)       { tc.trace.VetPassed = &passed }
func (tc *TraceCollector) setDiffStat(stat string)  { tc.trace.DiffStat = stat }

// writeTrace serializes the trace to a JSON file in the traces directory.
func (tc *TraceCollector) writeTrace(workDir string) error {
	traceDir := filepath.Join(workDir, "agents", "traces")
	if err := os.MkdirAll(traceDir, 0755); err != nil {
		return fmt.Errorf("creating traces dir: %w", err)
	}

	filename := fmt.Sprintf("%s_%s.json", tc.trace.Role, tc.trace.RunID)
	data, err := json.MarshalIndent(tc.trace, "", "  ")
	if err != nil {
		return fmt.Errorf("marshaling trace: %w", err)
	}

	path := filepath.Join(traceDir, filename)
	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("writing trace: %w", err)
	}

	return nil
}
