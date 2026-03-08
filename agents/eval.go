package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// EvalResult is the output of evaluating a single agent trace.
type EvalResult struct {
	RunID       string        `json:"run_id"`
	Role        string        `json:"role"`
	Timestamp   time.Time     `json:"timestamp"`
	Scores      []Score       `json:"scores"`
	Overall     float64       `json:"overall"`
	Pass        bool          `json:"pass"`
	Findings    []Finding     `json:"findings"`
}

type Score struct {
	Name   string  `json:"name"`
	Value  float64 `json:"value"`   // 0.0 to 1.0
	Weight float64 `json:"weight"`
	Detail string  `json:"detail,omitempty"`
}

type Finding struct {
	Severity string `json:"severity"` // "error", "warning", "info"
	Scorer   string `json:"scorer"`
	Message  string `json:"message"`
}

// EvalReport aggregates results across multiple traces for trend analysis.
type EvalReport struct {
	GeneratedAt  time.Time             `json:"generated_at"`
	TracesCount  int                   `json:"traces_count"`
	Results      []EvalResult          `json:"results"`
	RoleSummary  map[string]RoleStats  `json:"role_summary"`
	TopIssues    []Finding             `json:"top_issues"`
}

type RoleStats struct {
	Runs         int     `json:"runs"`
	PassRate     float64 `json:"pass_rate"`
	AvgScore     float64 `json:"avg_score"`
	AvgTurns     float64 `json:"avg_turns"`
	BuildPassPct float64 `json:"build_pass_pct"`
	ScopeViolations int  `json:"scope_violations"`
}

func runEval() {
	workDir := envOrDefault("AGENT_WORKDIR", ".")
	traceDir := filepath.Join(workDir, "agents", "traces")
	reportDir := filepath.Join(workDir, "agents", "eval-reports")

	traceFilter := envOrDefault("EVAL_TRACE", "")  // specific trace file, or empty for all
	goldenDir := filepath.Join(workDir, "agents", "golden")

	traces, err := loadTraces(traceDir, traceFilter)
	if err != nil {
		log.Fatalf("loading traces: %v", err)
	}
	if len(traces) == 0 {
		log.Println("no traces found to evaluate")
		return
	}
	log.Printf("evaluating %d trace(s)", len(traces))

	goldenCases, _ := loadGoldenCases(goldenDir)

	vllmEndpoint := envOrDefault("VLLM_ENDPOINT", "")
	vllmModel := envOrDefault("VLLM_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct")
	enableJudge := vllmEndpoint != ""

	var results []EvalResult
	for _, trace := range traces {
		result := evaluateTrace(trace, goldenCases, workDir, enableJudge, vllmEndpoint, vllmModel)
		results = append(results, result)
		logResult(result)
	}

	report := buildReport(results)

	if err := os.MkdirAll(reportDir, 0755); err != nil {
		log.Fatalf("creating report dir: %v", err)
	}
	reportPath := filepath.Join(reportDir, fmt.Sprintf("eval_%d.json", time.Now().Unix()))
	data, _ := json.MarshalIndent(report, "", "  ")
	if err := os.WriteFile(reportPath, data, 0644); err != nil {
		log.Fatalf("writing report: %v", err)
	}
	log.Printf("eval report written: %s", reportPath)

	printReportSummary(report)
}

func loadTraces(dir, filter string) ([]AgentTrace, error) {
	if filter != "" {
		data, err := os.ReadFile(filter)
		if err != nil {
			return nil, err
		}
		var t AgentTrace
		if err := json.Unmarshal(data, &t); err != nil {
			return nil, err
		}
		return []AgentTrace{t}, nil
	}

	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}

	var traces []AgentTrace
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		data, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil {
			log.Printf("warning: skipping %s: %v", e.Name(), err)
			continue
		}
		var t AgentTrace
		if err := json.Unmarshal(data, &t); err != nil {
			log.Printf("warning: skipping %s: %v", e.Name(), err)
			continue
		}
		traces = append(traces, t)
	}
	return traces, nil
}

// GoldenCase defines expected behavior for a specific task.
type GoldenCase struct {
	Role          string   `json:"role"`
	Task          string   `json:"task"`
	ExpectedFiles []string `json:"expected_files"` // files the agent should modify
	ForbiddenTools []string `json:"forbidden_tools,omitempty"`
	MustCallTools  []string `json:"must_call_tools,omitempty"`
	MaxTurns       int      `json:"max_turns,omitempty"`
}

func loadGoldenCases(dir string) (map[string]GoldenCase, error) {
	cases := make(map[string]GoldenCase)
	entries, err := os.ReadDir(dir)
	if err != nil {
		return cases, err
	}
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		data, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil {
			continue
		}
		var gc GoldenCase
		if err := json.Unmarshal(data, &gc); err != nil {
			continue
		}
		key := gc.Role + "/" + gc.Task
		cases[key] = gc
	}
	return cases, nil
}

func evaluateTrace(trace AgentTrace, golden map[string]GoldenCase, workDir string, enableJudge bool, vllmEndpoint, vllmModel string) EvalResult {
	result := EvalResult{
		RunID:     trace.RunID,
		Role:      trace.Role,
		Timestamp: time.Now(),
	}

	result.Scores = append(result.Scores, scoreBuildGate(trace))
	result.Scores = append(result.Scores, scoreVetGate(trace))
	result.Scores = append(result.Scores, scoreScopeAdherence(trace))
	result.Scores = append(result.Scores, scoreProtocolCompliance(trace))
	result.Scores = append(result.Scores, scoreEfficiency(trace))
	result.Scores = append(result.Scores, scoreToolErrorRate(trace))
	result.Scores = append(result.Scores, scoreCompletionSignal(trace))

	if gc, ok := golden[trace.Role+"/"+trace.Task]; ok {
		result.Scores = append(result.Scores, scoreGoldenMatch(trace, gc))
	}

	if enableJudge {
		judgeScore, judgeFindings := scoreLLMJudge(trace, vllmEndpoint, vllmModel)
		result.Scores = append(result.Scores, judgeScore)
		result.Findings = append(result.Findings, judgeFindings...)
	}

	// Collect findings from all scorers
	for _, s := range result.Scores {
		if s.Value < 0.5 {
			result.Findings = append(result.Findings, Finding{
				Severity: "error",
				Scorer:   s.Name,
				Message:  s.Detail,
			})
		} else if s.Value < 0.8 {
			result.Findings = append(result.Findings, Finding{
				Severity: "warning",
				Scorer:   s.Name,
				Message:  s.Detail,
			})
		}
	}

	// Weighted overall score
	var totalWeight, weightedSum float64
	for _, s := range result.Scores {
		totalWeight += s.Weight
		weightedSum += s.Value * s.Weight
	}
	if totalWeight > 0 {
		result.Overall = weightedSum / totalWeight
	}
	result.Pass = result.Overall >= 0.6 && !hasCriticalFailure(result.Scores)

	return result
}

func hasCriticalFailure(scores []Score) bool {
	for _, s := range scores {
		if (s.Name == "build_gate" || s.Name == "scope_adherence") && s.Value == 0 {
			return true
		}
	}
	return false
}

// --- Deterministic Scorers ---

func scoreBuildGate(t AgentTrace) Score {
	s := Score{Name: "build_gate", Weight: 3.0}
	if t.BuildPassed == nil {
		s.Value = 0.5
		s.Detail = "build status not recorded"
		return s
	}
	if *t.BuildPassed {
		s.Value = 1.0
		s.Detail = "go build passed"
	} else {
		s.Value = 0.0
		s.Detail = "go build FAILED — agent produced code that does not compile"
	}
	return s
}

func scoreVetGate(t AgentTrace) Score {
	s := Score{Name: "vet_gate", Weight: 2.0}
	if t.VetPassed == nil {
		s.Value = 0.5
		s.Detail = "vet status not recorded"
		return s
	}
	if *t.VetPassed {
		s.Value = 1.0
		s.Detail = "go vet passed"
	} else {
		s.Value = 0.3
		s.Detail = "go vet FAILED — static analysis found issues"
	}
	return s
}

func scoreScopeAdherence(t AgentTrace) Score {
	s := Score{Name: "scope_adherence", Weight: 2.5}
	role, ok := roles[t.Role]
	if !ok {
		s.Value = 0.5
		s.Detail = fmt.Sprintf("unknown role %q, cannot check scope", t.Role)
		return s
	}

	violations := 0
	var violatedFiles []string
	for _, f := range t.FilesChanged {
		if !isInScope(f, role.WatchPaths) {
			violations++
			violatedFiles = append(violatedFiles, f)
		}
	}

	if violations == 0 {
		s.Value = 1.0
		s.Detail = fmt.Sprintf("all %d changed files within scope", len(t.FilesChanged))
	} else {
		s.Value = 1.0 - float64(violations)/float64(len(t.FilesChanged))
		if s.Value < 0 {
			s.Value = 0
		}
		s.Detail = fmt.Sprintf("%d file(s) outside scope: %s", violations, strings.Join(violatedFiles, ", "))
	}
	return s
}

func isInScope(file string, watchPaths []string) bool {
	// AGENCY.md and traces are always in scope (coordination files)
	if strings.HasPrefix(file, "agents/") {
		return true
	}
	for _, wp := range watchPaths {
		if strings.HasSuffix(wp, "/") {
			if strings.HasPrefix(file, wp) {
				return true
			}
		} else if file == wp {
			return true
		}
	}
	return false
}

func scoreProtocolCompliance(t AgentTrace) Score {
	s := Score{Name: "protocol_compliance", Weight: 1.5}

	checks := 0
	passed := 0

	// Check 1: Did the agent read AGENCY.md?
	checks++
	readAgency := false
	for _, tc := range t.ToolCalls {
		if tc.Tool == "read_file" {
			var args map[string]interface{}
			if json.Unmarshal([]byte(tc.Arguments), &args) == nil {
				if path, ok := args["path"].(string); ok && strings.Contains(path, "AGENCY.md") {
					readAgency = true
					break
				}
			}
		}
	}
	if readAgency {
		passed++
	}

	// Check 2: Did the agent read files before editing them?
	checks++
	readsBeforeEdits := true
	editedFiles := make(map[string]int) // file -> first edit turn
	readFiles := make(map[string]int)   // file -> first read turn

	for _, tc := range t.ToolCalls {
		var args map[string]interface{}
		if json.Unmarshal([]byte(tc.Arguments), &args) != nil {
			continue
		}
		path, _ := args["path"].(string)
		if path == "" {
			continue
		}
		switch tc.Tool {
		case "read_file":
			if _, exists := readFiles[path]; !exists {
				readFiles[path] = tc.Turn
			}
		case "edit_file":
			if _, exists := editedFiles[path]; !exists {
				editedFiles[path] = tc.Turn
			}
		}
	}
	for file, editTurn := range editedFiles {
		readTurn, wasRead := readFiles[file]
		if !wasRead || readTurn >= editTurn {
			readsBeforeEdits = false
			break
		}
	}
	if readsBeforeEdits {
		passed++
	}

	// Check 3: Did the agent call done()?
	checks++
	if t.ExitReason == "done" {
		passed++
	}

	// Check 4: Did the agent run build verification?
	checks++
	ranBuild := false
	for _, tc := range t.ToolCalls {
		if tc.Tool == "run_command" {
			var args map[string]interface{}
			if json.Unmarshal([]byte(tc.Arguments), &args) == nil {
				cmd, _ := args["command"].(string)
				if strings.Contains(cmd, "go build") || strings.Contains(cmd, "go vet") || strings.Contains(cmd, "make") {
					ranBuild = true
					break
				}
			}
		}
	}
	if ranBuild {
		passed++
	}

	s.Value = float64(passed) / float64(checks)
	var issues []string
	if !readAgency {
		issues = append(issues, "did not read AGENCY.md")
	}
	if !readsBeforeEdits {
		issues = append(issues, "edited files without reading them first")
	}
	if t.ExitReason != "done" {
		issues = append(issues, fmt.Sprintf("exit reason was %q, not done()", t.ExitReason))
	}
	if !ranBuild {
		issues = append(issues, "did not run build verification")
	}
	if len(issues) == 0 {
		s.Detail = "all protocol checks passed"
	} else {
		s.Detail = strings.Join(issues, "; ")
	}
	return s
}

func scoreEfficiency(t AgentTrace) Score {
	s := Score{Name: "efficiency", Weight: 1.0}

	if t.Turns == 0 {
		s.Value = 0.0
		s.Detail = "zero turns recorded"
		return s
	}

	ratio := float64(t.Turns) / float64(t.MaxTurns)
	if ratio <= 0.3 {
		s.Value = 1.0 // used 30% or fewer turns — efficient
	} else if ratio <= 0.6 {
		s.Value = 0.8
	} else if ratio <= 0.85 {
		s.Value = 0.5
	} else {
		s.Value = 0.2
	}

	filesPerTurn := float64(len(t.FilesChanged)) / float64(t.Turns)
	s.Detail = fmt.Sprintf("%d turns / %d max (%.0f%%), %.1f files changed per turn", t.Turns, t.MaxTurns, ratio*100, filesPerTurn)
	return s
}

func scoreToolErrorRate(t AgentTrace) Score {
	s := Score{Name: "tool_error_rate", Weight: 1.0}
	if len(t.ToolCalls) == 0 {
		s.Value = 1.0
		s.Detail = "no tool calls"
		return s
	}

	errors := 0
	for _, tc := range t.ToolCalls {
		if tc.Error {
			errors++
		}
	}

	errorRate := float64(errors) / float64(len(t.ToolCalls))
	s.Value = 1.0 - errorRate
	if s.Value < 0 {
		s.Value = 0
	}
	s.Detail = fmt.Sprintf("%d errors in %d tool calls (%.0f%% error rate)", errors, len(t.ToolCalls), errorRate*100)
	return s
}

func scoreCompletionSignal(t AgentTrace) Score {
	s := Score{Name: "completion_signal", Weight: 1.5}
	switch t.ExitReason {
	case "done":
		s.Value = 1.0
		s.Detail = "agent called done() cleanly"
	case "stop":
		s.Value = 0.7
		s.Detail = "LLM stopped without calling done()"
	case "max_turns":
		s.Value = 0.2
		s.Detail = "hit max turns limit — agent did not finish"
	case "error":
		s.Value = 0.0
		s.Detail = "agent errored: " + t.ExitSummary
	default:
		s.Value = 0.3
		s.Detail = "unknown exit: " + t.ExitReason
	}
	return s
}

func scoreGoldenMatch(t AgentTrace, gc GoldenCase) Score {
	s := Score{Name: "golden_match", Weight: 2.0}

	checks := 0
	passed := 0

	// Check expected files were modified
	if len(gc.ExpectedFiles) > 0 {
		for _, ef := range gc.ExpectedFiles {
			checks++
			for _, fc := range t.FilesChanged {
				if fc == ef {
					passed++
					break
				}
			}
		}
	}

	// Check forbidden tools were not called
	if len(gc.ForbiddenTools) > 0 {
		for _, ft := range gc.ForbiddenTools {
			checks++
			called := false
			for _, tc := range t.ToolCalls {
				if tc.Tool == ft {
					called = true
					break
				}
			}
			if !called {
				passed++
			}
		}
	}

	// Check required tools were called
	if len(gc.MustCallTools) > 0 {
		for _, mt := range gc.MustCallTools {
			checks++
			for _, tc := range t.ToolCalls {
				if tc.Tool == mt {
					passed++
					break
				}
			}
		}
	}

	// Check turn limit
	if gc.MaxTurns > 0 {
		checks++
		if t.Turns <= gc.MaxTurns {
			passed++
		}
	}

	if checks == 0 {
		s.Value = 1.0
		s.Detail = "no golden assertions defined"
	} else {
		s.Value = float64(passed) / float64(checks)
		s.Detail = fmt.Sprintf("%d/%d golden assertions passed", passed, checks)
	}
	return s
}

// --- LLM-as-Judge Scorer ---

func scoreLLMJudge(t AgentTrace, endpoint, model string) (Score, []Finding) {
	s := Score{Name: "llm_judge", Weight: 2.0}

	toolSummary := summarizeToolCalls(t)
	prompt := fmt.Sprintf(`You are evaluating an AI agent's work on a Kubernetes operator codebase.

Role: %s
Task: %s
Exit reason: %s
Summary: %s
Turns used: %d / %d
Files changed: %s
Tool call summary:
%s

Evaluate the agent's performance on these dimensions:
1. Did it accomplish its stated task?
2. Were the code changes focused and minimal?
3. Did it follow good software engineering practices?
4. Were there wasted or unnecessary actions?

Respond with ONLY a JSON object:
{
  "score": <float 0.0 to 1.0>,
  "reasoning": "<one paragraph>",
  "issues": ["<issue1>", "<issue2>"]
}`, t.Role, t.Task, t.ExitReason, t.ExitSummary, t.Turns, t.MaxTurns,
		strings.Join(t.FilesChanged, ", "), toolSummary)

	client := &http.Client{Timeout: 2 * time.Minute}
	reqBody := ChatRequest{
		Model: model,
		Messages: []ChatMessage{
			{Role: "user", Content: prompt},
		},
		MaxTokens: 1024,
	}
	data, _ := json.Marshal(reqBody)
	url := strings.TrimSuffix(endpoint, "/") + "/chat/completions"
	req, _ := http.NewRequest("POST", url, bytes.NewReader(data))
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		s.Value = 0.5
		s.Detail = fmt.Sprintf("LLM judge unavailable: %v", err)
		return s, nil
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	var chatResp ChatResponse
	if err := json.Unmarshal(body, &chatResp); err != nil || len(chatResp.Choices) == 0 {
		s.Value = 0.5
		s.Detail = "LLM judge returned invalid response"
		return s, nil
	}

	content := chatResp.Choices[0].Message.Content
	// Extract JSON from response (may be wrapped in markdown code blocks)
	content = strings.TrimPrefix(content, "```json")
	content = strings.TrimPrefix(content, "```")
	content = strings.TrimSuffix(content, "```")
	content = strings.TrimSpace(content)

	var judgeResult struct {
		JudgeScore float64  `json:"score"`
		Reasoning  string   `json:"reasoning"`
		Issues     []string `json:"issues"`
	}
	if err := json.Unmarshal([]byte(content), &judgeResult); err != nil {
		s.Value = 0.5
		s.Detail = "could not parse judge response"
		return s, nil
	}

	s.Value = judgeResult.JudgeScore
	s.Detail = judgeResult.Reasoning

	var findings []Finding
	for _, issue := range judgeResult.Issues {
		findings = append(findings, Finding{
			Severity: "warning",
			Scorer:   "llm_judge",
			Message:  issue,
		})
	}
	return s, findings
}

func summarizeToolCalls(t AgentTrace) string {
	var lines []string
	for _, tc := range t.ToolCalls {
		status := "ok"
		if tc.Error {
			status = "ERROR: " + tc.ErrorMsg
		}
		lines = append(lines, fmt.Sprintf("  turn %d: %s (%dms) -> %s", tc.Turn, tc.Tool, tc.DurationMs, status))
	}
	return strings.Join(lines, "\n")
}

// --- Report Generation ---

func buildReport(results []EvalResult) EvalReport {
	report := EvalReport{
		GeneratedAt: time.Now(),
		TracesCount: len(results),
		Results:     results,
		RoleSummary: make(map[string]RoleStats),
	}

	// Aggregate per-role stats
	roleResults := make(map[string][]EvalResult)
	for _, r := range results {
		roleResults[r.Role] = append(roleResults[r.Role], r)
	}

	for role, rr := range roleResults {
		stats := RoleStats{Runs: len(rr)}
		var scoreSum, turnSum float64
		var passCount, buildCount, scopeViolations int

		for _, r := range rr {
			scoreSum += r.Overall
			if r.Pass {
				passCount++
			}
			// Find specific scores
			for _, s := range r.Scores {
				switch s.Name {
				case "build_gate":
					if s.Value == 1.0 {
						buildCount++
					}
				case "scope_adherence":
					if s.Value < 1.0 {
						scopeViolations++
					}
				}
			}
		}

		for _, tr := range rr {
			turnSum += float64(tr.Overall) // use trace turns from trace data
		}

		stats.PassRate = float64(passCount) / float64(len(rr))
		stats.AvgScore = scoreSum / float64(len(rr))
		stats.BuildPassPct = float64(buildCount) / float64(len(rr))
		stats.ScopeViolations = scopeViolations
		report.RoleSummary[role] = stats
	}

	// Collect top issues (most frequent findings)
	issueCounts := make(map[string]int)
	issueMap := make(map[string]Finding)
	for _, r := range results {
		for _, f := range r.Findings {
			key := f.Scorer + ": " + f.Message
			issueCounts[key]++
			issueMap[key] = f
		}
	}

	type issueCount struct {
		key   string
		count int
	}
	var sorted []issueCount
	for k, c := range issueCounts {
		sorted = append(sorted, issueCount{k, c})
	}
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].count > sorted[j].count })

	for i, ic := range sorted {
		if i >= 10 {
			break
		}
		report.TopIssues = append(report.TopIssues, issueMap[ic.key])
	}

	return report
}

func logResult(r EvalResult) {
	status := "PASS"
	if !r.Pass {
		status = "FAIL"
	}
	log.Printf("[%s] %s/%s overall=%.2f", status, r.Role, r.RunID, r.Overall)
	for _, s := range r.Scores {
		log.Printf("  %-22s %.2f  %s", s.Name, s.Value, truncate(s.Detail, 80))
	}
	for _, f := range r.Findings {
		if f.Severity == "error" {
			log.Printf("  ERROR: [%s] %s", f.Scorer, f.Message)
		}
	}
}

func printReportSummary(report EvalReport) {
	fmt.Println("\n=== Agent Eval Report ===")
	fmt.Printf("Traces evaluated: %d\n", report.TracesCount)
	fmt.Printf("Generated: %s\n\n", report.GeneratedAt.Format(time.RFC3339))

	fmt.Println("Per-Role Summary:")
	fmt.Printf("  %-16s %6s %8s %8s %10s\n", "Role", "Runs", "PassRate", "AvgScore", "BuildPass%")
	fmt.Printf("  %-16s %6s %8s %8s %10s\n", "----", "----", "--------", "--------", "----------")
	for role, stats := range report.RoleSummary {
		fmt.Printf("  %-16s %6d %7.0f%% %8.2f %9.0f%%\n",
			role, stats.Runs, stats.PassRate*100, stats.AvgScore, stats.BuildPassPct*100)
	}

	if len(report.TopIssues) > 0 {
		fmt.Println("\nTop Issues:")
		for i, issue := range report.TopIssues {
			fmt.Printf("  %d. [%s] %s: %s\n", i+1, issue.Severity, issue.Scorer, issue.Message)
		}
	}
	fmt.Println()
}
