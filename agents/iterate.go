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

// IterateResult captures the output of the iterate phase:
// what patterns were found and what SKILL.md changes are proposed.
type IterateResult struct {
	GeneratedAt   time.Time          `json:"generated_at"`
	TracesAnalyzed int               `json:"traces_analyzed"`
	ReportsUsed    int               `json:"reports_used"`
	Patterns       []FailurePattern  `json:"patterns"`
	Proposals      []SkillProposal   `json:"proposals"`
}

// FailurePattern is a recurring issue detected across multiple agent traces.
type FailurePattern struct {
	Role       string   `json:"role"`
	Category   string   `json:"category"`    // "scope", "protocol", "build", "efficiency", "quality"
	Frequency  int      `json:"frequency"`
	Severity   string   `json:"severity"`    // "high", "medium", "low"
	Examples   []string `json:"examples"`
	RootCause  string   `json:"root_cause"`
}

// SkillProposal is a concrete suggestion to modify a role's SKILL.md.
type SkillProposal struct {
	Role       string `json:"role"`
	Section    string `json:"section"`     // which section of SKILL.md to update
	Action     string `json:"action"`      // "add", "modify", "remove"
	Content    string `json:"content"`     // the proposed text
	Rationale  string `json:"rationale"`   // why this change would help
	Priority   string `json:"priority"`    // "high", "medium", "low"
	Pattern    string `json:"pattern"`     // which failure pattern this addresses
}

func runIterate() {
	workDir := envOrDefault("AGENT_WORKDIR", ".")
	reportDir := filepath.Join(workDir, "agents", "eval-reports")
	traceDir := filepath.Join(workDir, "agents", "traces")
	outputDir := filepath.Join(workDir, "agents", "iterate-reports")
	applyChanges := envOrDefault("ITERATE_APPLY", "false") == "true"

	vllmEndpoint := envOrDefault("VLLM_ENDPOINT", "")
	vllmModel := envOrDefault("VLLM_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct")

	// Load all eval reports
	reports, err := loadEvalReports(reportDir)
	if err != nil {
		log.Printf("warning: no eval reports found: %v", err)
	}

	// Load all traces
	traces, err := loadTraces(traceDir, "")
	if err != nil {
		log.Printf("warning: no traces found: %v", err)
	}

	if len(reports) == 0 && len(traces) == 0 {
		log.Println("no data to iterate on — run agents and eval first")
		return
	}

	log.Printf("iterate: analyzing %d report(s) and %d trace(s)", len(reports), len(traces))

	// Phase 1: Extract failure patterns from deterministic analysis
	patterns := extractPatterns(reports, traces)

	// Phase 2: Use LLM to propose SKILL.md improvements
	var proposals []SkillProposal
	if vllmEndpoint != "" {
		proposals = proposeImprovements(patterns, traces, workDir, vllmEndpoint, vllmModel)
	} else {
		proposals = proposeImprovementsHeuristic(patterns)
	}

	result := IterateResult{
		GeneratedAt:    time.Now(),
		TracesAnalyzed: len(traces),
		ReportsUsed:    len(reports),
		Patterns:       patterns,
		Proposals:      proposals,
	}

	// Write iterate report
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		log.Fatalf("creating output dir: %v", err)
	}
	reportPath := filepath.Join(outputDir, fmt.Sprintf("iterate_%d.json", time.Now().Unix()))
	data, _ := json.MarshalIndent(result, "", "  ")
	if err := os.WriteFile(reportPath, data, 0644); err != nil {
		log.Fatalf("writing iterate report: %v", err)
	}
	log.Printf("iterate report written: %s", reportPath)

	printIterateResult(result)

	// Phase 3: Optionally apply proposals to SKILL.md files
	if applyChanges && len(proposals) > 0 {
		applyProposals(workDir, proposals)
	}
}

func loadEvalReports(dir string) ([]EvalReport, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var reports []EvalReport
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		data, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil {
			continue
		}
		var r EvalReport
		if err := json.Unmarshal(data, &r); err != nil {
			continue
		}
		reports = append(reports, r)
	}
	return reports, nil
}

// extractPatterns finds recurring failure modes across eval reports and traces.
func extractPatterns(reports []EvalReport, traces []AgentTrace) []FailurePattern {
	type patternKey struct {
		role     string
		category string
	}
	counts := make(map[patternKey]int)
	examples := make(map[patternKey][]string)

	// From eval reports
	for _, report := range reports {
		for _, result := range report.Results {
			for _, score := range result.Scores {
				if score.Value >= 0.8 {
					continue
				}
				key := patternKey{role: result.Role, category: score.Name}
				counts[key]++
				if len(examples[key]) < 3 {
					examples[key] = append(examples[key], fmt.Sprintf("run %s: %s", result.RunID, score.Detail))
				}
			}
		}
	}

	// From raw traces: detect behavioral anti-patterns
	for _, t := range traces {
		// Anti-pattern: too many read_file calls without edits (analysis paralysis)
		reads, edits := 0, 0
		for _, tc := range t.ToolCalls {
			switch tc.Tool {
			case "read_file":
				reads++
			case "edit_file", "write_file":
				edits++
			}
		}
		if reads > 10 && edits == 0 {
			key := patternKey{role: t.Role, category: "analysis_paralysis"}
			counts[key]++
			if len(examples[key]) < 3 {
				examples[key] = append(examples[key],
					fmt.Sprintf("run %s: %d reads, 0 edits in %d turns", t.RunID, reads, t.Turns))
			}
		}

		// Anti-pattern: repeated tool errors on the same tool
		errorTools := make(map[string]int)
		for _, tc := range t.ToolCalls {
			if tc.Error {
				errorTools[tc.Tool]++
			}
		}
		for tool, count := range errorTools {
			if count >= 3 {
				key := patternKey{role: t.Role, category: "repeated_tool_error"}
				counts[key]++
				if len(examples[key]) < 3 {
					examples[key] = append(examples[key],
						fmt.Sprintf("run %s: %s failed %d times", t.RunID, tool, count))
				}
			}
		}

		// Anti-pattern: max_turns exit (ran out of budget)
		if t.ExitReason == "max_turns" {
			key := patternKey{role: t.Role, category: "budget_exhaustion"}
			counts[key]++
			if len(examples[key]) < 3 {
				examples[key] = append(examples[key],
					fmt.Sprintf("run %s: hit %d turns without finishing", t.RunID, t.MaxTurns))
			}
		}
	}

	// Convert to sorted list
	var patterns []FailurePattern
	for key, count := range counts {
		severity := "low"
		if count >= 5 {
			severity = "high"
		} else if count >= 2 {
			severity = "medium"
		}
		patterns = append(patterns, FailurePattern{
			Role:      key.role,
			Category:  key.category,
			Frequency: count,
			Severity:  severity,
			Examples:  examples[key],
			RootCause: inferRootCause(key.category),
		})
	}

	sort.Slice(patterns, func(i, j int) bool {
		return patterns[i].Frequency > patterns[j].Frequency
	})

	return patterns
}

func inferRootCause(category string) string {
	switch category {
	case "build_gate":
		return "Agent produces code that does not compile. SKILL.md may need stronger emphasis on build verification before committing."
	case "vet_gate":
		return "Agent introduces go vet violations. SKILL.md should remind the agent to run go vet and address warnings."
	case "scope_adherence":
		return "Agent modifies files outside its WatchPaths. SKILL.md needs clearer boundaries on which files to touch."
	case "protocol_compliance":
		return "Agent skips protocol steps (reading AGENCY.md, reading before editing, calling done()). SKILL.md working protocol section may be unclear."
	case "efficiency":
		return "Agent uses too many turns for the task. SKILL.md may need to encourage more focused, direct approaches."
	case "tool_error_rate":
		return "Agent frequently gets tool call errors. May need better examples of correct tool usage in SKILL.md."
	case "analysis_paralysis":
		return "Agent reads many files but never makes edits. SKILL.md may need to emphasize taking action after understanding."
	case "repeated_tool_error":
		return "Agent retries the same failing tool call. SKILL.md should instruct the agent to try a different approach after 2 failures."
	case "budget_exhaustion":
		return "Agent runs out of turns without completing the task. Task may be too large for a single run, or SKILL.md needs to encourage incremental progress."
	default:
		return "Unknown failure pattern — needs manual investigation."
	}
}

// proposeImprovements uses the LLM to generate targeted SKILL.md edits.
func proposeImprovements(patterns []FailurePattern, traces []AgentTrace, workDir, endpoint, model string) []SkillProposal {
	// Group patterns by role
	rolePatterns := make(map[string][]FailurePattern)
	for _, p := range patterns {
		rolePatterns[p.Role] = append(rolePatterns[p.Role], p)
	}

	var allProposals []SkillProposal
	client := &http.Client{Timeout: 3 * time.Minute}

	for role, pats := range rolePatterns {
		r, ok := roles[role]
		if !ok {
			continue
		}

		// Read the current SKILL.md
		skillPath := filepath.Join(workDir, r.SkillPath)
		skillData, err := os.ReadFile(skillPath)
		if err != nil {
			log.Printf("warning: cannot read %s: %v", r.SkillPath, err)
			continue
		}

		patternDesc := formatPatterns(pats)

		prompt := fmt.Sprintf(`You are improving an AI agent's instructions (SKILL.md) based on observed failure patterns.

## Current SKILL.md for role "%s":

%s

## Observed Failure Patterns:

%s

## Task

Propose specific, minimal edits to this SKILL.md that would prevent these failure patterns.
Each proposal should be a concrete addition or modification to the instructions.

Rules:
- Be specific — reference exact sections or add new constraints
- Keep proposals minimal — one focused change per proposal
- Prioritize high-frequency, high-severity patterns
- Don't rewrite the whole file — target the root cause

Respond with ONLY a JSON array of proposals:
[
  {
    "section": "<which section of SKILL.md to modify>",
    "action": "add|modify",
    "content": "<the text to add or the modified text>",
    "rationale": "<why this would help>",
    "priority": "high|medium|low",
    "pattern": "<which failure pattern this addresses>"
  }
]`, role, string(skillData), patternDesc)

		reqBody := ChatRequest{
			Model: model,
			Messages: []ChatMessage{
				{Role: "user", Content: prompt},
			},
			MaxTokens: 4096,
		}
		data, _ := json.Marshal(reqBody)
		url := strings.TrimSuffix(endpoint, "/") + "/chat/completions"
		req, _ := http.NewRequest("POST", url, bytes.NewReader(data))
		req.Header.Set("Content-Type", "application/json")

		resp, err := client.Do(req)
		if err != nil {
			log.Printf("warning: LLM iterate failed for %s: %v", role, err)
			continue
		}
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()

		var chatResp ChatResponse
		if json.Unmarshal(body, &chatResp) != nil || len(chatResp.Choices) == 0 {
			continue
		}

		content := chatResp.Choices[0].Message.Content
		content = strings.TrimPrefix(content, "```json")
		content = strings.TrimPrefix(content, "```")
		content = strings.TrimSuffix(content, "```")
		content = strings.TrimSpace(content)

		var proposals []SkillProposal
		if err := json.Unmarshal([]byte(content), &proposals); err != nil {
			log.Printf("warning: could not parse proposals for %s: %v", role, err)
			continue
		}

		for i := range proposals {
			proposals[i].Role = role
		}
		allProposals = append(allProposals, proposals...)
	}

	return allProposals
}

// proposeImprovementsHeuristic generates proposals without an LLM,
// based on pattern categories and root causes.
func proposeImprovementsHeuristic(patterns []FailurePattern) []SkillProposal {
	var proposals []SkillProposal

	for _, p := range patterns {
		if p.Frequency < 2 {
			continue // only propose for recurring issues
		}

		var proposal SkillProposal
		proposal.Role = p.Role
		proposal.Pattern = p.Category
		proposal.Priority = p.Severity
		proposal.Rationale = p.RootCause

		switch p.Category {
		case "build_gate":
			proposal.Section = "Constraints"
			proposal.Action = "add"
			proposal.Content = "- CRITICAL: Always run `go build ./...` before calling git_commit. If the build fails, fix the errors before committing. Never commit code that does not compile."
		case "scope_adherence":
			proposal.Section = "Scope"
			proposal.Action = "add"
			proposal.Content = "- IMPORTANT: Only modify files listed in your Watched Files section. If you need changes to files outside your scope, describe the needed change in your done() summary so the appropriate agent can pick it up."
		case "protocol_compliance":
			proposal.Section = "Working Protocol"
			proposal.Action = "modify"
			proposal.Content = "- Step 0 (before anything else): Read agents/AGENCY.md to understand current project state and find your task. Step 1: Read the source files you plan to modify. Never edit a file you haven't read first."
		case "analysis_paralysis":
			proposal.Section = "Working Protocol"
			proposal.Action = "add"
			proposal.Content = "- After reading 3-5 files, you should have enough context to start making changes. Don't read the entire codebase — focus on the files in your scope and make targeted edits."
		case "repeated_tool_error":
			proposal.Section = "Constraints"
			proposal.Action = "add"
			proposal.Content = "- If a tool call fails twice with the same error, try a different approach. Don't retry the same failing operation more than twice."
		case "budget_exhaustion":
			proposal.Section = "Working Protocol"
			proposal.Action = "add"
			proposal.Content = "- If you've used more than half your turns and haven't started editing, commit whatever partial progress you have and call done() with a summary of what remains."
		default:
			continue
		}

		proposals = append(proposals, proposal)
	}

	return proposals
}

func formatPatterns(patterns []FailurePattern) string {
	var lines []string
	for _, p := range patterns {
		lines = append(lines, fmt.Sprintf("- [%s] %s (frequency: %d): %s",
			p.Severity, p.Category, p.Frequency, p.RootCause))
		for _, ex := range p.Examples {
			lines = append(lines, fmt.Sprintf("  Example: %s", ex))
		}
	}
	return strings.Join(lines, "\n")
}

// applyProposals writes the high-priority proposals into the SKILL.md files.
func applyProposals(workDir string, proposals []SkillProposal) {
	applied := 0
	for _, p := range proposals {
		if p.Priority != "high" {
			continue
		}

		role, ok := roles[p.Role]
		if !ok {
			continue
		}

		skillPath := filepath.Join(workDir, role.SkillPath)
		data, err := os.ReadFile(skillPath)
		if err != nil {
			log.Printf("warning: cannot read %s: %v", role.SkillPath, err)
			continue
		}

		content := string(data)

		switch p.Action {
		case "add":
			// Append to the specified section or to the end
			sectionHeader := "## " + p.Section
			idx := strings.Index(content, sectionHeader)
			if idx >= 0 {
				// Find the end of the section header line
				lineEnd := strings.Index(content[idx:], "\n")
				if lineEnd >= 0 {
					insertAt := idx + lineEnd + 1
					content = content[:insertAt] + "\n" + p.Content + "\n" + content[insertAt:]
				}
			} else {
				content += "\n\n## " + p.Section + "\n\n" + p.Content + "\n"
			}
		case "modify":
			// For modify, append as a note since we can't reliably find the exact text to replace
			content += "\n\n## Updated Guidance (auto-generated from eval)\n\n" + p.Content + "\n"
		}

		if err := os.WriteFile(skillPath, []byte(content), 0644); err != nil {
			log.Printf("warning: cannot write %s: %v", role.SkillPath, err)
			continue
		}
		log.Printf("applied proposal to %s: [%s] %s", role.SkillPath, p.Action, truncate(p.Content, 80))
		applied++
	}

	log.Printf("applied %d high-priority proposals", applied)
}

func printIterateResult(result IterateResult) {
	fmt.Println("\n=== Iterate Report ===")
	fmt.Printf("Traces analyzed: %d | Reports used: %d\n", result.TracesAnalyzed, result.ReportsUsed)
	fmt.Printf("Generated: %s\n\n", result.GeneratedAt.Format(time.RFC3339))

	if len(result.Patterns) > 0 {
		fmt.Println("Failure Patterns:")
		for _, p := range result.Patterns {
			fmt.Printf("  [%s] %-12s %-25s freq=%d\n", p.Severity, p.Role, p.Category, p.Frequency)
			fmt.Printf("         Root cause: %s\n", truncate(p.RootCause, 100))
		}
	}

	if len(result.Proposals) > 0 {
		fmt.Println("\nSKILL.md Proposals:")
		for i, p := range result.Proposals {
			fmt.Printf("  %d. [%s] %s / %s: %s\n", i+1, p.Priority, p.Role, p.Section, p.Action)
			fmt.Printf("     %s\n", truncate(p.Content, 100))
			fmt.Printf("     Rationale: %s\n", truncate(p.Rationale, 100))
		}
	}

	fmt.Println()
}
