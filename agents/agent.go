package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// ChatMessage is a message in the OpenAI chat completions format.
type ChatMessage struct {
	Role       string     `json:"role"`
	Content    string     `json:"content,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

// ChatRequest is the request body for /v1/chat/completions.
type ChatRequest struct {
	Model    string           `json:"model"`
	Messages []ChatMessage    `json:"messages"`
	Tools    []ToolDefinition `json:"tools,omitempty"`
	MaxTokens int             `json:"max_tokens,omitempty"`
}

// ChatResponse is the response from /v1/chat/completions.
type ChatResponse struct {
	Choices []struct {
		Message      ChatMessage `json:"message"`
		FinishReason string      `json:"finish_reason"`
	} `json:"choices"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

func main() {
	mode := envOrDefault("AGENT_MODE", "run")

	switch mode {
	case "run":
		runAgent()
	case "eval":
		runEval()
	case "iterate":
		runIterate()
	default:
		log.Fatalf("unknown AGENT_MODE %q (valid: run, eval, iterate)", mode)
	}
}

func runAgent() {
	cfg, err := loadConfig()
	if err != nil {
		log.Fatalf("config error: %v", err)
	}

	log.Printf("agent starting: role=%s repo=%s endpoint=%s", cfg.Role.Name, cfg.RepoURL, cfg.VLLMEndpoint)

	if err := gitClone(cfg); err != nil {
		log.Fatalf("git clone failed: %v", err)
	}

	agencyContent, err := readAgency(cfg.WorkDir)
	if err != nil {
		log.Printf("warning: could not read AGENCY.md: %v", err)
		agencyContent = "(no AGENCY.md found)"
	}

	skillContent, err := readSkill(cfg)
	if err != nil {
		log.Fatalf("could not read skill: %v", err)
	}

	systemPrompt := buildSystemPrompt(cfg, skillContent, agencyContent)

	runID := fmt.Sprintf("%d", time.Now().UnixNano())
	branchName := fmt.Sprintf("agent/%s/%s", cfg.Role.Name, runID)
	if err := gitCheckoutBranch(cfg.WorkDir, branchName); err != nil {
		log.Fatalf("git checkout failed: %v", err)
	}

	collector := newTraceCollector(runID, cfg.Role.Name, branchName, cfg.VLLMModel, cfg.MaxTurns, cfg.DryRun)

	if err := agentLoop(cfg, systemPrompt, branchName, collector); err != nil {
		log.Fatalf("agent loop failed: %v", err)
	}

	if err := collector.writeTrace(cfg.WorkDir); err != nil {
		log.Printf("warning: failed to write trace: %v", err)
	} else {
		log.Printf("trace written: agents/traces/%s_%s.json", cfg.Role.Name, runID)
	}

	log.Printf("agent finished: role=%s", cfg.Role.Name)
}

func gitClone(cfg *Config) error {
	if _, err := os.Stat(filepath.Join(cfg.WorkDir, ".git")); err == nil {
		log.Println("workspace already has a git repo, pulling latest")
		cmd := exec.Command("git", "pull", "--ff-only")
		cmd.Dir = cfg.WorkDir
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		return cmd.Run()
	}
	log.Printf("cloning %s into %s", cfg.RepoURL, cfg.WorkDir)
	cmd := exec.Command("git", "clone", "--depth=1", "--branch", cfg.RepoBranch, cfg.RepoURL, cfg.WorkDir)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func gitCheckoutBranch(workDir, branch string) error {
	cmd := exec.Command("git", "checkout", "-b", branch)
	cmd.Dir = workDir
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("checkout branch %s: %s: %w", branch, string(out), err)
	}
	log.Printf("on branch %s", branch)
	return nil
}

func readAgency(workDir string) (string, error) {
	data, err := os.ReadFile(filepath.Join(workDir, "agents", "AGENCY.md"))
	if err != nil {
		return "", err
	}
	return string(data), nil
}

func readSkill(cfg *Config) (string, error) {
	data, err := os.ReadFile(filepath.Join(cfg.WorkDir, cfg.Role.SkillPath))
	if err != nil {
		return "", fmt.Errorf("reading %s: %w", cfg.Role.SkillPath, err)
	}
	return string(data), nil
}

func buildSystemPrompt(cfg *Config, skill, agency string) string {
	var b strings.Builder
	b.WriteString("You are an AI agent working on the trainjob-operator Kubernetes project.\n\n")
	b.WriteString(fmt.Sprintf("## Your Role: %s\n\n", cfg.Role.Name))
	b.WriteString(fmt.Sprintf("%s\n\n", cfg.Role.Description))
	b.WriteString("## Domain Instructions (from SKILL.md)\n\n")
	b.WriteString(skill)
	b.WriteString("\n\n## Project Coordination (from AGENCY.md)\n\n")
	b.WriteString(agency)
	b.WriteString("\n\n## Watched Files\n\n")
	b.WriteString("You are responsible for these files/directories:\n")
	for _, p := range cfg.Role.WatchPaths {
		b.WriteString(fmt.Sprintf("- %s\n", p))
	}
	b.WriteString("\n## Working Protocol\n\n")
	b.WriteString("1. Read AGENCY.md to find tasks matching your role\n")
	b.WriteString("2. Examine the relevant source files using read_file\n")
	b.WriteString("3. Make focused, well-tested changes using edit_file or write_file\n")
	b.WriteString("4. Run 'go build ./...' and 'go vet ./...' to verify your changes compile\n")
	b.WriteString("5. Commit your changes with git_commit\n")
	b.WriteString("6. Call done() with a summary of what you did\n\n")
	b.WriteString("IMPORTANT:\n")
	b.WriteString("- Never push to main. You are on a feature branch.\n")
	b.WriteString("- Make small, focused changes. Don't rewrite entire files.\n")
	b.WriteString("- If a task is claimed by another agent, skip it.\n")
	b.WriteString("- If you're blocked by a dependency, note it and call done().\n")
	if cfg.DryRun {
		b.WriteString("- DRY RUN MODE: Do not push changes. Only show what you would do.\n")
	}
	return b.String()
}

func agentLoop(cfg *Config, systemPrompt, branch string, collector *TraceCollector) error {
	messages := []ChatMessage{
		{Role: "system", Content: systemPrompt},
		{Role: "user", Content: fmt.Sprintf(
			"You are the %s agent. Review AGENCY.md and the source files in your domain. "+
				"Find an unclaimed task that matches your role, or identify an improvement you can make. "+
				"Make the change, verify it compiles, commit, and call done().",
			cfg.Role.Name,
		)},
	}

	tools := allTools()
	client := &http.Client{Timeout: 5 * time.Minute}

	for turn := 0; turn < cfg.MaxTurns; turn++ {
		log.Printf("turn %d/%d", turn+1, cfg.MaxTurns)

		resp, err := chatCompletion(client, cfg, messages, tools)
		if err != nil {
			collector.finish("error", err.Error(), turn)
			return fmt.Errorf("chat completion failed at turn %d: %w", turn, err)
		}

		if len(resp.Choices) == 0 {
			collector.finish("error", "empty response", turn)
			return fmt.Errorf("empty response from LLM at turn %d", turn)
		}

		choice := resp.Choices[0]
		messages = append(messages, choice.Message)

		if len(choice.Message.ToolCalls) == 0 {
			log.Printf("LLM response (no tool calls): %s", truncate(choice.Message.Content, 200))
			if choice.FinishReason == "stop" {
				log.Println("LLM finished (stop)")
				collector.finish("stop", choice.Message.Content, turn+1)
				return nil
			}
			continue
		}

		for _, tc := range choice.Message.ToolCalls {
			log.Printf("tool call: %s(%s)", tc.Function.Name, truncate(tc.Function.Arguments, 100))

			if tc.Function.Name == "done" {
				var args map[string]interface{}
				_ = json.Unmarshal([]byte(tc.Function.Arguments), &args)
				summary, _ := args["summary"].(string)
				log.Printf("DONE: %s", summary)

				collectBuildStatus(cfg.WorkDir, collector)
				collector.finish("done", summary, turn+1)

				if !cfg.DryRun {
					if pushResult, err := toolGitPush(cfg.WorkDir, map[string]interface{}{"branch": branch}); err != nil {
						log.Printf("git push failed (non-fatal): %v: %s", err, pushResult)
					} else {
						log.Printf("pushed to %s", branch)
					}
				}
				return nil
			}

			callStart := time.Now()
			result, toolErr := executeTool(cfg.WorkDir, tc)
			callDuration := time.Since(callStart)

			collector.recordToolCall(turn, tc, result, toolErr, callDuration)

			var content string
			if toolErr != nil {
				content = fmt.Sprintf("ERROR: %v", toolErr)
				log.Printf("tool error: %s: %v", tc.Function.Name, toolErr)
			} else {
				content = result
				log.Printf("tool result: %s -> %s", tc.Function.Name, truncate(result, 100))
			}

			messages = append(messages, ChatMessage{
				Role:       "tool",
				Content:    content,
				ToolCallID: tc.ID,
			})
		}
	}

	collectBuildStatus(cfg.WorkDir, collector)
	collector.finish("max_turns", fmt.Sprintf("reached %d turns", cfg.MaxTurns), cfg.MaxTurns)
	log.Printf("max turns (%d) reached", cfg.MaxTurns)
	return nil
}

// collectBuildStatus runs go build and go vet, recording pass/fail in the trace.
func collectBuildStatus(workDir string, collector *TraceCollector) {
	buildCmd := exec.Command("go", "build", "./...")
	buildCmd.Dir = workDir
	buildPassed := buildCmd.Run() == nil
	collector.setBuild(buildPassed)

	vetCmd := exec.Command("go", "vet", "./...")
	vetCmd.Dir = workDir
	vetPassed := vetCmd.Run() == nil
	collector.setVet(vetPassed)

	diffCmd := exec.Command("git", "diff", "--stat", "HEAD~1")
	diffCmd.Dir = workDir
	if out, err := diffCmd.Output(); err == nil {
		collector.setDiffStat(strings.TrimSpace(string(out)))
	}
}

func chatCompletion(client *http.Client, cfg *Config, messages []ChatMessage, tools []ToolDefinition) (*ChatResponse, error) {
	reqBody := ChatRequest{
		Model:     cfg.VLLMModel,
		Messages:  messages,
		Tools:     tools,
		MaxTokens: 4096,
	}

	data, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}

	endpoint := strings.TrimSuffix(cfg.VLLMEndpoint, "/") + "/chat/completions"
	req, err := http.NewRequest("POST", endpoint, bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("HTTP request to %s failed: %w", endpoint, err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("vLLM returned %d: %s", resp.StatusCode, truncate(string(body), 500))
	}

	var chatResp ChatResponse
	if err := json.Unmarshal(body, &chatResp); err != nil {
		return nil, fmt.Errorf("parsing response: %w", err)
	}

	if chatResp.Error != nil {
		return nil, fmt.Errorf("vLLM error: %s", chatResp.Error.Message)
	}

	return &chatResp, nil
}

// parseAgencyTasks extracts the component table from AGENCY.md.
// Returns tasks as a list of maps with keys: Component, Owner, Status, Branch, Description.
func parseAgencyTasks(content string) []map[string]string {
	var tasks []map[string]string
	scanner := bufio.NewScanner(strings.NewReader(content))
	inTable := false
	var headers []string

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if strings.HasPrefix(line, "| Component") {
			inTable = true
			headers = splitTableRow(line)
			scanner.Scan() // skip separator row
			continue
		}
		if inTable {
			if !strings.HasPrefix(line, "|") {
				break
			}
			cols := splitTableRow(line)
			task := make(map[string]string)
			for i, h := range headers {
				if i < len(cols) {
					task[h] = cols[i]
				}
			}
			tasks = append(tasks, task)
		}
	}
	return tasks
}

func splitTableRow(line string) []string {
	parts := strings.Split(line, "|")
	var result []string
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p != "" {
			result = append(result, p)
		}
	}
	return result
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
