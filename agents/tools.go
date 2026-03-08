package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// ToolDefinition matches the OpenAI function-calling schema that vLLM
// supports via --enable-auto-tool-choice.
type ToolDefinition struct {
	Type     string         `json:"type"`
	Function FunctionSchema `json:"function"`
}

type FunctionSchema struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
}

// ToolCall is a parsed tool invocation from the LLM response.
type ToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

// allTools returns the tool definitions sent to the vLLM chat completions API.
func allTools() []ToolDefinition {
	return []ToolDefinition{
		{
			Type: "function",
			Function: FunctionSchema{
				Name:        "read_file",
				Description: "Read the contents of a file in the repository. Returns the full file content as a string.",
				Parameters:  json.RawMessage(`{"type":"object","properties":{"path":{"type":"string","description":"File path relative to the repository root"}},"required":["path"]}`),
			},
		},
		{
			Type: "function",
			Function: FunctionSchema{
				Name:        "edit_file",
				Description: "Replace an exact string in a file with a new string. The old_string must appear exactly once in the file. Use for targeted edits.",
				Parameters:  json.RawMessage(`{"type":"object","properties":{"path":{"type":"string","description":"File path relative to the repository root"},"old_string":{"type":"string","description":"Exact string to find (must be unique in the file)"},"new_string":{"type":"string","description":"Replacement string"}},"required":["path","old_string","new_string"]}`),
			},
		},
		{
			Type: "function",
			Function: FunctionSchema{
				Name:        "write_file",
				Description: "Create or overwrite a file with the given content.",
				Parameters:  json.RawMessage(`{"type":"object","properties":{"path":{"type":"string","description":"File path relative to the repository root"},"content":{"type":"string","description":"Full file content to write"}},"required":["path","content"]}`),
			},
		},
		{
			Type: "function",
			Function: FunctionSchema{
				Name:        "run_command",
				Description: "Run a shell command in the repository root. Use for go build, go test, go vet, make, etc. Commands are sandboxed to the workspace. Timeout: 120 seconds.",
				Parameters:  json.RawMessage(`{"type":"object","properties":{"command":{"type":"string","description":"Shell command to execute"}},"required":["command"]}`),
			},
		},
		{
			Type: "function",
			Function: FunctionSchema{
				Name:        "list_files",
				Description: "List files in a directory. Returns newline-separated file paths.",
				Parameters:  json.RawMessage(`{"type":"object","properties":{"path":{"type":"string","description":"Directory path relative to the repository root (default: \".\")"},"pattern":{"type":"string","description":"Optional glob pattern to filter files (e.g., \"*.go\")"}},"required":[]}`),
			},
		},
		{
			Type: "function",
			Function: FunctionSchema{
				Name:        "search_files",
				Description: "Search for a pattern across files in the repository using grep. Returns matching lines with file paths.",
				Parameters:  json.RawMessage(`{"type":"object","properties":{"pattern":{"type":"string","description":"Search pattern (regex supported)"},"path":{"type":"string","description":"Directory to search in (default: \".\")"},"glob":{"type":"string","description":"File glob to filter (e.g., \"*.go\")"}},"required":["pattern"]}`),
			},
		},
		{
			Type: "function",
			Function: FunctionSchema{
				Name:        "git_diff",
				Description: "Show the current git diff of uncommitted changes.",
				Parameters:  json.RawMessage(`{"type":"object","properties":{}}`),
			},
		},
		{
			Type: "function",
			Function: FunctionSchema{
				Name:        "git_commit",
				Description: "Stage all changes and create a git commit with the given message.",
				Parameters:  json.RawMessage(`{"type":"object","properties":{"message":{"type":"string","description":"Commit message"}},"required":["message"]}`),
			},
		},
		{
			Type: "function",
			Function: FunctionSchema{
				Name:        "git_push",
				Description: "Push the current branch to origin. Creates the remote branch if it doesn't exist.",
				Parameters:  json.RawMessage(`{"type":"object","properties":{"branch":{"type":"string","description":"Branch name to push (default: current branch)"}}}`),
			},
		},
		{
			Type: "function",
			Function: FunctionSchema{
				Name:        "done",
				Description: "Signal that the agent has finished its task. Call this when all changes are committed and pushed, or when no action is needed.",
				Parameters:  json.RawMessage(`{"type":"object","properties":{"summary":{"type":"string","description":"Brief summary of what was accomplished"}},"required":["summary"]}`),
			},
		},
	}
}

// executeTool runs a tool call and returns the result string.
func executeTool(workDir string, call ToolCall) (string, error) {
	var args map[string]interface{}
	if err := json.Unmarshal([]byte(call.Function.Arguments), &args); err != nil {
		return "", fmt.Errorf("invalid tool arguments: %w", err)
	}

	switch call.Function.Name {
	case "read_file":
		return toolReadFile(workDir, args)
	case "edit_file":
		return toolEditFile(workDir, args)
	case "write_file":
		return toolWriteFile(workDir, args)
	case "run_command":
		return toolRunCommand(workDir, args)
	case "list_files":
		return toolListFiles(workDir, args)
	case "search_files":
		return toolSearchFiles(workDir, args)
	case "git_diff":
		return toolGitDiff(workDir)
	case "git_commit":
		return toolGitCommit(workDir, args)
	case "git_push":
		return toolGitPush(workDir, args)
	case "done":
		summary, _ := args["summary"].(string)
		return fmt.Sprintf("DONE: %s", summary), nil
	default:
		return "", fmt.Errorf("unknown tool: %s", call.Function.Name)
	}
}

func toolReadFile(workDir string, args map[string]interface{}) (string, error) {
	path, _ := args["path"].(string)
	if path == "" {
		return "", fmt.Errorf("path is required")
	}
	fullPath := filepath.Join(workDir, path)
	if !strings.HasPrefix(fullPath, workDir) {
		return "", fmt.Errorf("path escapes workspace")
	}
	data, err := os.ReadFile(fullPath)
	if err != nil {
		return "", err
	}
	if len(data) > 100_000 {
		return string(data[:100_000]) + "\n... (truncated, file too large)", nil
	}
	return string(data), nil
}

func toolEditFile(workDir string, args map[string]interface{}) (string, error) {
	path, _ := args["path"].(string)
	oldStr, _ := args["old_string"].(string)
	newStr, _ := args["new_string"].(string)
	if path == "" || oldStr == "" {
		return "", fmt.Errorf("path and old_string are required")
	}
	fullPath := filepath.Join(workDir, path)
	if !strings.HasPrefix(fullPath, workDir) {
		return "", fmt.Errorf("path escapes workspace")
	}
	data, err := os.ReadFile(fullPath)
	if err != nil {
		return "", err
	}
	content := string(data)
	count := strings.Count(content, oldStr)
	if count == 0 {
		return "", fmt.Errorf("old_string not found in %s", path)
	}
	if count > 1 {
		return "", fmt.Errorf("old_string appears %d times in %s (must be unique)", count, path)
	}
	content = strings.Replace(content, oldStr, newStr, 1)
	if err := os.WriteFile(fullPath, []byte(content), 0644); err != nil {
		return "", err
	}
	return fmt.Sprintf("edited %s", path), nil
}

func toolWriteFile(workDir string, args map[string]interface{}) (string, error) {
	path, _ := args["path"].(string)
	content, _ := args["content"].(string)
	if path == "" {
		return "", fmt.Errorf("path is required")
	}
	fullPath := filepath.Join(workDir, path)
	if !strings.HasPrefix(fullPath, workDir) {
		return "", fmt.Errorf("path escapes workspace")
	}
	dir := filepath.Dir(fullPath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return "", err
	}
	if err := os.WriteFile(fullPath, []byte(content), 0644); err != nil {
		return "", err
	}
	return fmt.Sprintf("wrote %s (%d bytes)", path, len(content)), nil
}

func toolRunCommand(workDir string, args map[string]interface{}) (string, error) {
	command, _ := args["command"].(string)
	if command == "" {
		return "", fmt.Errorf("command is required")
	}
	// Block destructive commands
	lower := strings.ToLower(command)
	for _, blocked := range []string{"rm -rf /", "rm -rf /*", ":(){ :|:&", "mkfs", "dd if="} {
		if strings.Contains(lower, blocked) {
			return "", fmt.Errorf("blocked command: %s", command)
		}
	}

	cmd := exec.Command("bash", "-c", command)
	cmd.Dir = workDir
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err := cmd.Run()
	result := stdout.String()
	if stderr.Len() > 0 {
		result += "\nSTDERR:\n" + stderr.String()
	}
	if err != nil {
		result += "\nEXIT: " + err.Error()
	}
	// Truncate long output
	if len(result) > 50_000 {
		result = result[:50_000] + "\n... (truncated)"
	}
	return result, nil
}

func toolListFiles(workDir string, args map[string]interface{}) (string, error) {
	path, _ := args["path"].(string)
	if path == "" {
		path = "."
	}
	pattern, _ := args["pattern"].(string)
	fullPath := filepath.Join(workDir, path)

	var files []string
	err := filepath.Walk(fullPath, func(p string, info os.FileInfo, err error) error {
		if err != nil {
			return nil
		}
		if info.IsDir() {
			if info.Name() == ".git" || info.Name() == "vendor" {
				return filepath.SkipDir
			}
			return nil
		}
		rel, _ := filepath.Rel(workDir, p)
		if pattern != "" {
			matched, _ := filepath.Match(pattern, info.Name())
			if !matched {
				return nil
			}
		}
		files = append(files, rel)
		return nil
	})
	if err != nil {
		return "", err
	}
	return strings.Join(files, "\n"), nil
}

func toolSearchFiles(workDir string, args map[string]interface{}) (string, error) {
	pattern, _ := args["pattern"].(string)
	if pattern == "" {
		return "", fmt.Errorf("pattern is required")
	}
	path, _ := args["path"].(string)
	if path == "" {
		path = "."
	}
	glob, _ := args["glob"].(string)

	cmdArgs := []string{"-rn", "--max-count=50"}
	if glob != "" {
		cmdArgs = append(cmdArgs, "--include="+glob)
	}
	cmdArgs = append(cmdArgs, pattern, path)

	cmd := exec.Command("grep", cmdArgs...)
	cmd.Dir = workDir
	var out bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &out
	_ = cmd.Run() // grep returns exit 1 on no match
	result := out.String()
	if len(result) > 30_000 {
		result = result[:30_000] + "\n... (truncated)"
	}
	if result == "" {
		return "no matches found", nil
	}
	return result, nil
}

func toolGitDiff(workDir string) (string, error) {
	cmd := exec.Command("git", "diff")
	cmd.Dir = workDir
	out, err := cmd.CombinedOutput()
	if err != nil {
		return string(out), err
	}
	result := string(out)
	if result == "" {
		return "no uncommitted changes", nil
	}
	if len(result) > 50_000 {
		result = result[:50_000] + "\n... (truncated)"
	}
	return result, nil
}

func toolGitCommit(workDir string, args map[string]interface{}) (string, error) {
	message, _ := args["message"].(string)
	if message == "" {
		return "", fmt.Errorf("message is required")
	}
	addCmd := exec.Command("git", "add", "-A")
	addCmd.Dir = workDir
	if out, err := addCmd.CombinedOutput(); err != nil {
		return string(out), err
	}
	commitCmd := exec.Command("git", "commit", "-m", message)
	commitCmd.Dir = workDir
	out, err := commitCmd.CombinedOutput()
	if err != nil {
		return string(out), err
	}
	return string(out), nil
}

func toolGitPush(workDir string, args map[string]interface{}) (string, error) {
	branch, _ := args["branch"].(string)
	if branch == "" {
		// Get current branch
		cmd := exec.Command("git", "rev-parse", "--abbrev-ref", "HEAD")
		cmd.Dir = workDir
		out, err := cmd.Output()
		if err != nil {
			return "", fmt.Errorf("failed to get current branch: %w", err)
		}
		branch = strings.TrimSpace(string(out))
	}
	cmd := exec.Command("git", "push", "-u", "origin", branch)
	cmd.Dir = workDir
	out, err := cmd.CombinedOutput()
	if err != nil {
		return string(out), err
	}
	return string(out), nil
}
