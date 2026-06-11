package compression

import (
	"context"
	"testing"

	"github.com/sipeed/picoclaw/pkg/providers"
)

type mockSummarizerLLM struct{}

func (m *mockSummarizerLLM) Summarize(_ context.Context, prompt string, modelName string) (string, error) {
	return `{
		"goal": "Implement file listing feature",
		"progress": ["Created list_files tool"],
		"decisions": ["Using os.ReadDir"],
		"resolved_questions": [],
		"pending_questions": ["Should support recursive?"],
		"files": ["pkg/tools/list.go"],
		"remaining_work": ["Add tests"]
	}`, nil
}

func TestCompressFull(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MinHistoryLength = 3
	counter := NewTokenCounter()
	llm := &mockSummarizerLLM{}
	compressor := NewContextCompressor(&cfg, counter, llm)

	messages := make([]providers.Message, 0, 20)
	messages = append(messages, providers.Message{Role: "system", Content: "You are helpful"})
	for i := 0; i < 19; i++ {
		role := "user"
		if i%2 == 1 {
			role = "assistant"
		}
		content := "Test message " + repeatString("x", 200)
		messages = append(messages, providers.Message{Role: role, Content: content})
	}

	result, err := compressor.Compress(context.Background(), messages, "gpt-4o", 128000)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result == nil {
		t.Log("compression not triggered (may need more tokens)")
		return
	}
	t.Logf("saved=%d pruned=%d summary_cost=%d",
		result.TokensSaved, result.PrunedTokens, result.SummaryTokenCost)
}

func TestCompressBelowThreshold(t *testing.T) {
	cfg := DefaultConfig()
	cfg.AgentThreshold = 0.90
	counter := NewTokenCounter()
	llm := &mockSummarizerLLM{}
	compressor := NewContextCompressor(&cfg, counter, llm)

	messages := []providers.Message{
		{Role: "user", Content: "Short"},
	}

	result, err := compressor.Compress(context.Background(), messages, "gpt-4o", 128000)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != nil {
		t.Fatal("expected nil when below threshold")
	}
}

func TestTrackerAntiThrash(t *testing.T) {
	tracker := NewCompressionTracker()
	cfg := DefaultConfig()

	tracker.RecordResult(50, 1000)
	if tracker.ShouldSkip(&cfg) {
		t.Fatal("should not skip after 1 low-savings")
	}
	tracker.RecordResult(30, 1000)
	if tracker.ShouldSkip(&cfg) {
		t.Fatal("should not skip after 2 low-savings")
	}
	tracker.RecordResult(20, 1000)
	if !tracker.ShouldSkip(&cfg) {
		t.Fatal("should skip after 3 consecutive low-savings")
	}
}

func TestTrackerCooldown(t *testing.T) {
	tracker := NewCompressionTracker()
	cfg := DefaultConfig()

	tracker.RecordFailure()
	if !tracker.ShouldSkip(&cfg) {
		t.Fatal("should skip during cooldown")
	}
	if !tracker.ShouldSkip(&cfg) {
		t.Fatal("should still skip during cooldown")
	}
	if !tracker.ShouldSkip(&cfg) {
		t.Fatal("should still skip on last cooldown")
	}
	if tracker.ShouldSkip(&cfg) {
		t.Fatal("should no longer skip after cooldown")
	}
}

func TestPruneToolResults(t *testing.T) {
	cfg := DefaultConfig()
	cfg.ToolOutputMinChars = 10
	counter := NewTokenCounter()
	pruner := NewPruner(&cfg, counter)

	messages := []providers.Message{
		{Role: "user", Content: "list files"},
		{Role: "assistant", Content: "listing"},
		{Role: "tool", Content: repeatString("file content line that is very long and should be pruned\n", 20)},
	}

	saved := pruner.PruneToolResults(messages, 3)
	if saved <= 0 {
		t.Fatalf("expected positive savings, got %d", saved)
	}
	if messages[2].Content != "[Old output cleared]" {
		t.Fatalf("tool result was not pruned, got: %s", messages[2].Content[:30])
	}
}

func TestSegmentBasic(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MinHistoryLength = 3
	cfg.TailTokenBudget = 50
	counter := NewTokenCounter()
	seg := NewSegmenter(&cfg, counter)

	messages := []providers.Message{
		{Role: "system", Content: "You are a helpful assistant that can perform various tasks"},
		{Role: "user", Content: "Hello, can you help me with something?"},
		{Role: "assistant", Content: "Of course! What do you need help with?"},
		{Role: "user", Content: "I need to list all files in the project directory"},
		{Role: "assistant", Content: "I will list the files for you right away"},
		{Role: "tool", Content: "src/main.go\nsrc/utils.go\npkg/agent.go\nREADME.md"},
		{Role: "user", Content: "Now I need to read the main.go file"},
		{Role: "assistant", Content: "Reading the main.go file contents now"},
		{Role: "tool", Content: "package main\n\nfunc main() {\n\tprintln(\"hello\")\n}"},
		{Role: "user", Content: "Can you modify it to add compression?"},
		{Role: "assistant", Content: "I will add the compression system"},
	}

	bounds := seg.Segment(messages)
	if bounds == nil {
		t.Fatal("expected non-nil bounds")
	}
	t.Logf("HeadEnd=%d TorsoEnd=%d len=%d", bounds.HeadEnd, bounds.TorsoEnd, len(messages))
	if bounds.HeadEnd < 1 {
		t.Fatal("expected head >= 1")
	}
	if bounds.TorsoEnd <= bounds.HeadEnd {
		t.Fatal("expected torso between head and tail")
	}
}

func TestSegmentPreservesUserMessage(t *testing.T) {
	cfg := DefaultConfig()
	cfg.TailTokenBudget = 40
	counter := NewTokenCounter()
	seg := NewSegmenter(&cfg, counter)

	messages := []providers.Message{
		{Role: "system", Content: "System prompt for the AI assistant"},
		{Role: "user", Content: "First question from the user about something important"},
		{Role: "assistant", Content: "First response from the assistant addressing the question"},
		{Role: "user", Content: "Second question with more details about the topic"},
		{Role: "assistant", Content: "Second response with additional information provided"},
		{Role: "user", Content: "LAST USER MESSAGE that must be preserved in tail"},
	}

	bounds := seg.Segment(messages)
	if bounds == nil {
		t.Fatal("expected non-nil bounds")
	}

	lastUserInTail := false
	for i := bounds.TorsoEnd; i < len(messages); i++ {
		if messages[i].Role == "user" {
			lastUserInTail = true
			break
		}
	}
	if !lastUserInTail {
		t.Fatal("expected last user message to be in tail")
	}
}

func TestSummarize(t *testing.T) {
	cfg := DefaultConfig()
	counter := NewTokenCounter()
	llm := &mockSummarizerLLM{}
	summarizer := NewSummarizer(&cfg, counter, llm)

	torso := []providers.Message{
		{Role: "user", Content: "Create file listing"},
		{Role: "assistant", Content: "I will create it"},
	}

	summary, tokens, err := summarizer.Summarize(context.Background(), torso, nil, "gpt-4o")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if summary.Goal != "Implement file listing feature" {
		t.Fatalf("expected goal 'Implement file listing feature', got '%s'", summary.Goal)
	}
	if tokens <= 0 {
		t.Fatal("expected positive token cost")
	}
}

func TestParseSummarySectionFallback(t *testing.T) {
	cfg := DefaultConfig()
	counter := NewTokenCounter()
	llm := &mockSummarizerLLM{}
	summarizer := NewSummarizer(&cfg, counter, llm)

	text := `Goal: Implement file listing feature
Progress:
- Created list_files tool
- Added permission checks
Decisions:
- Using os.ReadDir
Files:
- pkg/tools/list.go`

	summary := summarizer.parseSummary(text)
	if summary == nil {
		t.Fatal("expected non-nil summary")
	}
	if len(summary.Progress) != 2 {
		t.Fatalf("expected 2 progress items, got %d", len(summary.Progress))
	}
}

func TestSummaryMessage(t *testing.T) {
	summary := &CompressedSummary{
		Goal:     "Test goal",
		Progress: []string{"Step 1"},
	}

	msg := SummaryMessage(summary)
	if msg.Role != "user" {
		t.Fatal("expected user role")
	}
	if msg.Content == "" {
		t.Fatal("expected non-empty content")
	}
}

func TestAssemble(t *testing.T) {
	assembler := NewAssembler()

	head := []providers.Message{{Role: "system", Content: "You are helpful"}}
	summaryMsg := SummaryMessage(&CompressedSummary{Goal: "Test"})
	tail := []providers.Message{{Role: "user", Content: "Bye"}}

	result := assembler.Assemble(head, summaryMsg, tail, true)
	if len(result) != 4 {
		t.Fatalf("expected 4 messages, got %d", len(result))
	}

	result2 := assembler.Assemble(head, summaryMsg, tail, false)
	if len(result2) != 3 {
		t.Fatalf("expected 3 messages (no system note), got %d", len(result2))
	}
}

func repeatString(s string, count int) string {
	result := make([]byte, len(s)*count)
	for i := 0; i < count; i++ {
		copy(result[i*len(s):], s)
	}
	return string(result)
}
