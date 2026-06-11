package compression

import (
	"sync"
	"strings"

	"github.com/pkoukk/tiktoken-go"
	"github.com/sipeed/picoclaw/pkg/providers"
)

type TokenCounter struct {
	mu              sync.RWMutex
	cache           map[string]*tiktoken.Tiktoken
	defaultEncoding string
}

func NewTokenCounter() *TokenCounter {
	return &TokenCounter{
		cache:           make(map[string]*tiktoken.Tiktoken),
		defaultEncoding: "cl100k_base",
	}
}

func (tc *TokenCounter) CountTokens(text string, modelName string) (int, error) {
	if text == "" {
		return 0, nil
	}
	tke, err := tc.getEncoding(modelName)
	if err != nil {
		return 0, err
	}
	return len(tke.Encode(text, nil, nil)), nil
}

func (tc *TokenCounter) CountMessageTokens(msg providers.Message, modelName string) (int, error) {
	total := 0

	if msg.Content != "" {
		tokens, err := tc.CountTokens(msg.Content, modelName)
		if err != nil {
			return 0, err
		}
		total += tokens
	}

	if msg.ReasoningContent != "" {
		tokens, err := tc.CountTokens(msg.ReasoningContent, modelName)
		if err != nil {
			return 0, err
		}
		total += tokens
	}

	for _, toolCall := range msg.ToolCalls {
		if toolCall.ID != "" {
			t, _ := tc.CountTokens(toolCall.ID, modelName)
			total += t
		}
		if toolCall.Function != nil {
			t, _ := tc.CountTokens(toolCall.Function.Name, modelName)
			total += t
			t, _ = tc.CountTokens(toolCall.Function.Arguments, modelName)
			total += t
		}
	}

	total += 4

	const mediaTokensPerItem = 256
	total += len(msg.Media) * mediaTokensPerItem

	return total, nil
}

func (tc *TokenCounter) CountMessages(messages []providers.Message, modelName string) (int, error) {
	total := 0
	for _, msg := range messages {
		t, err := tc.CountMessageTokens(msg, modelName)
		if err != nil {
			return 0, err
		}
		total += t
	}
	return total, nil
}

func (tc *TokenCounter) EstimateMessageTokens(msg providers.Message) int {
	chars := len(msg.Content)
	chars += len(msg.ReasoningContent)

	for _, toolCall := range msg.ToolCalls {
		chars += len(toolCall.ID) + len(toolCall.Type)
		if toolCall.Function != nil {
			chars += len(toolCall.Function.Name) + len(toolCall.Function.Arguments)
		} else {
			chars += len(toolCall.Name)
		}
	}

	if msg.ToolCallID != "" {
		chars += len(msg.ToolCallID)
	}

	const messageOverhead = 12
	chars += messageOverhead

	tokens := chars * 2 / 5
	if tokens < 1 {
		tokens = 1
	}

	const mediaTokensPerItem = 256
	tokens += len(msg.Media) * mediaTokensPerItem

	return tokens
}

func (tc *TokenCounter) EstimateMessages(messages []providers.Message) int {
	total := 0
	for _, msg := range messages {
		total += tc.EstimateMessageTokens(msg)
	}
	return total
}

func (tc *TokenCounter) getEncoding(modelName string) (*tiktoken.Tiktoken, error) {
	tc.mu.RLock()
	if tke, ok := tc.cache[modelName]; ok {
		tc.mu.RUnlock()
		return tke, nil
	}
	tc.mu.RUnlock()

	encoding := modelToEncoding(modelName)
	tke, err := tiktoken.GetEncoding(encoding)
	if err != nil {
		return nil, err
	}

	tc.mu.Lock()
	tc.cache[modelName] = tke
	tc.mu.Unlock()
	return tke, nil
}

func modelToEncoding(modelName string) string {
	name := strings.ToLower(modelName)
	if strings.Contains(name, "gpt-4o") || strings.Contains(name, "gpt-5") || strings.Contains(name, "o1") || strings.Contains(name, "o3") {
		return "o200k_base"
	}
	if strings.Contains(name, "gpt-4") || strings.Contains(name, "gpt-3") {
		return "cl100k_base"
	}
	if strings.Contains(name, "claude") {
		return "cl100k_base"
	}
	if strings.Contains(name, "gemini") {
		return "cl100k_base"
	}
	if strings.Contains(name, "llama") || strings.Contains(name, "mistral") || strings.Contains(name, "deepseek") {
		return "cl100k_base"
	}
	return "cl100k_base"
}
