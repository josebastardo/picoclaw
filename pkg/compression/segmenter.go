package compression

import (
	"github.com/sipeed/picoclaw/pkg/providers"
)

type Segmenter struct {
	config  *Config
	counter *TokenCounter
}

func NewSegmenter(config *Config, counter *TokenCounter) *Segmenter {
	return &Segmenter{config: config, counter: counter}
}

func (s *Segmenter) Segment(messages []providers.Message) *SegmentBounds {
	if len(messages) <= s.config.MinHistoryLength {
		return nil
	}

	headEnd := 0
	foundUser := false
	for i, msg := range messages {
		if msg.Role == "user" && i > 0 {
			headEnd = i
			foundUser = true
			break
		}
	}

	if !foundUser {
		headEnd = 1
	}

	j := headEnd + 1
	for j < len(messages) && messages[j].Role == "assistant" {
		j++
	}
	headEnd = j

	tailTokens := 0
	tailStart := len(messages)

	lastUserIdx := -1
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "user" {
			lastUserIdx = i
			break
		}
	}

	for i := len(messages) - 1; i >= headEnd; i-- {
		tokens := s.counter.EstimateMessageTokens(messages[i])
		if tailTokens+tokens > s.config.TailTokenBudget && tailStart < len(messages) {
			break
		}
		tailTokens += tokens
		tailStart = i
	}

	if lastUserIdx >= 0 && tailStart > lastUserIdx {
		tailStart = lastUserIdx
	}

	tailStart = s.alignBackward(messages, tailStart)

	if tailStart <= headEnd {
		return nil
	}

	return &SegmentBounds{
		HeadEnd:  headEnd,
		TorsoEnd: tailStart,
	}
}

func (s *Segmenter) alignBackward(messages []providers.Message, start int) int {
	if start <= 0 || start >= len(messages) {
		return start
	}
	if messages[start].Role == "tool" && start > 0 {
		prev := messages[start-1]
		if prev.Role == "assistant" && len(prev.ToolCalls) > 0 {
			return start - 1
		}
	}
	return start
}
