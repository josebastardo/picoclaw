package compression

import (
	"github.com/sipeed/picoclaw/pkg/providers"
)

type Pruner struct {
	config  *Config
	counter *TokenCounter
}

func NewPruner(config *Config, counter *TokenCounter) *Pruner {
	return &Pruner{config: config, counter: counter}
}

func (p *Pruner) PruneToolResults(messages []providers.Message, protectFrom int) int {
	saved := 0
	placeholder := "[Old output cleared]"

	for i := 0; i < protectFrom && i < len(messages); i++ {
		msg := &messages[i]
		if msg.Role != "tool" {
			continue
		}
		if len(msg.Content) > p.config.ToolOutputMinChars {
			oldTokens := p.counter.EstimateMessageTokens(*msg)
			msg.Content = placeholder
			newTokens := p.counter.EstimateMessageTokens(*msg)
			saved += oldTokens - newTokens
		}
	}

	return saved
}
