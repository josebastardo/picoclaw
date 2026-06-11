package compression

import (
	"fmt"

	"github.com/sipeed/picoclaw/pkg/providers"
)

type CompressedSummary struct {
	Goal              string   `json:"goal"`
	Progress          []string `json:"progress"`
	Decisions         []string `json:"decisions"`
	ResolvedQuestions []string `json:"resolved_questions"`
	PendingQuestions  []string `json:"pending_questions"`
	Files             []string `json:"files"`
	RemainingWork     []string `json:"remaining_work"`
}

type SegmentBounds struct {
	HeadEnd  int
	TorsoEnd int
}

type CompressionResult struct {
	Messages         []providers.Message
	TokensSaved      int
	WasCompressed    bool
	SummaryTokenCost int
	PrunedTokens     int
	SegmentBounds    *SegmentBounds
}

func SummaryMessage(summary *CompressedSummary) providers.Message {
	text := "[CONTEXT COMPRESSION CHECKPOINT]\n\n"
	text += "## Goal\n" + summary.Goal + "\n\n"
	text += "## Progress\n"
	for _, p := range summary.Progress {
		text += "- " + p + "\n"
	}
	text += "\n## Decisions\n"
	for _, d := range summary.Decisions {
		text += "- " + d + "\n"
	}
	text += "\n## Resolved Questions\n"
	for _, q := range summary.ResolvedQuestions {
		text += "- [RESOLVED] " + q + "\n"
	}
	text += "\n## Pending Questions\n"
	for _, q := range summary.PendingQuestions {
		text += "- [PENDING] " + q + "\n"
	}
	text += "\n## Files Modified/Created\n"
	for _, f := range summary.Files {
		text += "- " + f + "\n"
	}
	text += "\n## Remaining Work\n"
	for _, w := range summary.RemainingWork {
		text += "- " + w + "\n"
	}
	return providers.Message{
		Role:    "user",
		Content: text,
	}
}

type Config struct {
	Enabled                 bool    `json:"enabled"`
	AgentThreshold          float64 `json:"agent_threshold"`
	GatewayThreshold        float64 `json:"gateway_threshold"`
	SummaryTokenRatio       float64 `json:"summary_token_ratio"`
	MinSummaryTokens        int     `json:"min_summary_tokens"`
	MaxSummaryTokens        int     `json:"max_summary_tokens"`
	TailTokenBudget         int     `json:"tail_token_budget"`
	ProtectLastN            int     `json:"protect_last_n"`
	SummaryModel            string  `json:"summary_model,omitempty"`
	AntiThrashMinSavings    float64 `json:"anti_thrash_min_savings"`
	AntiThrashMaxConsecutive int    `json:"anti_thrash_max_consecutive"`
	CooldownRounds          int     `json:"cooldown_rounds"`
	ToolOutputMinChars      int     `json:"tool_output_min_chars"`
	MinHistoryLength        int     `json:"min_history_length"`
}

func DefaultConfig() Config {
	return Config{
		Enabled:                  true,
		AgentThreshold:           0.50,
		GatewayThreshold:         0.85,
		SummaryTokenRatio:        0.20,
		MinSummaryTokens:         500,
		MaxSummaryTokens:         4000,
		TailTokenBudget:          20000,
		ProtectLastN:            10,
		SummaryModel:            "",
		AntiThrashMinSavings:    0.10,
		AntiThrashMaxConsecutive: 2,
		CooldownRounds:          3,
		ToolOutputMinChars:      200,
		MinHistoryLength:        4,
	}
}

func (c *Config) Validate() error {
	if c.AgentThreshold <= 0 || c.AgentThreshold > 1 {
		return fmt.Errorf("agent_threshold must be between 0 and 1, got %f", c.AgentThreshold)
	}
	if c.GatewayThreshold <= 0 || c.GatewayThreshold > 1 {
		return fmt.Errorf("gateway_threshold must be between 0 and 1, got %f", c.GatewayThreshold)
	}
	if c.SummaryTokenRatio <= 0 || c.SummaryTokenRatio > 1 {
		return fmt.Errorf("summary_token_ratio must be between 0 and 1, got %f", c.SummaryTokenRatio)
	}
	if c.MinSummaryTokens < 100 {
		return fmt.Errorf("min_summary_tokens must be >= 100, got %d", c.MinSummaryTokens)
	}
	if c.MaxSummaryTokens < c.MinSummaryTokens {
		return fmt.Errorf("max_summary_tokens (%d) < min_summary_tokens (%d)", c.MaxSummaryTokens, c.MinSummaryTokens)
	}
	if c.TailTokenBudget < 1000 {
		return fmt.Errorf("tail_token_budget must be >= 1000, got %d", c.TailTokenBudget)
	}
	return nil
}
