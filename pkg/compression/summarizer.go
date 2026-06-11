package compression

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/sipeed/picoclaw/pkg/providers"
)

type LLMSummarizer interface {
	Summarize(ctx context.Context, prompt string, modelName string) (string, error)
}

type Summarizer struct {
	config  *Config
	counter *TokenCounter
	llm     LLMSummarizer
}

func NewSummarizer(config *Config, counter *TokenCounter, llm LLMSummarizer) *Summarizer {
	return &Summarizer{config: config, counter: counter, llm: llm}
}

func (s *Summarizer) Summarize(ctx context.Context, torso []providers.Message, previousSummary *CompressedSummary, modelName string) (*CompressedSummary, int, error) {
	torsoText := s.formatTorso(torso)

	contentTokens := s.counter.EstimateMessageTokens(providers.Message{Content: torsoText})
	summaryBudget := int(float64(contentTokens) * s.config.SummaryTokenRatio)
	if summaryBudget < s.config.MinSummaryTokens {
		summaryBudget = s.config.MinSummaryTokens
	}
	if summaryBudget > s.config.MaxSummaryTokens {
		summaryBudget = s.config.MaxSummaryTokens
	}

	prompt := s.buildPrompt(torsoText, previousSummary, summaryBudget)

	summaryModel := s.config.SummaryModel
	if summaryModel == "" {
		summaryModel = modelName
	}

	summaryText, err := s.llm.Summarize(ctx, prompt, summaryModel)
	if err != nil {
		return nil, 0, fmt.Errorf("summarization failed: %w", err)
	}

	summary := s.parseSummary(summaryText)
	if summary == nil {
		return nil, 0, fmt.Errorf("failed to parse summary from LLM response")
	}

	summaryTokens := s.counter.EstimateMessageTokens(SummaryMessage(summary))
	return summary, summaryTokens, nil
}

func (s *Summarizer) formatTorso(messages []providers.Message) string {
	var b strings.Builder
	for _, msg := range messages {
		role := msg.Role
		if role == "" {
			role = "unknown"
		}

		if role == "tool" {
			content := msg.Content
			if len(content) > 500 {
				content = content[:500] + "... [truncated]"
			}
			b.WriteString(fmt.Sprintf("[%s](call_id:%s):\n%s\n", role, msg.ToolCallID, content))
			continue
		}

		b.WriteString(fmt.Sprintf("[%s]:\n", role))
		if msg.Content != "" {
			b.WriteString(msg.Content)
			b.WriteString("\n")
		}
		if msg.ReasoningContent != "" {
			b.WriteString("[thinking]: " + msg.ReasoningContent + "\n")
		}
		for _, tc := range msg.ToolCalls {
			args := tc.Function.Arguments
			if len(args) > 300 {
				args = args[:300] + "..."
			}
			b.WriteString(fmt.Sprintf("  Tool Call: %s(%s)\n", tc.Function.Name, args))
		}
		b.WriteString("\n")
	}
	return b.String()
}

func (s *Summarizer) buildPrompt(torsoText string, previousSummary *CompressedSummary, budget int) string {
	var b strings.Builder
	b.WriteString("You are a summarization agent. Produce a structured summary of the conversation turns below. ")
	b.WriteString("Output ONLY the summary, no greeting, no commentary.\n\n")
	b.WriteString(fmt.Sprintf("Target summary length: approximately %d tokens.\n\n", budget))

	if previousSummary != nil {
		b.WriteString("PREVIOUS SUMMARY (preserve existing information and add new progress):\n")
		data, _ := json.MarshalIndent(previousSummary, "", "  ")
		b.WriteString(string(data))
		b.WriteString("\n\nINSTRUCTIONS: PRESERVE existing information. ADD new progress from the new turns. ")
		b.WriteString("Move completed items from RemainingWork to Progress.\n\n")
	}

	b.WriteString("CONVERSATION TURNS:\n")
	b.WriteString(torsoText)
	b.WriteString("\n\n")
	b.WriteString("STRUCTURED SUMMARY (JSON format ONLY):\n")
	b.WriteString(`{
  "goal": "What was being accomplished",
  "progress": ["What has been done - include file paths, decisions, results"],
  "decisions": ["Architectural and implementation decisions made"],
  "resolved_questions": ["Questions that have been answered"],
  "pending_questions": ["Questions still open"],
  "files": ["Relevant files modified or created"],
  "remaining_work": ["What still needs to be done"]
}`)

	return b.String()
}

func (s *Summarizer) parseSummary(text string) *CompressedSummary {
	text = strings.TrimSpace(text)

	jsonStart := strings.Index(text, "{")
	jsonEnd := strings.LastIndex(text, "}")
	if jsonStart >= 0 && jsonEnd > jsonStart {
		jsonStr := text[jsonStart : jsonEnd+1]
		var summary CompressedSummary
		if err := json.Unmarshal([]byte(jsonStr), &summary); err == nil {
			return &summary
		}
	}

	summary := &CompressedSummary{}
	lines := strings.Split(text, "\n")
	var currentSection string
	sections := map[string]*[]string{
		"Progress":           &summary.Progress,
		"Decisions":          &summary.Decisions,
		"Resolved Questions": &summary.ResolvedQuestions,
		"Pending Questions":  &summary.PendingQuestions,
		"Files":              &summary.Files,
		"Remaining Work":     &summary.RemainingWork,
	}

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		for sectionName := range sections {
			if strings.HasPrefix(trimmed, sectionName+":") || strings.HasPrefix(trimmed, "## "+sectionName) {
				currentSection = sectionName
				rest := strings.TrimPrefix(trimmed, sectionName+":")
				rest = strings.TrimPrefix(rest, "## ")
				rest = strings.TrimSpace(rest)
				if rest != "" {
					list := sections[currentSection]
					*list = append(*list, rest)
				}
				goto nextLine
			}
		}
		if strings.HasPrefix(trimmed, "Goal:") || strings.HasPrefix(trimmed, "## Goal") {
			currentSection = "Goal"
			rest := strings.TrimPrefix(trimmed, "Goal:")
			rest = strings.TrimPrefix(rest, "## Goal")
			rest = strings.TrimSpace(rest)
			if rest != "" {
				summary.Goal = rest
			}
			goto nextLine
		}
		if currentSection == "Goal" && trimmed != "" {
			if summary.Goal != "" {
				summary.Goal += " " + trimmed
			} else {
				summary.Goal = trimmed
			}
			goto nextLine
		}
		if currentSection != "" && trimmed != "" {
			item := strings.TrimPrefix(trimmed, "- ")
			item = strings.TrimPrefix(item, "* ")
			item = strings.TrimSpace(item)
			if item != "" {
				if list, ok := sections[currentSection]; ok {
					*list = append(*list, item)
				}
			}
		}
	nextLine:
	}

	if summary.Goal == "" && len(summary.Progress) == 0 && len(summary.Decisions) == 0 {
		return nil
	}
	return summary
}
