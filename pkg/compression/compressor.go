package compression

import (
	"context"
	"fmt"

	"github.com/sipeed/picoclaw/pkg/providers"
)

type ContextCompressor struct {
	config     *Config
	counter    *TokenCounter
	pruner     *Pruner
	segmenter  *Segmenter
	summarizer *Summarizer
	assembler  *Assembler
	tracker    *CompressionTracker
}

func NewContextCompressor(config *Config, counter *TokenCounter, summarizerLLM LLMSummarizer) *ContextCompressor {
	return &ContextCompressor{
		config:     config,
		counter:    counter,
		pruner:     NewPruner(config, counter),
		segmenter:  NewSegmenter(config, counter),
		summarizer: NewSummarizer(config, counter, summarizerLLM),
		assembler:  NewAssembler(),
		tracker:    NewCompressionTracker(),
	}
}

func (cc *ContextCompressor) Compress(
	ctx context.Context,
	messages []providers.Message,
	modelName string,
	modelContextSize int,
) (*CompressionResult, error) {
	if !cc.config.Enabled {
		return nil, nil
	}
	if len(messages) < cc.config.MinHistoryLength {
		return nil, nil
	}

	currentTokens := cc.counter.EstimateMessages(messages)
	thresholdTokens := int(float64(modelContextSize) * cc.config.AgentThreshold)
	if currentTokens < thresholdTokens {
		return nil, nil
	}

	if cc.tracker.ShouldSkip(cc.config) {
		return nil, nil
	}

	result := &CompressionResult{WasCompressed: false}

	msgCopy := make([]providers.Message, len(messages))
	copy(msgCopy, messages)

	savedPhase1 := cc.pruner.PruneToolResults(msgCopy, len(msgCopy))
	result.PrunedTokens = savedPhase1

	bounds := cc.segmenter.Segment(msgCopy)
	if bounds == nil {
		if savedPhase1 > 0 {
			result.WasCompressed = true
			result.TokensSaved = savedPhase1
			result.Messages = msgCopy
			return result, nil
		}
		return nil, nil
	}
	result.SegmentBounds = bounds

	head := msgCopy[:bounds.HeadEnd]
	torso := msgCopy[bounds.HeadEnd:bounds.TorsoEnd]
	tail := msgCopy[bounds.TorsoEnd:]

	summary, summaryTokens, err := cc.summarizer.Summarize(ctx, torso, nil, modelName)
	if err != nil {
		cc.tracker.RecordFailure()
		return nil, fmt.Errorf("summarization failed: %w", err)
	}
	result.SummaryTokenCost = summaryTokens

	torsoTokens := 0
	for _, msg := range torso {
		torsoTokens += cc.counter.EstimateMessageTokens(msg)
	}
	totalTokensAfterPrune := cc.counter.EstimateMessages(msgCopy)
	totalTokensAfter := totalTokensAfterPrune - torsoTokens + summaryTokens
	tokensSaved := currentTokens - totalTokensAfter

	savingsRatio := float64(tokensSaved) / float64(currentTokens)
	if savingsRatio < cc.config.AntiThrashMinSavings {
		cc.tracker.RecordResult(0, currentTokens)
		return nil, nil
	}

	summaryMsg := SummaryMessage(summary)
	compressed := cc.assembler.Assemble(head, summaryMsg, tail, cc.tracker.IsFirstCompaction())

	result.Messages = compressed
	result.TokensSaved = tokensSaved
	result.WasCompressed = true

	cc.tracker.RecordResult(tokensSaved, currentTokens)
	return result, nil
}

func (cc *ContextCompressor) CompressWithPrevious(
	ctx context.Context,
	messages []providers.Message,
	modelName string,
	modelContextSize int,
	previousSummary *CompressedSummary,
) (*CompressionResult, error) {
	if !cc.config.Enabled {
		return nil, nil
	}
	if len(messages) < cc.config.MinHistoryLength {
		return nil, nil
	}

	currentTokens := cc.counter.EstimateMessages(messages)
	thresholdTokens := int(float64(modelContextSize) * cc.config.AgentThreshold)
	if currentTokens < thresholdTokens {
		return nil, nil
	}

	if cc.tracker.ShouldSkip(cc.config) {
		return nil, nil
	}

	result := &CompressionResult{WasCompressed: false}

	msgCopy := make([]providers.Message, len(messages))
	copy(msgCopy, messages)

	savedPhase1 := cc.pruner.PruneToolResults(msgCopy, len(msgCopy))
	result.PrunedTokens = savedPhase1

	bounds := cc.segmenter.Segment(msgCopy)
	if bounds == nil {
		if savedPhase1 > 0 {
			result.WasCompressed = true
			result.TokensSaved = savedPhase1
			result.Messages = msgCopy
			return result, nil
		}
		return nil, nil
	}
	result.SegmentBounds = bounds

	head := msgCopy[:bounds.HeadEnd]
	torso := msgCopy[bounds.HeadEnd:bounds.TorsoEnd]
	tail := msgCopy[bounds.TorsoEnd:]

	summary, summaryTokens, err := cc.summarizer.Summarize(ctx, torso, previousSummary, modelName)
	if err != nil {
		cc.tracker.RecordFailure()
		return nil, fmt.Errorf("summarization failed: %w", err)
	}
	result.SummaryTokenCost = summaryTokens

	torsoTokens := 0
	for _, msg := range torso {
		torsoTokens += cc.counter.EstimateMessageTokens(msg)
	}
	totalTokensAfterPrune := cc.counter.EstimateMessages(msgCopy)
	totalTokensAfter := totalTokensAfterPrune - torsoTokens + summaryTokens
	tokensSaved := currentTokens - totalTokensAfter

	savingsRatio := float64(tokensSaved) / float64(currentTokens)
	if savingsRatio < cc.config.AntiThrashMinSavings {
		cc.tracker.RecordResult(0, currentTokens)
		return nil, nil
	}

	summaryMsg := SummaryMessage(summary)
	compressed := cc.assembler.Assemble(head, summaryMsg, tail, cc.tracker.IsFirstCompaction())

	result.Messages = compressed
	result.TokensSaved = tokensSaved
	result.WasCompressed = true

	cc.tracker.RecordResult(tokensSaved, currentTokens)
	return result, nil
}

func (cc *ContextCompressor) EstimateMessages(messages []providers.Message) int {
	return cc.counter.EstimateMessages(messages)
}

func (cc *ContextCompressor) EstimateMessageTokens(msg providers.Message) int {
	return cc.counter.EstimateMessageTokens(msg)
}
