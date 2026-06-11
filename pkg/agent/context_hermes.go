package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/sipeed/picoclaw/pkg/compression"
	runtimeevents "github.com/sipeed/picoclaw/pkg/events"
	"github.com/sipeed/picoclaw/pkg/logger"
	"github.com/sipeed/picoclaw/pkg/providers"
)

func init() {
	RegisterContextManager("hermes", newHermesContextManager)
}

func newHermesContextManager(cfg json.RawMessage, al *AgentLoop) (ContextManager, error) {
	compressionCfg := compression.DefaultConfig()
	if cfg != nil {
		if err := json.Unmarshal(cfg, &compressionCfg); err != nil {
			return nil, fmt.Errorf("invalid hermes context manager config: %w", err)
		}
	}
	if err := compressionCfg.Validate(); err != nil {
		return nil, fmt.Errorf("invalid hermes config: %w", err)
	}

	counter := compression.NewTokenCounter()
	llmSummarizer := &agentLLMSummarizer{al: al}
	compressor := compression.NewContextCompressor(&compressionCfg, counter, llmSummarizer)

	return &hermesContextManager{
		al:         al,
		config:     &compressionCfg,
		counter:    counter,
		compressor: compressor,
		summarizing: sync.Map{},
	}, nil
}

type agentLLMSummarizer struct {
	al *AgentLoop
}

func (s *agentLLMSummarizer) Summarize(ctx context.Context, prompt string, modelName string) (string, error) {
	agent := s.al.registry.GetDefaultAgent()
	if agent == nil {
		return "", fmt.Errorf("no default agent available")
	}

	const maxRetries = 3
	var resp *providers.LLMResponse
	var err error

	for attempt := 0; attempt < maxRetries; attempt++ {
		s.al.activeRequests.Add(1)
		resp, err = func() (*providers.LLMResponse, error) {
			defer s.al.activeRequests.Done()
			return agent.Provider.Chat(
				ctx,
				[]providers.Message{{Role: "user", Content: prompt}},
				nil,
				modelName,
				map[string]any{
					"max_tokens":       1024,
					"temperature":      0.3,
					"prompt_cache_key": agent.ID + ":summarizer",
				},
			)
		}()

		if err == nil && resp != nil && resp.Content != "" {
			return resp.Content, nil
		}
		if attempt < maxRetries-1 {
			time.Sleep(time.Duration(attempt+1) * 200 * time.Millisecond)
		}
	}

	return "", fmt.Errorf("summarization LLM call failed after %d retries: %w", maxRetries, err)
}

type hermesContextManager struct {
	al          *AgentLoop
	config      *compression.Config
	counter     *compression.TokenCounter
	compressor  *compression.ContextCompressor
	summarizing sync.Map
}

func (m *hermesContextManager) Assemble(_ context.Context, req *AssembleRequest) (*AssembleResponse, error) {
	agent := m.al.registry.GetDefaultAgent()
	if agent == nil {
		return &AssembleResponse{}, nil
	}
	history := agent.Sessions.GetHistory(req.SessionKey)
	summary := agent.Sessions.GetSummary(req.SessionKey)
	return &AssembleResponse{
		History: history,
		Summary: summary,
	}, nil
}

func (m *hermesContextManager) Compact(ctx context.Context, req *CompactRequest) error {
	switch req.Reason {
	case ContextCompressReasonProactive, ContextCompressReasonRetry:
		result, err := m.compressSession(ctx, req.SessionKey, req.Reason)
		if err != nil {
			return err
		}
		if result {
			m.al.emitEvent(
				runtimeevents.KindAgentContextCompress,
				m.al.newTurnEventScope("", req.SessionKey, nil).meta(0, "hermesCompress", "turn.context.compress"),
				ContextCompressPayload{
					Reason: req.Reason,
				},
			)
		}
	case ContextCompressReasonSummarize:
		m.maybeCompress(req.SessionKey)
	}
	return nil
}

func (m *hermesContextManager) Ingest(_ context.Context, _ *IngestRequest) error {
	return nil
}

func (m *hermesContextManager) Clear(_ context.Context, sessionKey string) error {
	agent := m.al.registry.GetDefaultAgent()
	if agent == nil || agent.Sessions == nil {
		return fmt.Errorf("sessions not initialized")
	}
	agent.Sessions.SetHistory(sessionKey, []providers.Message{})
	agent.Sessions.SetSummary(sessionKey, "")
	return agent.Sessions.Save(sessionKey)
}

func (m *hermesContextManager) compressSession(ctx context.Context, sessionKey string, reason ContextCompressReason) (bool, error) {
	agent := m.al.registry.GetDefaultAgent()
	if agent == nil {
		return false, nil
	}

	history := agent.Sessions.GetHistory(sessionKey)
	if len(history) < m.config.MinHistoryLength {
		return false, nil
	}

	modelName := agent.Model
	contextWindow := agent.ContextWindow
	if contextWindow <= 0 {
		contextWindow = 128000
	}

	result, err := m.compressor.Compress(ctx, history, modelName, contextWindow)
	if err != nil {
		return false, fmt.Errorf("compression failed: %w", err)
	}
	if result == nil || !result.WasCompressed {
		return false, nil
	}

	agent.Sessions.SetHistory(sessionKey, result.Messages)
	if err := agent.Sessions.Save(sessionKey); err != nil {
		return false, fmt.Errorf("failed to save compressed session: %w", err)
	}

	logger.InfoCF("agent", "Hermes context compressed", map[string]any{
		"session_key":  sessionKey,
		"reason":       reason,
		"tokens_saved": result.TokensSaved,
	})

	return true, nil
}

func (m *hermesContextManager) maybeCompress(sessionKey string) {
	agent := m.al.registry.GetDefaultAgent()
	if agent == nil {
		return
	}

	history := agent.Sessions.GetHistory(sessionKey)
	tokenEstimate := m.counter.EstimateMessages(history)
	threshold := int(float64(agent.ContextWindow) * m.config.AgentThreshold)

	if len(history) <= m.config.MinHistoryLength || tokenEstimate < threshold {
		return
	}

	compressKey := agent.ID + ":" + sessionKey
	if _, loading := m.summarizing.LoadOrStore(compressKey, true); !loading {
		go func() {
			defer m.summarizing.Delete(compressKey)
			defer func() {
				if r := recover(); r != nil {
					logger.WarnCF("agent", "Hermes compression panic", map[string]any{
						"session_key": sessionKey,
						"panic":       r,
					})
				}
			}()

			ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
			defer cancel()

			result, err := m.compressor.Compress(ctx, history, agent.Model, agent.ContextWindow)
			if err != nil {
				logger.WarnCF("agent", "Hermes compression failed", map[string]any{
					"session_key": sessionKey,
					"error":       err.Error(),
				})
				return
			}
			if result == nil || !result.WasCompressed {
				return
			}

			agent.Sessions.SetHistory(sessionKey, result.Messages)
			if err := agent.Sessions.Save(sessionKey); err != nil {
				logger.WarnCF("agent", "Failed to save compressed session", map[string]any{
					"session_key": sessionKey,
					"error":       err.Error(),
				})
				return
			}

			logger.InfoCF("agent", "Hermes async compression complete", map[string]any{
				"session_key":  sessionKey,
				"tokens_saved": result.TokensSaved,
			})
		}()
	}
}
