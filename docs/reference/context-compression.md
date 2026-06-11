# Context Compression

PicoClaw includes a pluggable context compression system, modeled after the Hermes Agent compression methodology. It reduces context window usage by selectively pruning verbose tool outputs, segmenting history, and summarizing older messages via the LLM.

## Architecture

The compression pipeline runs in four phases:

- **Phase 1 — Pruner**: Replaces verbose tool results (>200 characters) with short inline placeholders.
- **Phase 2 — Segmenter**: Divides conversation history into Head (non-compressible), Torso (compressible), and Tail (recent, preserved). Tool-call/tool-result pairs are kept intact. The last user message is preserved.
- **Phase 3 — Summarizer**: Calls the LLM to produce a structured summary of the Torso segment.
- **Phase 4 — Assembler**: Builds the compressed history by combining Head + summary + Tail, with a first-compaction system note. Anti-thrashing and adaptive cooldown prevent excessive compression.

## Configuration

Set the context manager to `"hermes"` to enable compression:

```json
{
  "agents": {
    "defaults": {
      "context_manager": "hermes",
      "context_manager_config": {
        "model": "",
        "threshold": 0.50,
        "soft_threshold": 0.0,
        "hard_threshold": 0.0,
        "tail_budget_ratio": 0.20,
        "summary_token_ratio": 0.20,
        "min_summary_tokens": 256,
        "max_summary_tokens": 1024,
        "max_sequential_low_savings": 2,
        "cooldown_rounds": 3
      }
    }
  }
}
```

### Configuration fields

| Field | Default | Description |
|---|---|---|
| `model` | `""` | Model for summarization; empty = use the agent's current model |
| `threshold` | `0.50` | Fraction of context window that triggers proactive compression (agent-level) |
| `soft_threshold` | `0.0` | Soft warning threshold (0 = same as threshold) |
| `hard_threshold` | `0.0` | Hard limit threshold (0 = same as threshold + 0.15) |
| `tail_budget_ratio` | `0.20` | Fraction of context budget reserved for the Tail segment |
| `summary_token_ratio` | `0.20` | Fraction of the Torso token count allocated for the summary |
| `min_summary_tokens` | `256` | Minimum summary length in tokens |
| `max_summary_tokens` | `1024` | Maximum summary length in tokens |
| `max_sequential_low_savings` | `2` | Consecutive low-savings compressions before anti-thrashing kicks in |
| `cooldown_rounds` | `3` | Number of turns to skip compression after thrashing is detected |

## Triggers

Compression fires in two modes:

1. **Proactive (sync)**: Before the first LLM call in a turn, if the history exceeds the threshold budget (50% by default).
2. **Summarize (async)**: After each turn, if the tracker allows it.

Both are handled by the `hermesContextManager.Compact()` method, which is called automatically by the agent loop.

## Manual Compression

Use the `/compress` command in any channel to force an immediate compression of the current session:

```
/compress
```

This is equivalent to the async summarize trigger.

## Compatibility

The `"hermes"` context manager is implemented as a drop-in replacement for the default `"legacy"` manager. All existing `ContextManager` interface consumers (agent loop, CLI, channels) work without changes. The legacy manager remains available and is used when `context_manager` is empty or `"legacy"`.

## File Reference

| File | Purpose |
|---|---|
| `pkg/compression/compressor.go` | Phase orchestrator — entry point for the 4-phase pipeline |
| `pkg/compression/token_counter.go` | Token counting via tiktoken-go with heuristic fallback |
| `pkg/compression/types.go` | Shared types (`Config`, `CompressedSummary`) |
| `pkg/compression/config.go` | Configuration defaults and validation |
| `pkg/compression/pruner.go` | Phase 1 — tool output replacement |
| `pkg/compression/segmenter.go` | Phase 2 — head/torso/tail boundary detection |
| `pkg/compression/summarizer.go` | Phase 3 — LLM structured summarization |
| `pkg/compression/assembler.go` | Phase 4 — compressed message assembly |
| `pkg/compression/tracker.go` | Anti-thrashing and adaptive cooldown |
| `pkg/agent/context_hermes.go` | `hermesContextManager` implementing `ContextManager` |
