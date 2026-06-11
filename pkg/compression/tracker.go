package compression

import "sync"

type CompressionTracker struct {
	mu                     sync.Mutex
	consecutiveLowSavings  int
	cooldownRemaining      int
	totalCompressions      int
	lastTokensSaved        int
	isFirstCompaction      bool
}

func NewCompressionTracker() *CompressionTracker {
	return &CompressionTracker{
		isFirstCompaction: true,
	}
}

func (ct *CompressionTracker) RecordResult(tokensSaved int, totalBefore int) {
	ct.mu.Lock()
	defer ct.mu.Unlock()

	ct.totalCompressions++
	ct.lastTokensSaved = tokensSaved
	ct.isFirstCompaction = false

	if totalBefore > 0 {
		ratio := float64(tokensSaved) / float64(totalBefore)
		if ratio < 0.10 {
			ct.consecutiveLowSavings++
		} else {
			ct.consecutiveLowSavings = 0
		}
	}
}

func (ct *CompressionTracker) RecordFailure() {
	ct.mu.Lock()
	defer ct.mu.Unlock()
	ct.cooldownRemaining = 3
}

func (ct *CompressionTracker) ShouldSkip(config *Config) bool {
	ct.mu.Lock()
	defer ct.mu.Unlock()

	if ct.cooldownRemaining > 0 {
		ct.cooldownRemaining--
		return true
	}
	if ct.consecutiveLowSavings > config.AntiThrashMaxConsecutive {
		return true
	}
	return false
}

func (ct *CompressionTracker) IsFirstCompaction() bool {
	ct.mu.Lock()
	defer ct.mu.Unlock()
	return ct.isFirstCompaction
}
