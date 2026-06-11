package compression

import (
	"github.com/sipeed/picoclaw/pkg/providers"
)

type Assembler struct{}

func NewAssembler() *Assembler {
	return &Assembler{}
}

func (a *Assembler) Assemble(
	head []providers.Message,
	summaryMsg providers.Message,
	tail []providers.Message,
	isFirstCompaction bool,
) []providers.Message {
	result := make([]providers.Message, 0, len(head)+1+len(tail)+1)
	result = append(result, head...)
	result = append(result, summaryMsg)
	result = append(result, tail...)

	if isFirstCompaction {
		result = append(result, providers.Message{
			Role: "system",
			Content: "Note: The middle portion of the conversation has been summarized " +
				"to save context space. The summary above preserves key information " +
				"from the compressed region.",
		})
	}

	return result
}
