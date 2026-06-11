package commands

import "context"

func compressCommand() Definition {
	return Definition{
		Name:        "compress",
		Description: "Manually trigger context compression for the current session",
		Usage:       "/compress",
		Aliases:     []string{"compact"},
		Handler: func(_ context.Context, req Request, rt *Runtime) error {
			if rt == nil || rt.Compact == nil {
				return req.Reply(unavailableMsg)
			}
			if err := rt.Compact(); err != nil {
				return req.Reply("Failed to compress context: " + err.Error())
			}
			return req.Reply("Context compression triggered successfully!")
		},
	}
}
