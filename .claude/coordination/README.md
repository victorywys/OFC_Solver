# coordination/

Shared multi-agent state. Committed to git so every agent sees the same truth.

- `tasks.json`     — task list (jq-edited by helpers)
- `messages.log`   — broadcasts and DMs (append-only)
- `PROJECT.md`     — project context for new agents
- `locks/`         — per-host advisory file locks (gitignored)

Edit through the `claude-agents-*` helpers, not by hand.
