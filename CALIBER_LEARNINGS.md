# Caliber Learnings

Accumulated patterns and anti-patterns from development sessions.
Auto-managed by [caliber](https://github.com/caliber-ai-org/ai-setup) — do not edit manually.

- **[gotcha]** `caliber score --json --quiet | head -1` only emits `{` — pipe the full output and parse with `python3 -c "import sys,json; d=json.load(sys.stdin); ..."` or omit `head`.
- **[fix]** `caliber refresh` may return an empty response when invoked without an explicit working directory — always run as `cd /path/to/project && caliber refresh` or from the project root to avoid silent no-ops.
- **[env]** `caliber config` is interactive-only and accepts no flags (only `-h/--help`). Options like `--show`, `--list-providers`, and `--provider` do not exist and will error. Check `~/.caliber/config.json` directly to read the current provider config.
- **[gotcha]** `caliber config --show` returns "NO_CONFIG" even when a provider is already configured — the actual provider config lives at `~/.caliber/config.json`, not surfaced by any non-interactive command.
- **[pattern]** To add GitHub Copilot as a sync target alongside Claude Code, use `caliber init --agent claude,github-copilot` — Copilot is a sync target only and requires another provider (e.g. `claude`) for generation.
