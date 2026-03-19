# Contributing & Documentation Conventions

## Docs vs. Issues — What Goes Where

| Content type | Location |
|---|---|
| Task progress, status, blockers | GitHub issue comments + open/closed state |
| Design rationale, architectural constraints, resolved decisions | `docs/pipeline.md` |
| Scientific decisions pending resolution | Relevant `docs/` file as an `**UNRESOLVED:**` block |

## GitHub Issue Conventions

- **Commits:** Use `Fixes #N` to auto-close issues on merge.
- **New issues:** Include a clear title, detailed description, and a label (`infrastructure`, `data`, `feature`, `modelling`, `investigation`, `bug`, `decision`).
- **Code references:** Use `[#N](https://github.com/MaxEtherington/PUB-framework-Alfonso/issues/N)` in comments and docstrings.
- **Decision issues:** If work is blocked on a non-coding decision requiring user input, create an issue labelled `decision` with full context before stopping work.
