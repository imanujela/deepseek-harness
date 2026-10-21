---
name: deepseek-harness
description: Cognitive plan/memory/checkpoint tools, no API key needed.
version: 0.1.0
author: Kulku (imanujela), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [deepseek, harness, plan, memory, checkpoint, hermes-plugin, local-first]
    related_skills: []
---

# DeepSeek Harness Skill

DeepSeek Harness registers cognitive plan, memory, and checkpoint tools into
Hermes from a pure-local Python distribution. Tool calls resolve in-process
against `harness_server.domain.HarnessDomain`, so no API key is required; the
optional Layer-2 context engine (`deepseek_context`) is the only key-gated
component. This skill covers the `deepseek-harness` CLI lifecycle and
deployment surface, not tool semantics.

## When to Use

- Install, upgrade, recover, or uninstall the DeepSeek Harness plugin for Hermes.
- Start, stop, or check the optional local Harness Server.
- Triage a failed install or doctor report before debugging plugin code.
- Explain why tools answer without an API key, or why Layer-2 context
  summarization is silent in a keyless session.
- Don't use for: authoring Hermes skills, Hermes core configuration, or
  changing the domain logic itself.

## Prerequisites

- Python `>=3.10,<3.13` (enforced by `pyproject.toml`).
- Editable install with the server extra, from the repo root:
  `terminal(command="python -m pip install -e .[server]", timeout=300)` —
  this puts `deepseek-harness` on PATH.
- No API key required. `DEEPSEEK_API_KEY` or `OPENROUTER_API_KEY` enables
  only the optional Layer-2 context engine.
- A Hermes home to deploy into: `HERMES_HOME` wins when set; otherwise the
  platform default — `%LOCALAPPDATA%\hermes` on Windows, `~/.hermes` on POSIX.
- `install` starts the managed server and requires it to reach the ready
  state, so the `server` extra is a hard prerequisite for the transaction.

## How to Run

Frame every command through the `terminal` tool and add `--json` for a
machine-readable payload:

```
terminal(command="deepseek-harness install --dry-run --json", timeout=60)
```

`--data-root PATH` relocates the product data root (default:
`<hermes home>/oh-my-deepseek-harness`).

## Quick Reference

- Dry run: `terminal(command="deepseek-harness install --dry-run --json")`
- Install: `terminal(command="deepseek-harness install --json", timeout=120)`
- Diagnostics: `terminal(command="deepseek-harness doctor --json")`
- Server: `terminal(command="deepseek-harness server status --json")` — subcommands `start|status|stop|restart`
- Memory: `terminal(command="deepseek-harness memory import --dry-run --json")` — subcommands `import` / `delete --source ID --confirm`
- Plan: `terminal(command="deepseek-harness plan archive --plan-id ID --json")` — subcommands `archive` / `delete --plan-id ID --confirm`
- Audit: `terminal(command="deepseek-harness audit --json")`
- Lifecycle: `terminal(command="deepseek-harness upgrade --json", timeout=120)`; also `recover --json`, `uninstall --json`

## Procedure

1. Preflight: run `install --dry-run --json`. Done when `state` is `planned`,
   `changed` is `false`, and the Hermes home tree is byte-for-byte unchanged.
2. Install: run `install --json`. Done when `state` is `installed` and
   `ready` is `true`.
3. Confirm deployment: `read_file` each of
   `<hermes home>/plugins/deepseek-harness/plugin.yaml`,
   `<hermes home>/plugins/deepseek-context/plugin.yaml`, and
   `<hermes home>/skills/deepseek-harness/SKILL.md`; done when all three exist
   and none is a symlink.
4. Validate: run `doctor --json`. Done when the exit code is 0 and the report
   status is `ok`.
5. Round-trip: restart Hermes, then call `plan_create` once. Done when the
   tool payload returns a plan id.

## Pitfalls

- Keyless Layer 2: with no API key, `deepseek_context` degrades and registers
  no summarization hooks. That is the designed behavior, not a failure.
- Doctor exits 0 with no server running: the tool path answers in-process;
  `server` only manages the optional FastAPI surface.
- Two deployment roots: plugin adapters live under the Hermes home `plugins/`
  directory, while the deployed skill copy lives under the Hermes home
  `skills/` directory. `HERMES_HOME` redirects both.
- Edited managed files are never overwritten: a same-version reinstall of a
  hand-edited managed file rejects the transaction (exit 4) and preserves the
  user file; remove the file, or take a version bump and run `upgrade`, which
  backs up everything first and then replaces managed content.

## Verification

1. `terminal(command="deepseek-harness doctor --json")` exits 0 with status
   `ok`.
2. Tool round trip: in a restarted Hermes session, `plan_create` returns a
   payload containing a plan id, and `plan_status` on that id reports the
   created steps. Both conditions hold, the deployment is verified.
