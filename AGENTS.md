# AGENTS.md — oh-my-deepseek-harness

Pointer sheet. Code, tests, and `docs/` are the sources of truth; this file
names what they cannot confess.

## Three packages

- `src/deepseek_harness/` — the Hermes plugin: cognitive gate, intent router,
  constraint audit, plan/memory/checkpoint tools. CLI: `deepseek-harness`
  (install, doctor, upgrade, recover, uninstall, server, memory, plan, audit).
- `src/harness_server/` — domain plus HTTP adapter. `domain.py` holds the pure
  `HarnessDomain` (plans, memories, checkpoints); `app.py` and `server.py`
  expose it through FastAPI.
- `src/deepseek_context/` — optional Layer-2 context engine. The engine is the
  only key-gated piece; gate, tools, memory, and plan work keyless.

## The invariant

Tool-path calls answer from `harness_server.domain.HarnessDomain` in-process.
Historical bug: per-call server spawning and `time.sleep` in the tool path hit
the 20 s supervisor poll loop and timed out every cold session; localization
fixed it. Keep spawning and sleeping out of the tool path;
`deepseek-harness server` is an optional surface, never a tool-path
dependency.

## Where things live

- Agent docs: this file, `skills/deepseek-harness/SKILL.md`, `docs/`.
- Deployment targets: plugin adapters under `<hermes home>/plugins/`, the
  deployed skill copy under `<hermes home>/skills/deepseek-harness/`.
  `HERMES_HOME` redirects both (defaults: `%LOCALAPPDATA%\hermes` on Windows,
  `~/.hermes` on POSIX).
- Managed files: `installer.py` builds `InstallPlan.desired_files` and
  `desired_directories`; lifecycle backup, replace, and removal operate on
  those maps. New managed content goes into the plan, not an ad-hoc write.

## Run

- Tests: `python -m pytest tests/ -q` (on Windows: under git-bash).
- `tests/test_installer_e2e.py` pins installer source text and the two-entry
  `deployment_paths` list; extend your own test file instead of editing it.

## Docs conventions

- Paths: Hermes-home-relative or repo-relative; every document states the
  platform default instead of baking in one machine's layout.
- Error handling: every except branch logs (`logger.debug(...,
  exc_info=True)`) or re-raises.
- Skills: the hardline authoring standard — frontmatter at byte 0, description
  ≤ 60 chars ending in a period, every procedure step closing on a checkable
  completion criterion.
