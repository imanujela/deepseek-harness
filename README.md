# deepseek-harness

A DeepSeek V4 optimization layer for agent harnesses — a Windows-native fork and
substantial rewrite of [`oh-my-deepseek-harness`](https://github.com/yuanchenglu/oh-my-deepseek-harness).

## Thanks

- **Original maker:** [`yuanchenglu/oh-my-deepseek-harness`](https://github.com/yuanchenglu/oh-my-deepseek-harness)
  — MIT © 2026 yuanchenglu. The cognitive-gate architecture, the plan/memory/checkpoint
  model and the innovation numbering (I-xx) are theirs. This fork keeps the upstream
  license and attribution, and preserves the original READMEs in
  [`docs/upstream/`](docs/upstream/).
- **DeepSeek** — the V4.x models this layer is tuned against. The behavioural
  findings referenced throughout come from public discussion of `DeepSeek-V4.1-Flash`
  on Hugging Face and Reddit, not from any private access.
- **Nous Research** — [Hermes Agent](https://github.com/NousResearch/hermes-agent),
  the host this plugin is written for.

---

## What it is

Three layers, no changes to the host's core code:

```
┌──────────────────────────────────────────────────────────────┐
│ Layer 1  deepseek_harness   cognitive gate · intent router    │
│                             plan/memory/checkpoint tools      │
├──────────────────────────────────────────────────────────────┤
│ Layer 2  deepseek_context   optional context engine           │
│                             (needs an API key; off by default)│
├──────────────────────────────────────────────────────────────┤
│ Layer 3  harness_server     FastAPI server + pure domain layer│
│                             (tools do NOT need it to run)     │
└──────────────────────────────────────────────────────────────┘
```

**Layer 1 works with no API key and no server.** That is the design centre of this
fork: the cognitive gate, the constraint audit, the plan/memory/checkpoint tools,
the thinking budget, the revision gate and the skill router all run locally.

---

## Installation

### Requirements

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11 or 3.12 | 3.10 works for the package core; the Hermes host matrix is 3.11–3.12 |
| Hermes Agent | v0.19.0 or newer | newer versions pass; the matrix is a floor, not a ceiling |
| API key | **none required** | only the optional Layer-2 context engine needs one |
| Bash | not required | only for the optional `scripts/*.sh` wrappers |

### Steps

```powershell
git clone https://github.com/imanujela/deepseek-harness
cd deepseek-harness
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

`pydantic` is a base dependency — the in-process domain models import it. The
`server` extra adds FastAPI + uvicorn for `deepseek-harness server`; the `context`
extra adds the OpenAI client for Layer 2.

**Tip:** use a venv. An editable install into a global interpreter works, but the
doctor's dependency report is then shared with everything else on the machine.

### Preview, then install

```powershell
deepseek-harness install --dry-run   # writes nothing; prints every target path
deepseek-harness install             # transactional: rolls back on failure
```

### Verify

```powershell
deepseek-harness doctor
```

```
status: ok
exit_code: 0
summary: 11 passed, 0 failed
```

`doctor` never prints a secret. With no key it reports the provider as a
**pass-with-note**: the core tools run fully offline.

### Uninstall

```powershell
deepseek-harness uninstall                         # keeps data + distribution
deepseek-harness uninstall --purge-data --confirm  # also deletes the data root
```

---

## Using it with different agents

The plugin **hooks** (`pre_llm_call`, `pre_tool_call`, …) are Hermes Agent's plugin
API. Other agents get the parts that are portable: the CLI, the JSON/TypeScript
contract, and the packaged skill. This section says exactly which is which rather
than pretending one integration fits all.

### Hermes Agent — native, everything wired

```powershell
deepseek-harness install
```

Deploys two plugin adapters and the skill:

| Path | Purpose |
|---|---|
| `%LOCALAPPDATA%\hermes\plugins\deepseek-harness\` | Layer 1 plugin adapter |
| `%LOCALAPPDATA%\hermes\plugins\deepseek-context\` | Layer 2 context engine adapter |
| `%LOCALAPPDATA%\hermes\skills\deepseek-harness\` | the skill (auto-loaded by Hermes) |

Restart Hermes (or start a new session) so the plugin registry reloads. Verify with
`hermes plugins list`.

### Claude Code — skill + CLI

Hooks are not portable, so the value here is the **skill** (process discipline for
the model) and the **CLI** (plan/memory/checkpoint operations from Bash).

```bash
# 1) give Claude Code the skill
mkdir -p ~/.claude/skills/deepseek-harness
cp skills/deepseek-harness/SKILL.md ~/.claude/skills/deepseek-harness/SKILL.md

# 2) the CLI is already on PATH from `pip install -e .`
deepseek-harness doctor
deepseek-harness plan list --json
```

If you use a skills registry, `npx skills@latest add imanujela/deepseek-harness`
pulls the same `skills/` tree; the manual copy above has no extra dependencies and
is the safer bet.

What you get: the model is told to check evidence before claiming done, to write
the failing test first, and to stop when a deliverable exists. What you do **not**
get: the `pre_llm_call` injections (constraint pinning, thinking budget) — those
require Hermes.

### Codex CLI / other coding agents — skill + CLI

Same shape as Claude Code: point the agent's skill loader at
`skills/deepseek-harness/SKILL.md`, and let it call the CLI. Every command accepts
`--json`, so an agent can parse the result instead of scraping prose.

```bash
deepseek-harness plan create --task "..." --json
deepseek-harness memory tag --content "..." --json
deepseek-harness audit --json
```

### Cursor / editors — contract + rules

The JSON Schema is the integration point: `schemas/harness_tools.schema.json`
describes every payload, and `types/harness_types.d.ts` is generated from it
(`python scripts/gen_ts_types.py --check` fails when the two drift).

Drop a rule file so the editor's agent knows the contract:

```
.cursor/rules/deepseek-harness.mdc   →  point it at types/harness_types.d.ts
```

### Anything that speaks MCP or shell

The CLI is the universal surface. `--json` everywhere, `--data-root` to select a
non-default data directory, `--help` per subcommand.

---

## WHAT YOU CHANGED — and why it is better

This is a rewrite, not a patch set. Five problems drove it.

### 1. The tools went to sleep — 20 seconds at a time

**Before:** every tool call ran `Supervisor().start()` before its request. That is a
`time.sleep(0.1)` readiness poll with a 20-second budget, then a 30-second httpx
timeout on top. A cold or wedged server meant
`StartError("Harness Server did not become ready before timeout")` on *every* call,
and the handler returned `{"error": ...}` — which the model silently skipped. The
symptom was not an error message; it was a turn that stalled and then quietly lost
facts it should have used.

**Now:** the nine registered tools dispatch **in-process** through `HarnessDomain`.
No spawn, no sleep, no HTTP. `plan_create` measures **240 ms** instead of ~20 s.

**Why it is better:** the timeout was the single largest source of wasted wall-clock
time and dropped context. Removing the round trip removes both. The server still
exists for people who want it (`deepseek-harness server`), but nothing in the tool
path depends on it.

### 2. It demanded an API key and a server it did not need

**Before:** `doctor` exited non-zero without `DEEPSEEK_API_KEY`, and the context
engine misbehaved when unkeyed. A user with no key could not tell whether the
product worked.

**Now:** the provider check passes with a note — "no key set; core tools run fully
offline; a key is only needed for the optional Layer-2 context engine". The engine
gained `is_configured()`; without a key `compress()` returns your messages
untouched, `register()` installs no hooks, and every error path logs
`exc_info=True` instead of swallowing.

**Why it is better:** you can install it, run it, and trust its reports before you
ever create a key. The failure mode changed from "refuses to start" to "starts,
does less, says so".

### 3. The model kept switching answers without evidence, and never stopped verifying

Two published behaviours drove this:

- Intrinsic self-correction *degrades* reasoning — GPT-3.5 falls 75.9% → 74.7% on
  GSM8K and 75.8% → **41.8%** on CommonSenseQA after two self-critique rounds
  ([arXiv 2310.01798](https://arxiv.org/abs/2310.01798)). Models flip correct
  answers to wrong more often than the reverse.
- Agents get the answer by turn 6–12 and then spend 20–40 turns proving they have
  not erred, with no stopping criterion: 2.9M input tokens where the actual edit was
  3% of spend ([DeepSeek-V4.1-Flash discussion #39](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/discussions/39)).

**Now:** `RevisionGate` refuses an answer switch unless it carries evidence of a
declared type (`tool_result`, `test_run`, `citation`, `user_instruction`) and
reports a `churn_rate()` you can drive to zero. `StopRule` treats a deliverable as a
valid stopping point; after it, only actions that name a falsifiable open question
are admitted, and past the soft cap the refusal becomes *"state what is missing or
unverified in one line, or stop"*.

**Why it is better:** both behaviours are gated mechanically instead of requested in
a prompt — which the same users measured to have **zero effect**.

### 4. Memory was written as a diary, and constraints were forgotten

**Before:** memory accepted anything, and hard constraints were extracted from the
user's message into a store that was then **never read again**. Upstream users
measured the result: the same hour of conversation wrote 13 KB of memory on V4 Flash
versus **161 KB** on V4.1 Flash — 12× — where the excess was decision logs, open
questions, self-advice and 109 `TBC` placeholders
([discussion #46](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/discussions/46)).
Constraint violations rise from 0% to 30–59% once the constraint leaves context.

**Now:** `memory_hygiene` classifies every candidate as fact / chronicle / placeholder
/ self_advice / preamble, refuses everything but facts under a byte budget, and can
compress a chronicle to the value-bearing line. `gate.py` re-pins up to seven hard
constraints verbatim every six turns — on the user-message side, so the cached system
prefix is untouched.

**Why it is better:** memory becomes a table of current values instead of a
narrative, and the constraint that mattered is still on screen when it matters.

### 5. Thinking had no floor, and skills were invisible

**Before:** reasoning effort was one of three words applied to everything, and the
agent had no idea which of the installed skills applied to the task in front of it.

**Now:** `thinking_budget` adds an `ultra` tier above max/xhigh/high (mapped to
`max` on the wire, because no provider accepts "ultra") with a **mechanical cut
rule**: a thought survives only if it is verb-led and not a ≥0.8 Jaccard duplicate
of one already kept; fewer than two survivors means converge, not think harder.
`skill_router` discovers **every installed skill from its own description** — 129 on
the author's machine — fires a fixed start set on turn one, and injects a per-child
skill contract into every `delegate_task` by rewriting its arguments.

**Why it is better:** more effort no longer means more tokens with no stopping rule,
and a skill installed tomorrow can trigger tomorrow with no code change.

### 6. The fork was inert on Windows

Upstream hardcoded `~/.hermes` in nine places; on Windows the real home is
`%LOCALAPPDATA%\hermes`, so the gate read a missing `MAP.md`, the learner discarded
every lesson, and the installer deployed plugins the host never loaded. Beyond that:
`os.fchmod`, `signal.SIGKILL`, `os.setsid`, `os.link` and POSIX mode verification all
crashed or silently failed on NT.

**Now:** every path resolves at call time through `harness_server/paths.py`
(`HERMES_HOME` → `%LOCALAPPDATA%\hermes` → `~/.hermes`), the POSIX-only calls are
guarded with Windows-correct equivalents, and `tests/test_windows_runtime.py` holds
the matrix.

**Why it is better:** the same code runs on both platforms, and the platform
difference is a tested contract rather than a crash.

---

## OVERALL CHANGES YOU MADE

Five sentences, if that is all you read:

1. **Killed the timeout** — tools dispatch in-process; ~20 s → 240 ms per call, no
   sleep loops, no spawn.
2. **Made it work with nothing** — no API key, no server; both are optional and the
   doctor says so instead of failing.
3. **Gated the two failure modes that waste the most** — answer churn requires
   evidence; verification stops when a deliverable exists.
4. **Made memory and constraints durable** — write-time hygiene for memory, verbatim
   re-pinning for constraints.
5. **Made the platform and the contract explicit** — Windows paths/termination fixed
   and tested; JSON Schema → TypeScript types with a drift check.

## Full list of major changes

**Runtime**

- `harness_server/domain.py` — all domain logic + `HarnessDomain`; `app.py` shrank
  1483 → 556 lines to a thin HTTP adapter.
- `tools.py` — in-process dispatch via a `_route` table; no `httpx`, `Supervisor`,
  `_ensure_server_running`, or `time.sleep` anywhere in the tool path.
- `doctor.py` — `hermes` = pass-with-note above the tested matrix; `provider` =
  pass-with-note with no key; `server`/`port` = note-only when nothing runs.
- `deepseek_context/*` — keyless degradation (`is_configured()`, lossless no-op
  `compress()`, no hooks without a key), `exc_info=True` on every error path.
- `installer.py` + `lifecycle.py` — deploy the skill alongside the adapters;
  uninstall/upgrade scoped to managed roots with a `packaged_skill` backup item.

**Discipline**

- `thinking_budget.py` — `ultra` tier, `wire_effort()`, Jaccard cut rule, yield
  floor, hysteresis demotion.
- `discipline.py` — `RevisionGate` (evidence-gated switches, `churn_rate()`) and
  `StopRule` (deliverable gate, cap refusal naming what is unverified).
- `memory_hygiene.py` — fact/chronicle/placeholder/self_advice/preamble classifier,
  `admit()`, `compress_chronicle()`.
- `gate.py` — constraint pin every six turns, capped at seven entries.
- `skill_router.py` — discovery of every installed skill, fixed start set,
  description scoring, per-child contract injection via `pre_tool_call` modify.

**Contract**

- `schemas/harness_tools.schema.json` + `scripts/gen_ts_types.py` +
  `types/harness_types.d.ts` — one source of truth, `--check` drift gate.
- `pyproject.toml` — `pydantic` promoted to a base dependency (the in-process domain
  models need it at import time).

**Platform**

- `harness_server/paths.py`, `deepseek_harness/paths.py` — platform-aware home
  resolution, call-time.
- `os.fchmod` guard, `SIGKILL → SIGTERM` fallback, `start_new_session` equivalence,
  `_link_or_replace()` hard-link fallback, Windows-honest permission enforcement.
- `scripts/security/sbom.py` — portable SBOM generator.

**Docs and tests**

- `skills/deepseek-harness/SKILL.md` — the packaged skill.
- `AGENTS.md` — repo map and the load-bearing invariant (no spawning, no sleeping in
  the tool path).
- `docs/research/deepseek-optimizations-by-others.md` — 64 cited sources on DeepSeek
  behaviour inside agent harnesses.
- New suites: `test_domain`, `test_discipline`, `test_thinking_budget`,
  `test_memory_hygiene`, `test_ts_typegen`, `test_skill_router`, `test_skill_deploy`,
  `test_local_plugin`, `test_context_no_api`, `test_constraint_pinning`,
  `test_windows_runtime`.

## Verification

```
deepseek-harness doctor   →  exit 0, 11 passed / 0 failed
python -m pytest tests/    →  578 passed, 22 skipped, 0 failed
```

The 22 skips are host-gated, not broken: POSIX-only assertions, symlink privileges,
and process-matrix tests that require POSIX.

---

## Where things land

| Path | Purpose |
|---|---|
| `%LOCALAPPDATA%\hermes\plugins\deepseek-harness\` | Layer 1 plugin adapter |
| `%LOCALAPPDATA%\hermes\plugins\deepseek-context\` | Layer 2 context engine adapter |
| `%LOCALAPPDATA%\hermes\skills\deepseek-harness\` | the packaged skill |
| `%LOCALAPPDATA%\hermes\oh-my-deepseek-harness\config\config.yaml` | runtime config |
| `%LOCALAPPDATA%\hermes\oh-my-deepseek-harness\data\harness.db` | single SQLite |
| `%LOCALAPPDATA%\hermes\oh-my-deepseek-harness\runtime\` | PID + lock file |
| `%LOCALAPPDATA%\hermes\oh-my-deepseek-harness\logs\` | server log |
| `%LOCALAPPDATA%\hermes\memories\` | audit events, violations, lessons |

## Usage

```powershell
# Server lifecycle (optional — the tools do not need it)
deepseek-harness server start | status | restart | stop

# Diagnostics
deepseek-harness doctor
deepseek-harness doctor --json

# Memory import from the Hermes memories directory
deepseek-harness memory import --dry-run
deepseek-harness memory import

# Constraint audit report
deepseek-harness audit
```

Every command accepts `--json` and `--data-root`.

## Documentation

| Guide | What it covers |
|---|---|
| [docs/guides/INSTALL.md](docs/guides/INSTALL.md) | step-by-step install, per-platform notes, troubleshooting |
| [docs/guides/DOCTOR.md](docs/guides/DOCTOR.md) | every doctor check, its exit code and what to do when it fails |
| [docs/guides/PRIVACY.md](docs/guides/PRIVACY.md) | what leaves the machine (nothing, without a key) and what is stored where |
| [docs/capabilities/CAPABILITY_MATRIX.md](docs/capabilities/CAPABILITY_MATRIX.md) | which capability works on which platform, with the tested version floor |
| [AGENTS.md](AGENTS.md) | repo map and the load-bearing invariants for contributors |

## Upstream reference

| Resource | Link |
|---|---|
| Upstream repository | [github.com/yuanchenglu/oh-my-deepseek-harness](https://github.com/yuanchenglu/oh-my-deepseek-harness) |
| Upstream README (ZH) | [docs/upstream/README.zh-CN.md](docs/upstream/README.zh-CN.md) |
| Upstream README (EN) | [docs/upstream/README.en.md](docs/upstream/README.en.md) |
| Hermes Agent | [github.com/NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) |
| Hermes Agent docs | [hermes-agent.nousresearch.com/docs](https://hermes-agent.nousresearch.com/docs) |

## License

MIT — see [LICENSE](LICENSE). Upstream is MIT © 2026 yuanchenglu; this fork keeps
that license and attribution.

Agent-facing docs: [`AGENTS.md`](AGENTS.md) (repo map and invariants) and the
deployable skill [`skills/deepseek-harness/SKILL.md`](skills/deepseek-harness/SKILL.md).
