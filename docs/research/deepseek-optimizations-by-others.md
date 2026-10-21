# What other people have found actually works for DeepSeek V4.x inside an agent harness

**Compiled:** 2026-10-21 · **Scope:** DeepSeek V4.1-Flash (API id `deepseek-flash`, build 1017) and V4-Pro, as used in agent harnesses (DSH, Hermes, pi, dsc, Claude Code, OpenCode).
**Method:** vendor docs + model-card encoding reference + HF/GitHub discussion threads with raw numbers + arXiv preprints + shipped plugins. Every claim carries a URL. Claims are graded **[A]** official/vendor, **[B]** measured third-party with method and numbers, **[C]** single report/anecdote, **[S]** my inference.

**Why this exists:** our plugin (cognitive gate, intent router, plan/memory/checkpoint tools, thinking budget, revision gate, memory hygiene, skill router) was built against observed V4.1-Flash failures: answer churn, 33–50 turn verification loops, 12× memory bloat, pre-compaction instruction decay, and post-compaction misattribution of its own edits. This report is what other people measured about those same failures and what they shipped in response.

---

## TL;DR — the seven things that repeat across sources

1. **The model is not overthinking; it is over-acting.** Reasoning length looks normal; the number of *actions* inflates after the answer is already in hand. HF #39 counted 33–50 turns for a one-field edit, with 1.25M of 2.95M input tokens spent re-verifying a change that was 3% of the spend. [B] https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/discussions/39
2. **A deliverable is not treated as a stopping point.** The same thread's first requested fix is an explicit "a deliverable is a valid stopping point" signal, plus a soft cap that forces the model to *state what is missing* rather than continue. [B/C] same URL
3. **Compaction is where constraints die, and it is measurable.** Across 7 models and 1,323 episodes, compaction raised constraint violation from 0% to 30% (up to 59%); a ~47-token pinned block restored it to 0%. [B] https://arxiv.org/abs/2606.22528
4. **Instruction-count decay is a cliff, not a slope.** Perfect-response rate collapses to zero by 80 simultaneous rules for every model and format tested (960 calls/model, N from 10 to 160). [B] https://arxiv.org/html/2607.19257v1
5. **Prompt rules alone did nothing for memory bloat.** The same system-prompt section, rewritten to forbid decision logs and unsolicited research, produced "zero effect. Not less — the same." [B] https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/discussions/46
6. **Prefix cache is byte-exact and brutal.** Changing the system prompt anywhere → 0% hit; reordering two tools → 0%; appending a message → 98.7%. Cache-hit input is 50× cheaper than cache-miss on the official price sheet. [B/A] https://inferloop.dev/deepseek-harness/prefix-stability/ · https://api-docs.deepseek.com/quick_start/pricing/
7. **Effort is front-loaded, and "off" is a real option.** DeepSeek's own curve: effort 25→100 buys +9.2 pts average Pass@1 for ~2.5× output tokens, with most of the gain recovered by 60–80. On 30 small edit-and-test tasks, thinking **off** solved 30/30 at 36% lower cost than default `high`. [B] https://pirouter.ai/blog/reasoning-effort-one-name-five-physics · https://synthorai.io/blog/deepseek-v4-1-flash-1017-cost/

---

## 1. Official DeepSeek guidance for V4.x

### 1.1 Prompt format: effort prefix at index 0, mid-conversation system messages, DSML

The standalone encoder reference is the authoritative prompt-format document. Three V4.1 changes matter to a harness:

- **Reasoning effort is rendered as a numeric prefix and only at conversation index 0, only in thinking mode.** `Reasoning Effort: {budget} (range 1-100, ...)`; string aliases `low`→50, `high`→75, `max`→100; default `high` (75). [A] https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blame/main/encoding/README.md
- **Mid-conversation system messages are supported** via a dedicated token, and "behaves like a user message for the purpose of appending the assistant generation header." This is the mechanism a harness should use to re-pin constraints without touching the system prompt. [A] same URL
- **V4.1 DSML tags carry a leading space** (`<｜DSML｜ calls>`, `<｜DSML｜ invoke>`, `<｜DSML｜ parameter>`) where V4 had none. Parsers written for V4 mis-detect V4.1 output. [A] same URL · corroborated by serving stacks: https://rocm.docs.amd.com/projects/atom/en/latest/deepseek_v41_protocol.html and https://github.com/vllm-project/vllm/issues/58640

The ROCm protocol notes add two operational details: earlier assistant reasoning is dropped according to official rules, but **tool conversations preserve it**; and top-level tools attach to the first system message or a synthetic one. [A] https://rocm.docs.amd.com/projects/atom/en/latest/deepseek_v41_protocol.html

### 1.2 Thinking mode and effort

- Thinking mode is **on by default**, default effort `high`. API accepts OpenAI `reasoning_effort: low/high/max`, Anthropic `output_config.effort`, or `thinking: {type: enabled/disabled}`. [A] https://api-docs.deepseek.com/guides/thinking_mode/
- The API mapping is coarse: `minimal`→low, `medium`→high, `xhigh`→high, `ultra`→max. Our `thinking_budget.py` already documents this asymmetry; the vendor confirms it. [A] same URL
- The **numeric 1–100 dial is documented in the encoder/self-hosted path** (and accepted by serving adapters as an exact integer, default 75); the hosted API documents only three gears. Treat integer effort as self-host-only until you have probed the hosted endpoint. [A] encoding README (above) · https://rocm.docs.amd.com/projects/atom/en/latest/deepseek_v41_protocol.html · https://api-docs.deepseek.com/guides/thinking_mode/
- DeepSeek publishes an effort→score→token curve in the V4.1-Flash technical report (§5.3.3, Fig. 9): 25→100 moves average Pass@1 from 67.1% to 76.3%, DeepSWE v1.1 66.0%→74.2%, Terminal-Bench 2.1 82.4%→90.6%, "at the cost of roughly 2.5× more output tokens"; the 60–80 range already recovers most of the accuracy. [B, secondary reporting of A] https://pirouter.ai/blog/reasoning-effort-one-name-five-physics
- Official same-family evals were run at maximum effort with `temperature=1.0`, `top_p=0.95`. [B, secondary] https://deepseekagent.io/guides/deepseek-v4-1-flash-vs-v4-flash-harness

### 1.3 `reasoning_content` echo — the rule most harnesses get wrong

When the request carries `tools`, **`reasoning_content` from all previous turns must be passed back** and is concatenated into context. Without `tools`, it does not need to be passed and is ignored if sent. [A] https://api-docs.deepseek.com/guides/thinking_mode/

Two consequences, one measured and one reported:

- Passing it back is mandatory for interleaved tool reasoning; Roo Code's provider config says exactly this ("DeepSeek requires reasoning_content to be passed back during tool call continuation"). [B] https://github.com/RooCodeInc/Roo-Code/blob/137d3f4f/packages/types/src/providers/deepseek.ts
- HF #39's first reply: *"Make sure your harness is not stripping reasoning out of prior context — that can cause this type of behavior."* [C] https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/discussions/39
- Where harnesses do strip it, the workaround shipped is a replay layer that stores and re-sends `reasoning_content`: https://github.com/diegosouzapw/OmniRoute/blob/main/docs/routing/REASONING_REPLAY.md and https://github.com/esengine/DeepSeek-Reasonix/pull/4022

This is the cheapest single explanation for "it forgets what it already decided" that is consistent with both the vendor rule and the field reports.

### 1.4 Temperature and top_p: the old 1.0–1.3 advice is mostly inert

The historical table still exists: default `temperature` 1.0; **coding/math 0.0, data analysis 1.0, general conversation 1.3, translation 1.3, creative writing 1.5**. [A] https://api-docs.deepseek.com/quick_start/parameter_settings

But in thinking mode, **`temperature`, `presence_penalty` and `frequency_penalty` have no effect** — setting them does not error, it just does nothing. `top_p` for `deepseek-flash` only takes effect in thinking mode and is clamped to 0.95–1.0 (values below 0.95 are treated as 0.95); in non-thinking mode it is fixed at 1.0 and ignored. [A] https://api-docs.deepseek.com/guides/thinking_mode/

So a harness that "sets temperature 0 for coding" while leaving thinking on is configuring nothing. The measured field report agrees: malformed-tool-call rate did not change between `temperature` 0 and 1.3. [B] https://github.com/deepseek-ai/DeepSeek-V3/issues/1541

### 1.5 `max_tokens` and what is billed

- Context length 1M; **maximum output 384K**. [A] https://api-docs.deepseek.com/quick_start/pricing/
- Billing is by total input and output tokens; **reasoning is billed as output tokens**. [A] pricing page; [B] https://synthorai.io/blog/deepseek-v4-1-flash-1017-cost/ ("Thinking is the model's hidden reasoning before it answers, billed as output tokens")
- Reasoning tokens count against `max_tokens`. [B] https://github.com/thevibeworks/deepseek-code (README, "properties of the API, not of dsc")
- Under-sizing `max_tokens` does not just truncate prose — it truncates tool-call argument JSON. A live run with `max_tokens: 96` produced **16 of 25 HTTP 200 responses ending in `finish_reason: length` with incomplete argument JSON**. [B] https://github.com/chatdeepai/deepseek-api-reproducible-tests/blob/main/tool-calls/LIVE_RUN.md

### 1.6 Prefix cache: official rules and what they cost

Official rules: caching is on by default; a hit requires the request to **fully match a persisted cache prefix unit**. Sliding-window attention means each cached prefix is an independent complete unit; units are persisted at request boundaries, at detected common prefixes, and at fixed token intervals for long inputs. A request that diverges mid-prefix does not partially hit. [A] https://api-docs.deepseek.com/guides/kv_cache

Price effect (off-peak, per 1M input tokens): **$0.003 cache hit vs $0.15 cache miss — a 50× difference**; peak $0.006 vs $0.30. Output $0.60/$1.20 off-peak/peak. [A] https://api-docs.deepseek.com/quick_start/pricing/

Measured sensitivity, on a 7,384-token request (system + 25 tool schemas + one message), `deepseek-v4-flash`, 2026-08-17: [B] https://inferloop.dev/deepseek-harness/prefix-stability/

| Mutation | Hit rate |
|---|---:|
| Nothing changed | 98.8% |
| Append a message at the end | 98.7% |
| Add one line at the end of system prompt | **0.0%** |
| Insert one line at the start of system prompt | **0.0%** |
| Add a tool at the very end | 18.3% |
| Add a tool in the middle | 0.0% |
| Add a tool at the front | 0.0% |
| Swap the order of the first two tools | **0.0%** |
| Change one character in the last tool description | 17.3% |
| Switch model | 0.0% |

Cross-session evidence from DSH: with a 10,083-char system prompt and 39,282 chars of tool schemas byte-identical, session A (`test`) = 13,088 input tokens, 0% hit; session B (`test 2`) = 7,586 tokens, 5,504 cache-read, 42%; session C (repeat of A) = 99.8%. The only variable was the first prompt text, and because injected context was placed *after* the user prompt, ~2k tokens of byte-stable context could never participate in a cross-session hit. [B] https://github.com/deepseek-ai/deepseek-harness/discussions/4749

The same lesson from a different harness: a 193.6 KB system prompt + 114.4 KB of tool definitions (139 tools) sits at position 0, so any change is a full miss; one semantic-inject block that appeared on one prompt and vanished on the next changed the prompt at byte offset 40,224 and cost two misses per session until it was moved out of the system prompt and into a message. [B] https://erfi.dev/reference/agent-context-and-prompt-cache/

**Actionable reading:** the system prompt must be byte-stable across a session; volatile state (timestamps, memory index, skill list, task status) belongs *after* the stable prefix, ideally as a mid-conversation system message or a trailing user-visible block. Tool order must be centrally fixed, not plugin-registration order.

---

## 2. Tool calling: failure modes and what people ship

### 2.1 `tool_choice` is restricted in thinking mode

`deepseek-v4-pro` and `deepseek-v4-flash` return HTTP 400 for `tool_choice: "required"` and for a named function choice: `{"error":{"message":"Thinking mode does not support this tool_choice"}}`. Only omitted/`auto`/`none` work, and this breaks `with_structured_output` in LangChain and the equivalent in AutoGen/CrewAI. [B, open issue since 2026-05-30] https://github.com/deepseek-ai/DeepSeek-V3/issues/1376

A second report shows the inverse trap: to force a named tool you must disable thinking, and with thinking disabled plus a large prompt, **enum values in tool arguments arrive unquoted ~40–60% of the time** in bad windows (`"resulting_emotion_b": Tense`), with 250–350 output tokens billed on the malformed response. Short prompts and small schemas stayed clean; temperature and forced/auto made no difference. [B] https://github.com/deepseek-ai/DeepSeek-V3/issues/1541

Implication for us: a router cannot rely on `tool_choice` to force a tool on V4.x. It has to express routing in the prompt/tool description, or validate and retry on the harness side.

### 2.2 Truncation, streaming corruption, and DSML leakage

- Streaming tool arguments can arrive with a run of bytes missing from the middle under the `deepseekv32`/`deepseekv4` parser. [B, fix PR] https://github.com/sgl-project/sglang/pull/36339
- V4.1's leading-space DSML tags break V4 parsers and suppress streaming of argument deltas. [B] https://github.com/vllm-project/vllm/issues/58640
- Since 2026-09-22, one provider saw V4.1-Flash stop returning `tool_calls` and instead emit DSML markup as plain text in `content`. [B] https://github.com/CommandCodeAI/command-code/issues/909
- Shipped recovery: parse complete DSML `invoke` blocks out of assistant text into synthetic tool calls. https://github.com/openclaw/openclaw/pull/86032 · https://github.com/CherryHQ/cherry-studio/pull/14747
- Markdown-fenced argument JSON is rejected by a bare `JSON.parse`; a pre-parse fence stripper is the fix. [B] https://github.com/deepseek-ai/deepseek-harness/discussions/3047

### 2.3 A malformed argument kills the turn, and sometimes the session

DSH's default retry policy retries `EMPTY_RESPONSE, RATE_LIMIT, SERVER, TIMEOUT, TRANSPORT` — **not** `MALFORMED_RESPONSE` and not `STREAM_CLOSED`. A production audit of 961 session logs found 43 sessions with `tool input is invalid JSON`, 220 occurrences, and **46 turns killed**, with zero retry events. [B] https://github.com/deepseek-ai/deepseek-harness/discussions/8976

Worse, if the malformed argument string is persisted, it is replayed verbatim on every later turn and poisons the session: servers that re-parse the chat template reject every subsequent request (vLLM `Expecting delimiter`, llama.cpp `parse_error.101`). The tested fix is to `JSON.parse` at serialization and substitute `"{}"` when unparsable. [B] https://github.com/deepseek-ai/deepseek-harness/discussions/1519

### 2.4 What a production tool loop should require before executing

From the live API test run (30 HTTP requests, `max_tokens` ≤96, 9,058 tokens total): [B] https://github.com/chatdeepai/deepseek-api-reproducible-tests/blob/main/tool-calls/LIVE_RUN.md

1. a complete response or completed stream;
2. the expected finish condition (`stop`, not `length`);
3. a recognized allowlisted tool name;
4. unique call IDs;
5. parseable argument JSON;
6. schema-valid arguments;
7. authorization for the specific local action.

That is the harness-side workaround list the field converged on: **validate, retry, and never replay a poisoned argument string**.

### 2.5 Tool count is a cost and accuracy lever

- DSH exposes 29 tools; pi ships 4. DSH's measured per-turn floor is 8,188 tokens vs ~2,500 for Hermes and ~4,500 for Claude Code, and startup-tax × turns predicts tokens/task with R² = 0.99 across two models. [B] https://github.com/deepseek-ai/deepseek-harness/discussions/8938
- Shorter, adaptive tool shortlists improve selection: 93.1% vs 87.1% when always shown 5 tools, widening to 76.8% vs 60.9% on medium-difficulty queries; on BFCL a learned policy nearly matched showing 50 tools while presenting 7 on average. [B] https://arxiv.org/html/2605.24660v2
- A cache-specific trick: anchor the **first request** to a minimal tool catalog (shell + read), then restore the full catalog after the session records its first durable tool call. https://github.com/dbydd/pi-anchored-tool-for-dspro

---

## 3. Prompt structure: order, rule count, and what degrades

### 3.1 Rule count collapses adherence at ~80

Controlled experiment, 960 calls/model, five models, four formats, system vs user placement, N from 10 to 160 simultaneous rules: **perfect-response rate collapses to zero by N = 80 for every model, every format, both placements**. No model showed a reliable markdown advantage; one 35B model reliably preferred plain text. Placement effects were at least as large as format at N=160 but model-specific (helps two, hurts two, no effect for one). [B] https://arxiv.org/html/2607.19257v1

A separate stress test found the same shape for guardrails: performance approaches zero as the number of guardrails grows, without any conflicting instructions. [B] https://arxiv.org/html/2502.12197v1

This is the strongest argument against a "big cognitive gate prompt": past a few dozen rules, additional rules buy nothing and cost adherence on the ones that matter.

### 3.2 Stable prefix first, volatile facts last

DSH's system-prompt package assembles **one ordered system prompt** from sections plus runtime facts plus tool schemas, with explicit per-section `order` and a central `toolOrder` list — because registration order is an accident of plugin loading, and the measurements above show reordering two tools costs the entire cache. [A] https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/core/system-prompt/README.md · [B] https://inferloop.dev/deepseek-harness/prefix-stability/

The architectural note behind it: "every fact in the prompt has exactly one owner" — model name/workspace are config/session facts exposed as variables; per-tool semantics live in the tool description, not duplicated in prose. https://github.com/deepseek-ai/deepseek-harness/blob/master/.agents/notes/implemented/architecture/2026-07-05-prompt-variables-and-tool-guidance-ownership.md

A practical ordering recipe from the cache literature: `stable_first` — put the byte-stable part first and the volatile part last; a layout that defeats the prefix cache makes every turn pay for the whole prompt. [B] https://www.profrod.ai/book/ch21-b-context-repair-transfer-exercise

### 3.3 Skills shadow each other as the library grows

Scaling a skill library from a small helpful set to **202 skills drops pass rate by up to 21%**. Decomposition shows the cause is **skill shadowing** — the agent selects wrong skills more often — while "context overhead" (bigger context even when selection is right) is small and indistinguishable from zero. [B] https://arxiv.org/html/2605.24050

Two harness details that make it worse: DSH's model-facing catalog renders only `name` + `description`, dropping the `whenToUse` routing hint, and skill bodies load only via a manual `skill()` call. [B] https://github.com/deepseek-ai/deepseek-harness/discussions/8909

A shipped counter-design: a **rule-first pre-step router** that matches the latest user message against user-editable YAML, pours matched skill bodies at most once per session, stays silent on no match, and costs zero tokens until a rule fires. [B, implementation] https://github.com/MJorgin/dsh-skill-router

For us the implication is blunt: injecting "3–5 skills, always including brainstorming and writing-for-agents" on every session is a fixed selection prior that can shadow better-matching skills, and it has no measured benefit.

### 3.4 "Rules in the system prompt had zero effect"

The memory case is the cleanest natural experiment: a dedicated system-prompt section with three explicit rules (don't research what wasn't asked, don't save what can be found again, don't save your own intermediate ideas/log) produced **"Zero effect. Not less — the same."** [B] https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/discussions/46

What worked instead, across the sources: move the constraint from prose to a **mechanism** — a write-side gate that rejects the bad artifact, a schema that cannot represent it, a deterministic pre-step check, or an independent verifier. Same pattern in three places: memory hygiene (write gate), skill routing (rule-first router), verification (deterministic gate outside the model).

---

## 4. Context management: pre-compaction decay, post-compaction amnesia

### 4.1 Compaction silently drops standing constraints

ConstraintRot, seven models, 1,323 episodes, deterministic violation grading: compaction raises violation from **0% to 30% (up to 59%)**. When the constraint survives the summary, violation is 0%; when it is dropped, 38%. Decay is **8.3× larger for soft organizational policies** than for hard safety norms. A training-free defense — **Constraint Pinning**, re-inserting the constraint after compaction — restores violation to 0% for **~47 pinned tokens (<0.5% of a production-scale compaction context)**. [B] https://arxiv.org/abs/2606.22528

The mechanism is not model-specific: the compaction call optimizes for task continuity and treats standing policy as low-salience. Two harness bugs make it worse — the compaction LLM call receives `system: []` (no project instructions at all), and instruction files are not re-read after compaction. [B] https://github.com/anomalyco/opencode/pull/16959 · https://github.com/anthropics/claude-code/issues/24572

### 4.2 Re-anchoring, with a canary

A production site reports its own session was **summarised 18 times in a month, each summary keeping under 2% of what it replaced**. Their fix: keep the rules in a behaviour-policy file and have the harness **print it back after every summary**, plus a short status canary computed from the transcript that ends every few answers — if the canary stops appearing, a rule was not read. [B, single-site production report] https://sgit.ai/articles/re-anchoring-agent-behaviour-policies.md

### 4.3 Check the handoff: anchor-based premise guard

A shipped DSH plugin makes the compaction summary a *checked* handoff: after `compaction/summary`, it re-derives the shadowed span's text from the append-only log, extracts distinctive literal anchors (file paths, quoted literals, `key=value` pairs, error codes), checks whether the committed summary still contains them, and if a critical anchor vanished injects a one-shot notice naming what was lost and how to recover it. Deterministic, no LLM calls, `maxAnchors` default 5, `minAnchorLength` 6, notice capped at 400 chars. [B, implementation] https://github.com/ICCuse/dsh-premise-guard

This is the closest shipped analogue to our post-compaction misattribution failure (e): the model does not just lose facts, it loses **provenance** — who wrote the file it is now re-deriving.

### 4.4 Restore the action ledger, not just the facts

HF #39's third harness suggestion, written from the exact failure we see: *"After context compaction, restore a summary of the actions this session has already executed — in run 4 the model, after compacting, misattributed its own earlier edit to 'someone earlier in this session' and re-derived everything from disk."* [B] https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/discussions/39

A summary of *state* is not enough; the handoff needs a summary of *actions and authorship* (what was changed, by this session, at which path, with what value).

### 4.5 The channel matters: mid-conversation system messages

Because V4.1 supports a mid-conversation system message that behaves like a user message for generation-header purposes, constraint re-pinning does **not** require mutating the system prompt (which would wipe the prefix cache). [A] encoding README, above. Anthropic's context-engineering guidance reaches the same conclusion from the other direction: keep a stable prefix, curate the tail, and use structured note-taking rather than re-sending everything. [A] https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents

---

## 5. Stop conditions: bounding verification loops

### 5.1 What the field asks for, in the words of the report that measured the loop

HF #39's three suggestions are effectively a spec for a `StopRule`: [B] https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/discussions/39

1. An explicit **"a deliverable is a valid stopping point"** signal, so an agent delegating final acceptance to the user does not have to self-build an end-to-end proof first.
2. A **soft step/budget cap** that, once hit, forces the model to **state what is missing** instead of continuing.
3. After compaction, restore a summary of actions already executed.

### 5.2 Independent gates beat self-assessment

A pilot implementing verification as an architectural layer (five configurations, one executor/tool set/scorer, MuSiQue 4-hop, 8B backbone, 47 executions): **both unverified configurations answered 0 of 10 — every run ended at a step cap or token limit. The verified configuration answered 8 of 10 and abstained on the rest. Deterministic gates produced 8 of the 9 observed corrections at zero marginal cost.** Planning degraded performance once verification was present. [B, pilot scale] https://arxiv.org/html/2609.31937v1

The same paper's diagnosis applies directly to RevisionGate: a ReAct agent concentrates strategy, action selection, formatting, and "is this adequate?" in one model, and "an agent that cannot make progress does not report failure; it continues until an externally imposed budget terminates it."

### 5.3 Budgets must be enforced inside an attempt, and gates must not be self-attesting

"Bounded Loops" formalizes the harness guarantee: a worker, an **independent gate the worker cannot write to**, and a declared budget; termination in closed form; the ceiling enforced **inside** an attempt, with a reserve partitioned out so a hard stop leaves a written handoff. It then measures gates and finds **47 vacuous or self-attesting gates in a 69-loop catalogue of code that had shipped and passed review** — a gate that is satisfied by absence of the artifact, or whose subject supplies the value being checked. [B] https://arxiv.org/abs/2609.27871

That is the failure mode to avoid in our own `StopRule`/`RevisionGate`: if the model's own claim "I verified it" counts as the verdict, the gate is vacuous.

Industry harness practice matches: per-session iteration and budget caps enforced outside the agent. [B, vendor docs] https://docs.litellm.ai/docs/a2a_iteration_budgets

### 5.4 The loop is amplified by repeated tool calls

DSH's own repository documents that repeated tool calls have no hard cap, with an industry-measured **~4× amplification on failing turns**, and compaction fires only reactively above a threshold. [B] https://github.com/deepseek-ai/deepseek-harness/discussions/8938

### 5.5 A deterministic pipeline is the extreme version

D-Engine reports the same `formatDate` task on DeepSeek V4.1-Flash at 2,552 tokens vs dsh factory defaults 107,000 (42×) and 34,600 in minimal mode, all producing functionally the same code. It is the author's own harness benchmark, so treat the ratio as self-reported; the mechanism (no agent loop, no re-sent trajectory) is the point. [C, self-reported] https://dev.to/sergiocorruchaga/the-ai-thinks-the-gate-decides-how-i-made-llm-code-edits-deterministic-and-cut-token-usage-42x-5cbi

A preregistered benchmark reaches the compatible conclusion: prompt wording changes reasoning and verification behavior without changing the task; extra effort can add cost without benefit; and the value of an efficiency intervention changes when the harness changes. [B] https://arxiv.org/html/2608.01347

---

## 6. Memory formats: what keeps a model from writing chronicles

### 6.1 The measured contrast

Same harness, same config, same prompts, same 36/36 exam score:

| | V4 Flash | V4.1 Flash |
|---|---:|---:|
| Topic memory files | 9 | 11 |
| Total size | 13 KB (~4.6k tokens) | **161 KB (~42k tokens)** |
| Largest file | 5.5 KB | **70 KB** |
| Run time | 53–70 min | **99 min** |

The extra is not facts. V4 Flash kept "a table of current values with short notes; edit history lives in one line inside a cell: `$42→$55/m²`, `was 80 L`." V4.1 kept a chronicle: the same table plus a "Decisions log" retelling every step as a paragraph, "open questions", advice to itself, eight paragraphs on groupset generations for a two-word fact, **109 "TBC" placeholders for facts the user never gave**, and preambles like "why this file exists". Reproduced on a second agent: 21 KB → 350–500 KB, one hour → two to three, **788k output tokens instead of 141k**. [B] https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/discussions/46

The prompt-level fix failed explicitly: "Zero effect. Not less — the same." [B] same URL

### 6.2 What a memory format should look like

- **One row per entity/attribute, current value in the cell, history compressed into the cell** (`was 80 L`, `$42→$55/m²`). [B] HF #46, above.
- **A bounded, derived index, not the corpus.** `dsh-auto-memory` stores one Markdown file per memory with typed frontmatter and rebuilds a `MEMORY.md` index of **one line per memory**, injected each step; the index is bounded, stable within a day, and absent when the store is empty. [B, implementation] https://neoneye.github.io/agent-memory-atlas/systems/dsh-auto-memory/
- **A small persistent note file injected every turn, for preferences and conventions only.** [B] https://github.com/mcpassport/DeepSeek-TUI/blob/main/docs/MEMORY.md
- **Structured note-taking outside the context window**, retrieved on demand, rather than always loaded. [A] https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents

### 6.3 Write-side gating is the layer that works

Since prompt rules were inert, the shipped pattern is a gate at the write end: classify each candidate (`fact` vs chronicle/decision-log/open-question/placeholder), admit only facts, budget the bytes, and compress rejected chronicles into a one-line value table. That is exactly what our `memory_hygiene.py` does; the HF thread is the evidence base for it. No source reports a controlled A/B of format-only changes, so the 13 KB vs 161 KB contrast is the best measured win available. [B]

---

## 7. Other measured evidence

### 7.1 Effort vs cost: the dial is front-loaded

- Technical-report curve: effort 25→100 = average Pass@1 67.1→76.3, DeepSWE 66.0→74.2, Terminal-Bench 82.4→90.6, at ~2.5× output tokens; **60–80 recovers most of the accuracy**. [B] https://pirouter.ai/blog/reasoning-effort-one-name-five-physics
- Independent 30-task edit-and-test suite on 1017: **thinking off 30/30 at $0.0042/solved task; default `high` 30/30 at $0.0066; `max` 30/30 at $0.0099**. Thinking off was 36% cheaper than default and 58% cheaper than max. Counter-test: on 13 single-shot reasoning tasks, thinking off scored **11/39** while every thinking level scored 39/39. [B] https://synthorai.io/blog/deepseek-v4-1-flash-1017-cost/
- Field anecdote: `xhigh`/`max` described as "a reasoning loop trap" on simple state checks; `low` the sweet spot, `off` for trivial checks. [C] https://www.reddit.com/r/DeepSeek/comments/1whgo3e/if_flash_v41_is_stuck_in_toolcall_loops_or/
- Effort values are not portable across providers: the same string is forwarded, converted to a token budget, honoured numerically, ignored, or rejected depending on vendor. [B] https://pirouter.ai/blog/reasoning-effort-one-name-five-physics

**Reading for us:** `ultra`/`xhigh` should not be the default for edit-shaped work. The evidence says effort is a task-difficulty dial, not a quality dial, and the biggest wins are at the *bottom* of the range for small deterministic edits.

### 7.2 Self-consistency / multi-sampling

**No V4.x-specific self-consistency measurement was found.** The closest published evidence is generic: difficulty-adaptive self-consistency cuts sampling cost versus fixed-size sampling, but still pays multiple full generations. [B, general] https://arxiv.org/html/2408.13457

Given the effort curve above (2.5× output tokens for +9 pts, front-loaded), multi-sampling at `max` is likely dominated by a single pass at 60–80 for most agent work; that is an inference, not a measurement. [S]

### 7.3 Sub-agent decomposition: fan-out costs latency, buys context isolation

A DeepSeek-native agent that publishes negative results reports: **fan-out does not buy speed** (33.0s → 96.7s and 70.2s → 97.5s measured); what it buys is **~30% fewer parent input tokens** and N independent context budgets. Delegation timing decides whether sub-agents pay: delegating **before** investigating was 1.30× faster; **after** investigating was 2.08× slower. Also: "reasoning tokens bill against `max_tokens`. These are properties of the API, not of dsc — they hold in any harness, dsh included." [B, single-team measurements] https://github.com/thevibeworks/deepseek-code

### 7.4 Spec-driven workflows: output leaner, input much heavier

One controlled experiment: a spec→TDD MCP layer cut **output tokens 21%** but added **+77% total tokens and +47% time** versus baseline; the author calls the cost-benefit "way off" and sets a ≤+10% overhead target. Single experiment, one codebase. [C] https://dev.to/dimwiddle/the-mcp-i-built-for-ai-agents-backfired-4c8j

The preregistered benchmark's framing is the better takeaway: efficiency is **cost per successful task**, and prompt/effort/harness are interacting factors, not independent knobs. [B] https://arxiv.org/html/2608.01347

### 7.5 Harness floor dominates token spend

Same model, 30 long-horizon multi-SaaS workflows, 240 runs, programmatic verifier: Hermes 192k tokens/task, Pi 559k, OpenCode 665k, Codex ~692k, Claude Code 742k, Prime 1.4M. The most expensive harness used 7.29× the cheapest. [B, third-party benchmark quoted in a DSH thread] https://github.com/deepseek-ai/deepseek-harness/discussions/8938

This is worth keeping in view: our plugin's gate adds prompt bytes every turn, and the measured lever with the largest coefficient is the per-turn floor, not per-call cleverness.

---

## 8. Evidence table (condensed)

| # | Claim | Number | Source | Grade |
|---|---|---|---|---|
| 1 | One-field edit takes 33–50 turns | 39/50/33/43 turns; 2,946,619 input tokens; 3% edit | https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/discussions/39 | B |
| 2 | Memory bloat on same task/prompt | 13 KB → 161 KB; 99 vs 53–70 min; 788k vs 141k output | https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/discussions/46 | B |
| 3 | Prompt rules for memory had no effect | "Zero effect. Not less — the same." | same as #2 | B |
| 4 | Compaction drops constraints | 0% → 30% (up to 59%); dropped 38% vs survived 0%; 8.3× soft policies | https://arxiv.org/abs/2606.22528 | B |
| 5 | Pinning restores compliance | ~47 tokens, <0.5% of context, violation → 0% | same as #4 | B |
| 6 | Rule-count cliff | zero perfect responses at N=80, all models/formats | https://arxiv.org/html/2607.19257v1 | B |
| 7 | Cache sensitivity | system change 0%; append message 98.7%; reorder tools 0% | https://inferloop.dev/deepseek-harness/prefix-stability/ | B |
| 8 | Cross-session cache loss from ordering | 42% vs 99.8% hit | https://github.com/deepseek-ai/deepseek-harness/discussions/4749 | B |
| 9 | Cache price ratio | $0.003 vs $0.15 /1M input (50×) | https://api-docs.deepseek.com/quick_start/pricing/ | A |
| 10 | `tool_choice` restricted | `required` and named → HTTP 400 in thinking mode | https://github.com/deepseek-ai/DeepSeek-V3/issues/1376 | B |
| 11 | Malformed enum JSON | 40–60% of long-prompt forced calls; 250–350 tokens billed | https://github.com/deepseek-ai/DeepSeek-V3/issues/1541 | B |
| 12 | Truncated tool JSON at low max_tokens | 16/25 HTTP 200 `finish_reason: length` | https://github.com/chatdeepai/deepseek-api-reproducible-tests/blob/main/tool-calls/LIVE_RUN.md | B |
| 13 | Malformed args kill turns | 46 dead turns in 961 sessions; 0 retries | https://github.com/deepseek-ai/deepseek-harness/discussions/8976 | B |
| 14 | Poisoned args persist | session unrecoverable until `"{}"` substitution | https://github.com/deepseek-ai/deepseek-harness/discussions/1519 | B |
| 15 | Effort curve | 25→100: 67.1→76.3 Pass@1, ~2.5× output; 60–80 recovers most | https://pirouter.ai/blog/reasoning-effort-one-name-five-physics | B |
| 16 | Thinking off on small edits | 30/30, $0.0042 vs $0.0066 vs $0.0099 | https://synthorai.io/blog/deepseek-v4-1-flash-1017-cost/ | B |
| 17 | Thinking off on single-shot reasoning | 11/39 vs 39/39 | same as #16 | B |
| 18 | Verification layer pilot | unverified 0/10 hit cap; verified 8/10 + abstain | https://arxiv.org/html/2609.31937v1 | B |
| 19 | Vacuous/self-attesting gates | 47 of 69 shipped gates | https://arxiv.org/abs/2609.27871 | B |
| 20 | Skill shadowing | up to 21% pass-rate drop at 202 skills | https://arxiv.org/html/2605.24050 | B |
| 21 | Tool shortlist | 93.1% vs 87.1%; 76.8% vs 60.9% medium | https://arxiv.org/html/2605.24660v2 | B |
| 22 | Fan-out | 33.0→96.7s, 70.2→97.5s; ~30% fewer parent input tokens | https://github.com/thevibeworks/deepseek-code | B |
| 23 | Harness floor | 8,188 tokens/turn (DSH) vs ~2,500 (Hermes); 29 vs 4 tools | https://github.com/deepseek-ai/deepseek-harness/discussions/8938 | B |
| 24 | Spec-driven overhead | +77% tokens, +47% time, −21% output | https://dev.to/dimwiddle/the-mcp-i-built-for-ai-agents-backfired-4c8j | C |
| 25 | Deterministic pipeline | 2,552 vs 107,000 tokens (42×), self-reported | https://dev.to/sergiocorruchaga/the-ai-thinks-the-gate-decides-how-i-made-llm-code-edits-deterministic-and-cut-token-usage-42x-5cbi | C |
| 26 | Effort loop anecdote | "xhigh/max is a reasoning loop trap" | https://www.reddit.com/r/DeepSeek/comments/1whgo3e/if_flash_v41_is_stuck_in_toolcall_loops_or/ | C |

---

## 9. Mapping onto our modules

Status: **covered** = the module already implements the mechanism the evidence supports; **partial** = right layer, missing a specific lever; **missing** = no mechanism today.

| Module | What the evidence says | Status | Concrete gap |
|---|---|---|---|
| `gate.py` | Rules decay to zero by ~80; every system-prompt mutation wipes the prefix cache; V4.1 supports mid-conversation system messages; ~47 pinned tokens restores 0% compliance after compaction. [2607.19257, 2606.22528, encoding README] | **partial** | Gate injects full MAP + L1/L2 + primitives on turn 1 and a short form later, but the rule count is unbounded, it re-injects without a pinned-token budget, and it does not use the mid-conversation system-message channel. Add a `PINNED` block (≤50 tokens) re-pinned after compaction and every N turns, and a rule-count ceiling. |
| `intent_router.py` | Routing accuracy degrades as the candidate set grows; model-side selection is the dominant failure (skill shadowing), not context size; `tool_choice` cannot be used to force a tool on V4.x. [2605.24050, #1376] | **partial** | Keyword/CJK classifier is deterministic (good), but there is no confidence threshold that stays silent, no labelled accuracy set, and no explicit prohibition on emitting `tool_choice`-forced routes. Add a gold set + "no match → no injection" and express routes as prompt hints, never `tool_choice`. |
| `thinking_budget.py` | Effort is front-loaded (60–80 ≈ max); thinking off wins on small edits (30/30, 36% cheaper) and loses badly on single-shot reasoning (11/39); ultra/xhigh are not on the wire; reasoning bills as output against `max_tokens`. [pirouter, synthorai, thinking_mode docs] | **partial** | Tier mapping and the ULTRA→max wire mapping are correct. Missing: a *downward* route — `off`/`low` for single-field edits and state checks — and any measured yield-floor calibration. The yield floor is currently heuristic with no published validation. |
| `discipline.py` | Deliverable-as-stopping-point + "state what is missing" at a soft cap is the top requested fix; unverified loops hit caps 10/10 while verified ones answer 8/10; gates that accept the worker's own claim are vacuous (47/69). [HF #39, 2609.31937, 2609.27871] | **partial** | RevisionGate/StopRule exist, but StopRule has no "deliverable reached → stop, name residual risk" template, no budget cap that forces a handoff message, and no external ledger the model cannot write. Add an append-only verdict record and an explicit `STOP_HANDOFF` output contract. |
| `memory_hygiene.py` | 12× bloat is chronicle/decision-log/TBC, not facts; prompt rules had zero effect; the winning format is a current-value table with history compressed into the cell. [HF #46] | **covered** (mechanism), **unmeasured** | Write-side `classify`/`admit`/`compress_chronicle` is exactly the layer the evidence supports. Missing: update-in-place/dedupe by entity+attribute, a hard per-run byte budget assertion, and an A/B against the 13 KB baseline to prove the gate holds under V4.1. |
| `skill_router.py` | 202 skills → up to 21% pass-rate drop from wrong selection; catalogs hide `whenToUse`; rule-first routers are zero-token and one-pour-per-session. [2605.24050, #8909, dsh-skill-router] | **partial** | Picking 3–5 skills, always including brainstorming + writing-for-agents, is a fixed prior that can shadow better matches. Missing: confidence gate, `whenToUse` in the match key, one-pour-per-session, and a measured hit-rate/precision log. Injection into the user message is cache-safe (after the system+tools prefix) — keep it there. |
| `reasoning_effort.py` | API accepts `reasoning_effort` directly (low/high/max); text injection cannot change the wire; provider translations differ. [thinking_mode docs, pirouter] | **missing** (wire) | The module documents that Hermes's `pre_llm_call` hook can only inject text. That is a real limit, but it means effort is currently advisory. Move the setting into provider config or a request-level hook; otherwise delete the tier table's claim to control effort and keep only the protocol string. |

**Row sources.** `gate.py`: https://arxiv.org/html/2607.19257v1 · https://arxiv.org/abs/2606.22528 · https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blame/main/encoding/README.md · `intent_router.py`: https://arxiv.org/html/2605.24050 · https://github.com/deepseek-ai/DeepSeek-V3/issues/1376 · `thinking_budget.py`: https://pirouter.ai/blog/reasoning-effort-one-name-five-physics · https://synthorai.io/blog/deepseek-v4-1-flash-1017-cost/ · https://api-docs.deepseek.com/guides/thinking_mode/ · `discipline.py`: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/discussions/39 · https://arxiv.org/html/2609.31937v1 · https://arxiv.org/abs/2609.27871 · `memory_hygiene.py`: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/discussions/46 · `skill_router.py`: https://arxiv.org/html/2605.24050 · https://github.com/deepseek-ai/deepseek-harness/discussions/8909 · https://github.com/MJorgin/dsh-skill-router · `reasoning_effort.py`: https://api-docs.deepseek.com/guides/thinking_mode/ · https://pirouter.ai/blog/reasoning-effort-one-name-five-physics

**Cross-cutting:** none of the seven modules currently emits a machine-readable trace of its own decisions (gate fired? constraint pinned? stop rule triggered?). The Bounded Loops result — 47 of 69 shipped gates vacuous or self-attesting — is the argument for adding one before adding more rules. https://arxiv.org/abs/2609.27871

---

## 10. TOP 8 changes, ranked by value per unit of effort

Effort: **S** ≤1 day, **M** 1–3 days, **L** >3 days. Effect is stated as the measured quantity where one exists, otherwise the mechanism.

| # | Change | Module | Effort | Expected effect |
|---|---|---|---|---|
| 1 | **StopRule v2: deliverable-as-stopping-point + forced handoff.** When an artifact exists and the residual risk is nameable, stop. At the soft cap, emit `STOP_HANDOFF: done=<artifact>; unverified=<list>; next=<one action>` instead of continuing. Reject "verify by doing more work" as a valid step. | `discipline.py` | **S** | Directly targets the measured 20–40 verification turns / 1.25M re-verifying tokens (2.95M total for a 3% edit). HF #39 asks for this first; the verification pilot shows unverified loops terminate only at caps (0/10). |
| 2 | **Effort routing downward: `off`/`low` for single-file edits and state checks, `high` for multi-file, `max` only for architecture/research.** Make the default for "one field / one function" be thinking off. | `thinking_budget.py` + `reasoning_effort.py` | **S** | 30/30 small tasks at $0.0042 vs $0.0066 (default) and $0.0099 (max) — 36–58% cheaper with no accuracy loss; effort 60–80 already captures most of max's accuracy. Guard: never `off` for single-shot reasoning (11/39). |
| 3 | **Constraint pinning via a ≤50-token mid-conversation system message**, re-pinned after compaction and every N turns (N≈6), containing only the 3–7 non-negotiables. | `gate.py` | **S** | ConstraintRot: violation 0% → 30–59% after compaction; pinning restores 0% for ~47 tokens (<0.5% of context). Uses V4.1's supported mid-conversation system channel, so the system-prompt prefix stays byte-stable. |
| 4 | **Post-compaction action ledger + premise guard.** After a compaction event, inject: files changed by this session (path, value, turn), decisions made, and a deterministic anchor diff (paths, literals, `key=value`, error codes) naming anything the summary dropped. | `discipline.py` + `gate.py` | **M** | Fixes failure (e): HF #39 run 4 misattributed its own edit to "someone earlier in this session" and re-derived from disk. Mirrors the shipped `dsh-premise-guard` design (`maxAnchors` 5, notice ≤400 chars). |
| 5 | **Rule diet: cap gate rules at 7, each machine-checkable; move the rest into tool descriptions/schemas.** Delete prose that only restates a tool contract. | `gate.py` | **S** | Perfect-response rate hits zero by 80 rules for every model/format; fewer rules also shrink the per-turn floor, the largest coefficient in tokens/task (R²=0.99 for floor × turns). |
| 6 | **Tool-call integrity path:** echo `reasoning_content` on tool turns; `JSON.parse`-validate replayed arguments and substitute `{}`; retry `MALFORMED_RESPONSE`/`STREAM_CLOSED`; require `finish_reason=stop`; enforce a `max_tokens` floor on tool turns; accept one tool call per turn with strict JSON. | `discipline.py` (tool layer) | **M** | Removes a class of dead turns: 46 turns killed in 961 sessions, 16/25 HTTP 200 truncated at `max_tokens: 96`, and permanently poisoned sessions from replayed malformed args. Also closes the "stripped reasoning" hypothesis for forgetting. |
| 7 | **Memory format contract + write gate:** one current-value row per entity/attribute, edit history compressed into the cell, no decision logs / open questions / TBC / preambles; bounded one-line index; update-in-place instead of append. | `memory_hygiene.py` | **M** | 161 KB → 13 KB and 99 → 53–70 min on the same 36/36 task, with 788k → 141k output tokens on the reproduction. Prompt rules alone measured zero effect, so this must stay a write-side gate. |
| 8 | **Cache-stable prompt assembly:** system prompt byte-identical for the session; volatile facts (time, memory index, skill list, plan state) after the user message or in a mid-conversation system message; central fixed tool order; one stable skill catalog. | `gate.py` + `skill_router.py` | **M** | 98.8% vs 0% hit rate depending on where a byte changes; 42% → 99.8% cross-session hit from ordering alone; cache-hit input is 50× cheaper ($0.003 vs $0.15/1M). A 2k-token misplaced context block can never hit. |

**Evidence anchors for the ranking.** Row 1: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/discussions/39 · https://arxiv.org/html/2609.31937v1 · Row 2: https://synthorai.io/blog/deepseek-v4-1-flash-1017-cost/ · https://pirouter.ai/blog/reasoning-effort-one-name-five-physics · Row 3: https://arxiv.org/abs/2606.22528 · https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blame/main/encoding/README.md · Row 4: https://github.com/ICCuse/dsh-premise-guard · https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/discussions/39 · Row 5: https://arxiv.org/html/2607.19257v1 · https://github.com/deepseek-ai/deepseek-harness/discussions/8938 · Row 6: https://github.com/deepseek-ai/deepseek-harness/discussions/8976 · https://github.com/deepseek-ai/deepseek-harness/discussions/1519 · Row 7: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/discussions/46 · Row 8: https://inferloop.dev/deepseek-harness/prefix-stability/ · https://github.com/deepseek-ai/deepseek-harness/discussions/4749 · https://api-docs.deepseek.com/quick_start/pricing/

**What I would implement first (in order): 1, 2, 3.** All three are S-sized, all three have a measured number behind them, and together they attack the two largest line items in the observed spend — verification turns and unnecessary thinking — plus the constraint decay that makes the model re-litigate settled requirements. Item 4 is the next one after those, because it is the only fix for post-compaction misattribution and it needs the ledger format that item 1's handoff already produces.

---

## 11. Limits, anecdotes, and things I did not find

- **Single-task evidence:** HF #39 is four runs of *one* task and says so ("reproducible observation, not a benchmark claim"). Treat 33–50 turns as a shape, not a universal constant. [B, self-limited] https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/discussions/39
- **Vendor-adjacent self-reports:** D-Engine's 42× and the spec-driven MCP's +77% are authors benchmarking their own tools. Marked [C]; do not use them as planning numbers. https://dev.to/sergiocorruchaga/the-ai-thinks-the-gate-decides-how-i-made-llm-code-edits-deterministic-and-cut-token-usage-42x-5cbi · https://dev.to/dimwiddle/the-mcp-i-built-for-ai-agents-backfired-4c8j
- **No V4.x self-consistency study found.** Statements about multi-sampling vs single pass on V4.x are inference from the effort curve, not measurement. Marked [S]. https://pirouter.ai/blog/reasoning-effort-one-name-five-physics
- **No controlled study of "long system prompt degrades V4.x more than other vendors" found.** The rule-count cliff (2607.19257) is cross-model and does not single out DeepSeek; the V4.x-specific evidence is the field report that memory rules had zero effect. Do not overclaim vendor specificity. https://arxiv.org/html/2607.19257v1 · https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/discussions/46
- **Cache geometry is provider-internal.** The inferloop author states the exact assembly order is internal and only the results are observable; the byte-exact mechanism is general, the percentages will move with provider and model. [B, self-limited] https://inferloop.dev/deepseek-harness/prefix-stability/
- **Post-compaction provenance:** the only direct V4.x report is HF #39 run 4; the general mechanism is supported by Governance Decay and by the shipped premise-guard plugin. The combination is strong but not a V4.1-specific benchmark.

---

## Bibliography

1. DeepSeek, *Thinking Mode* — https://api-docs.deepseek.com/guides/thinking_mode/
2. DeepSeek, *Context Caching* — https://api-docs.deepseek.com/guides/kv_cache
3. DeepSeek, *The Temperature Parameter* — https://api-docs.deepseek.com/quick_start/parameter_settings
4. DeepSeek, *Models & Pricing* — https://api-docs.deepseek.com/quick_start/pricing/
5. DeepSeek, *Your First API Call* — https://api-docs.deepseek.com/
6. DeepSeek, *Integrate with Hermes Agent* — https://api-docs.deepseek.com/quick_start/agent_integrations/hermes
7. DeepSeek V4.1-Flash, *encoding/README.md* (prompt-format reference) — https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blame/main/encoding/README.md
8. ROCm ATOM, *DeepSeek-V4.1 chat, tools and reasoning* — https://rocm.docs.amd.com/projects/atom/en/latest/deepseek_v41_protocol.html
9. HF discussion #39, *33–50 agent turns for a one-field config edit* — https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/discussions/39
10. HF discussion #46, *V4.1 Flash manages agent memory 10× less efficiently* — https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/discussions/46
11. DeepSeek-V3 issue #1376, *V4 rejects tool_choice="required" and named function* — https://github.com/deepseek-ai/DeepSeek-V3/issues/1376
12. DeepSeek-V3 issue #1541, *unquoted enum literals in forced tool_call arguments* — https://github.com/deepseek-ai/DeepSeek-V3/issues/1541
13. chatdeepai, *DeepSeek Tool Calls Live Run* — https://github.com/chatdeepai/deepseek-api-reproducible-tests/blob/main/tool-calls/LIVE_RUN.md
14. DSH discussion #8976, *One malformed tool-call argument kills the whole turn* — https://github.com/deepseek-ai/deepseek-harness/discussions/8976
15. DSH discussion #1519, *Truncated tool-call arguments permanently poison a session* — https://github.com/deepseek-ai/deepseek-harness/discussions/1519
16. DSH discussion #3047, *markdown-fenced tool-call arguments rejected* — https://github.com/deepseek-ai/deepseek-harness/discussions/3047
17. DSH discussion #4749, *Pre-step batch forfeits cross-session prefix cache (42% vs 99.8%)* — https://github.com/deepseek-ai/deepseek-harness/discussions/4749
18. DSH discussion #8909, *Include whenToUse in the skill catalog* — https://github.com/deepseek-ai/deepseek-harness/discussions/8909
19. DSH discussion #8938, *DSH burns more tokens on its own V4.1 Flash* — https://github.com/deepseek-ai/deepseek-harness/discussions/8938
20. DSH, *compaction subsystem* — https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/compaction.md
21. DSH, *system-prompt package* — https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/core/system-prompt/README.md
22. DSH, *prompt variables and tool-guidance ownership* — https://github.com/deepseek-ai/deepseek-harness/blob/master/.agents/notes/implemented/architecture/2026-07-05-prompt-variables-and-tool-guidance-ownership.md
23. sglang PR #36339, *Fix DeepSeekV32Detector streaming corrupting tool-call arguments* — https://github.com/sgl-project/sglang/pull/36339
24. vLLM issue #58640, *leading extra space in V4.1 DSML tags* — https://github.com/vllm-project/vllm/issues/58640
25. CommandCode issue #909, *deepseek-v4.1-flash tool-calling regression* — https://github.com/CommandCodeAI/command-code/issues/909
26. OpenClaw PR #86032, *recover DeepSeek DSML tool markup* — https://github.com/openclaw/openclaw/pull/86032
27. Cherry Studio PR #14747, *parse DeepSeek DSML tool-call markup* — https://github.com/CherryHQ/cherry-studio/pull/14747
28. dsh-plugin.org, *invalid JSON, bash timeout, snapshot loss* — https://dsh-plugin.org/tutorials/fix-tool-call-hang-and-invalid-output
29. inferloop, *把两个工具换个位置，作废九成缓存* (measured prefix-cache variants) — https://inferloop.dev/deepseek-harness/prefix-stability/
30. Erfi's Lexicanum, *Context, prompt cache, and compaction in a coding-agent harness* — https://erfi.dev/reference/agent-context-and-prompt-cache/
31. arXiv 2607.19257, *Prompt Design at Scale* — https://arxiv.org/html/2607.19257v1
32. arXiv 2502.12197, *A Closer Look at System Prompt Robustness* — https://arxiv.org/html/2502.12197v1
33. arXiv 2606.22528, *Governance Decay: How Context Compaction Silently Erases Safety Constraints* — https://arxiv.org/abs/2606.22528
34. arXiv 2609.27871, *Bounded Loops* — https://arxiv.org/abs/2609.27871
35. arXiv 2609.31937, *Verification as an Architectural Layer for LLM Agents* — https://arxiv.org/html/2609.31937v1
36. arXiv 2608.01347, *Prompt-Induced Waste in Coding Agents* — https://arxiv.org/html/2608.01347
37. arXiv 2607.01641, *Uncovering Infinite Agentic Loops in LLM Agents* — https://arxiv.org/abs/2607.01641
38. arXiv 2605.24050, *More Skills, Worse Agents? Skill Shadowing* — https://arxiv.org/html/2605.24050
39. arXiv 2605.24660, *How Many Tools Should an LLM Agent See?* — https://arxiv.org/html/2605.24660v2
40. arXiv 2408.13457, *Difficulty-Adaptive Self-Consistency* — https://arxiv.org/html/2408.13457
41. Anthropic, *Effective context engineering for AI agents* — https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
42. Anthropic, *Effective harnesses for long-running agents* — https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents
43. sgit.ai, *Re-anchoring: keeping an agent's rules alive through every summary* — https://sgit.ai/articles/re-anchoring-agent-behaviour-policies.md
44. ConstraintGuard — https://github.com/abhid1234/constraintguard
45. dsh-premise-guard — https://github.com/ICCuse/dsh-premise-guard
46. dsh-skill-router — https://github.com/MJorgin/dsh-skill-router
47. dsh-auto-memory (Agent Memory Atlas) — https://neoneye.github.io/agent-memory-atlas/systems/dsh-auto-memory/
48. DeepSeek-TUI memory note — https://github.com/mcpassport/DeepSeek-TUI/blob/main/docs/MEMORY.md
49. pi-anchored-tool-for-dspro — https://github.com/dbydd/pi-anchored-tool-for-dspro
50. thevibeworks/deepseek-code (measured findings, incl. negative) — https://github.com/thevibeworks/deepseek-code
51. Synthorai, *V4.1 Flash 1017: Thinking Off Fixed 30/30 Small Coding Tasks* — https://synthorai.io/blog/deepseek-v4-1-flash-1017-cost/
52. PiRouter, *What reasoning_effort actually does* — https://pirouter.ai/blog/reasoning-effort-one-name-five-physics
53. DeepSeekAgent.io, *V4.1 Flash vs V4 Flash in Harness* — https://deepseekagent.io/guides/deepseek-v4-1-flash-vs-v4-flash-harness
54. D-Engine, *The AI thinks, the gate decides* — https://dev.to/sergiocorruchaga/the-ai-thinks-the-gate-decides-how-i-made-llm-code-edits-deterministic-and-cut-token-usage-42x-5cbi
55. SpecLeft, *The MCP I built for AI agents backfired* — https://dev.to/dimwiddle/the-mcp-i-built-for-ai-agents-backfired-4c8j
56. r/DeepSeek, *If Flash V4.1 is stuck in tool-call loops, try lowering your Reasoning Effort* — https://www.reddit.com/r/DeepSeek/comments/1whgo3e/if_flash_v41_is_stuck_in_toolcall_loops_or/
57. r/DeepSeek, *v4.1 flash keeps forgetting stuff* — https://www.reddit.com/r/DeepSeek/comments/1wlssvo/v41_flash_keeps_forgetting_stuff/
58. Prof Rod, *Context Engineering Unit B* (pinned records, `stable_first`, measured prefill) — https://www.profrod.ai/book/ch21-b-context-repair-transfer-exercise
59. OpenCode PR #16959, *preserve instructions across compaction* — https://github.com/anomalyco/opencode/pull/16959
60. Claude Code issue #24572, *Re-read CLAUDE.md after context compaction* — https://github.com/anthropics/claude-code/issues/24572
61. LiteLLM, *Agent Iteration Budgets* — https://docs.litellm.ai/docs/a2a_iteration_budgets
62. Roo Code DeepSeek provider (reasoning_content echo comment) — https://github.com/RooCodeInc/Roo-Code/blob/137d3f4f/packages/types/src/providers/deepseek.ts
63. OmniRoute, *Reasoning replay* — https://github.com/diegosouzapw/OmniRoute/blob/main/docs/routing/REASONING_REPLAY.md
64. DeepSeek-Reasonix PR #4022, *round-trip reasoning_content on tool_calls turns* — https://github.com/esengine/DeepSeek-Reasonix/pull/4022
