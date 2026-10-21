# I-18 — ThinkingBudget: Ultra tier + a mechanical yield floor

Status: Experimental. Code `src/deepseek_harness/thinking_budget.py`; evidence
`tests/test_thinking_budget.py` and `tests/test_reasoning_effort.py`.

## The problem with "just one more tier"

I-17 already nudged the model toward deeper reasoning for architecture, research
and collaboration intents. The obvious next move — add `ultra` above `max` and
hope for more thinking — buys volume, not value. Longer traces mostly produce
restatements: the same insight rephrased, plans that never resolve into a step.
Volume is the failure mode, not the goal. I-18 pairs the tier with the thing
that actually bounds it: a **cut rule**.

## Tier ladder

`off < low < medium < high < xhigh < max < ultra`

`architecture` and `research` now plan `ultra`; `collaboration` plans `xhigh`;
`refactor`/`new`/`medium` plan `high`; `simple` and `spec_driven` plan
`medium` and inject nothing (unchanged I-17 behaviour, pinned by
`tests/test_reasoning_effort.py`).

## Wire honesty

The DeepSeek API accepts only `reasoning_effort` values `"max"`, `"high"`,
`"medium"`. There is no `xhigh` and no `ultra` on the wire, so:

| tier | wire value |
|---|---|
| ultra | `"max"` |
| xhigh | `"max"` |
| max   | `"max"` |
| high  | `"high"` |
| medium | `"medium"` |
| low / off | not sent |

`wire_effort()` encodes exactly that. The depth *above* max is carried by the
in-context protocol, never by a parameter the API does not have.
`_TIER_TOKEN_BUDGET` (`ultra: 24000`) is an advisory context budget, not a field.

## The cut rule (`ThinkingGovernor.review`)

For each thought in the round:

1. Normalise — lowercase, strip punctuation, token set.
2. **Duplicate?** Jaccard(tokens, tokens of any kept thought) >= 0.8 → cut,
   without discussion.
3. **Actionable?** the first 12 words contain an imperative verb (`add`,
   `extract`, `wire`, `fix`, `verify`, `measure`, …) **or** it carries a step
   marker (`->`, `step:`, `next:`, `todo:`). Otherwise cut.
4. Survivors become `kept` and `steps`.

Then the floor: `valuable >= floor` (default 2) → `CONTINUE`; below that →
`CONVERGE`, which means stop thinking and act on the best step or ask one
question. The floor is what makes "stop" a decision instead of a mood.

Real example — a one-word restatement dies at 0.83, a prose musing dies for
having no step, two distinct steps survive and keep the round CONTINUE:

```
1 Add a retry wrapper around the http client for transient failures   kept
2 Add a retry wrapper around the http client for transient errors     cut (0.83)
3 It might be interesting to consider the design more deeply          cut (no step)
```

## Hysteresis (`record`)

Counters, not clocks: two consecutive `CONVERGE` rounds demote the session one
tier; two consecutive `CONTINUE` rounds with `valuable >= floor + 1` promote one,
capped at `ultra`, floored at `off`. Break-even yield holds; a change of
direction resets both, so one noisy round never moves the tier.

## Why this beats a bigger badge

A tier sets how long the model may think; the yield floor says when thinking has
stopped paying, and the governor acts on that verdict for the rest of the
session — depth flows to rounds that produce steps, not prose.

Pure stdlib (`dataclasses`, `re`, `typing`): no I/O, no network, no clocks.
