"""Superpowers skill routing: session bootstrap + per-child skill contracts.

Two jobs, both mechanical and filesystem-only:

* ``on_pre_llm_call`` picks 3-5 installed skills for the first real prompt of a
  session: a fixed trio (brainstorming, writing-for-agents, using-superpowers),
  verification-before-completion when present, then intent, description-scored
  and keyword matches. The short ``[superpowers-skills]`` load list is injected
  into the user message on every first turn, trivial asks included.
* ``on_pre_tool_call`` rewrites every ``delegate_task`` invocation so each child
  receives its own ``[superpowers-skills: dsh]`` contract in ``context`` — this
  is the only path that reaches a subagent, because the ``subagent_start``
  hook's return value is discarded by the host.

The host merges a ``{"action": "modify", "args": {...}}`` pre_tool_call result
into the original args, so only changed keys are returned.

No network, no sleeps, no heavy imports: stdlib + filesystem reads only.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

MARKER = "[superpowers-skills: dsh]"

#: Fixed start-of-session trio, in order, whenever each skill is installed.
#: The prompt never decides whether these appear.
ALWAYS: Tuple[str, ...] = ("brainstorming", "writing-for-agents", "using-superpowers")

#: Fourth always-on skill when it resolves.
VERIFY_ALWAYS = "verification-before-completion"

#: intent_router intents -> skills that carry that kind of work.
INTENT_SKILLS: Dict[str, Tuple[str, ...]] = {
    "architecture": ("writing-plans", "subagent-driven-development"),
    "research": ("writing-plans",),
    "collaboration": ("dispatching-parallel-agents", "subagent-driven-development"),
    "refactor": ("test-driven-development", "verification-before-completion"),
    "new": ("test-driven-development", "brainstorming"),
    "medium": ("test-driven-development",),
    "simple": (),
    "spec_driven": ("writing-plans",),
}

#: (keywords, skill): a first-turn prompt containing a keyword pulls the skill.
KEYWORD_SKILLS: Tuple[Tuple[Tuple[str, ...], str], ...] = (
    (("bug", "broken", "fails", "error", "traceback", "crash"), "systematic-debugging"),
    (("review", "audit", "critique"), "requesting-code-review"),
    (("parallel", "fan out", "subagents", "children"), "dispatching-parallel-agents"),
    (("merge", "branch", "commit", "push"), "finishing-a-development-branch"),
    (("verify", "proof", "evidence", "prove"), "verification-before-completion"),
    (("plan", "roadmap", "milestone"), "writing-plans"),
    (("skill", "SKILL.md", "frontmatter"), "writing-skills"),
)

#: Small stopword set for description scoring (never exhaustive).
_STOPWORDS = frozenset(
    """
    a an and any are as at be been being before after but by can do does for from
    had has have here how i if in into is it its just me more most must my no not
    of on only or our out per should so some such than that the their them then
    there these they this those to up use using via was we were what when which
    who why will with would you your
    """.split()
)
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+#._-]*")

_FRONTMATTER_RE = re.compile(r"^\ufeff?---\s*\r?\n(.*?)\r?\n---\s*(?:\r?\n|$)", re.DOTALL)
_NAME_RE = re.compile(r"^name\s*:\s*(.+?)\s*$", re.MULTILINE)
_DESC_RE = re.compile(r"^description\s*:\s*(.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class SkillInfo:
    """One discovered ``SKILL.md``."""

    name: str
    description: str
    path: str
    source: str  # "superpowers" | "user"


# ── discovery ───────────────────────────────────────────


def _norm(path: str) -> str:
    """Normalise to a native path string; accept both slash styles on input."""
    return os.path.normpath(os.path.expanduser(path))


def _has_skill_roots(home: str) -> bool:
    base = Path(home)
    try:
        if (base / "skills").is_dir():
            return True
        return any((base / "plugins").glob("*/skills"))
    except OSError:
        return False


def hermes_home() -> str:
    """Resolve the Hermes home used for skill discovery.

    ``HERMES_HOME`` wins when set. Otherwise ``~/.hermes`` is preferred only if
    it actually carries skills; a bare ``~/.hermes`` (plugins but no skills) must
    not shadow the Windows profile home under ``%LOCALAPPDATA%/hermes``.
    """
    explicit = os.environ.get("HERMES_HOME")
    if explicit:
        return _norm(explicit)

    candidates: List[str] = [_norm(os.path.join(os.path.expanduser("~"), ".hermes"))]
    local = os.environ.get("LOCALAPPDATA")
    if local:
        candidates.append(_norm(os.path.join(local, "hermes")))

    existing = [c for c in candidates if os.path.isdir(c)]
    for candidate in existing:
        if _has_skill_roots(candidate):
            return candidate
    if existing:
        return existing[0]
    return candidates[0]


def default_roots() -> List[str]:
    """Skill roots, most specific first: env override, plugin skills, user skills."""
    roots: List[str] = []

    override = os.environ.get("SUPERPOWERS_SKILLS_DIR")
    if override:
        for piece in override.split(os.pathsep):
            if piece.strip():
                roots.append(_norm(piece.strip()))

    home = Path(hermes_home())
    try:
        plugins = sorted((home / "plugins").glob("*/skills"))
    except OSError:
        plugins = []
    roots.extend(_norm(str(p)) for p in plugins if p.is_dir())

    user_skills = home / "skills"
    if user_skills.is_dir():
        roots.append(_norm(str(user_skills)))

    deduped: List[str] = []
    seen = set()
    for root in roots:
        key = os.path.normcase(root)
        if key not in seen:
            seen.add(key)
            deduped.append(root)
    return deduped


def _iter_skill_files(root: str) -> Iterable[Path]:
    base = Path(root)
    try:
        if not base.is_dir():
            return
    except OSError:
        return
    seen = set()
    for pattern in ("*/SKILL.md", "**/SKILL.md"):
        try:
            matches = sorted(base.glob(pattern))
        except OSError:
            continue
        for path in matches:
            key = os.path.normcase(os.path.normpath(str(path)))
            if key in seen:
                continue
            seen.add(key)
            yield path


def _source_for(path: Path) -> str:
    text = str(path).replace("\\", "/").lower()
    if "superpowers" in text or "/plugins/" in text:
        return "superpowers"
    return "user"


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1].strip()
    return value


def _parse_skill(path: Path) -> SkillInfo:
    text = path.read_text(encoding="utf-8", errors="replace")
    name = ""
    description = ""
    match = _FRONTMATTER_RE.search(text)
    if match:
        frontmatter = match.group(1)
        name_match = _NAME_RE.search(frontmatter)
        if name_match:
            name = _unquote(name_match.group(1))
        desc_match = _DESC_RE.search(frontmatter)
        if desc_match:
            description = _unquote(desc_match.group(1))
    if not name:
        name = path.parent.name
    return SkillInfo(
        name=name,
        description=description,
        path=str(path),
        source=_source_for(path),
    )


def discover(roots: Optional[List[str]] = None) -> Dict[str, SkillInfo]:
    """Read every ``SKILL.md`` under the given roots (or the default roots).

    First occurrence of a name wins. Unreadable or malformed files are skipped
    rather than raising; a missing frontmatter falls back to the directory name.
    """
    if roots is None:
        try:
            roots = default_roots()
        except Exception:  # pragma: no cover - defensive: discovery must not raise
            logger.debug("skill_router: default_roots failed", exc_info=True)
            roots = []

    found: Dict[str, SkillInfo] = {}
    for root in roots:
        for path in _iter_skill_files(str(root)):
            try:
                info = _parse_skill(path)
            except Exception:  # unreadable / undecodable file: skip it
                logger.debug("skill_router: skipping %s", path, exc_info=True)
                continue
            if info.name and info.name not in found:
                found[info.name] = info
    return found


# ── session choice ──────────────────────────────────────


def _tokens(text: str) -> set:
    """Lowercase word tokens with punctuation stripped and stopwords dropped."""
    return {
        token
        for token in _TOKEN_RE.findall((text or "").lower())
        if len(token) > 1 and token not in _STOPWORDS
    }


def score_skills(
    prompt: str,
    skills: Optional[Dict[str, SkillInfo]] = None,
) -> List[Dict[str, object]]:
    """Rank any installed skill by description overlap with the prompt.

    Overlap ratio = shared tokens / prompt tokens. Matches at or above 0.25 are
    returned highest score first, ties broken by name. Nothing here knows the
    names of the Superpowers skills: a newly installed skill can win by its own
    frontmatter description.
    """
    pool = discover() if skills is None else skills
    prompt_tokens = _tokens(prompt)
    if not prompt_tokens or not pool:
        return []

    matches: List[Dict[str, object]] = []
    for name, info in pool.items():
        description_tokens = _tokens(info.description)
        shared = prompt_tokens & description_tokens
        if not shared:
            continue
        score = len(shared) / len(prompt_tokens)
        if score >= 0.25:
            matches.append(
                {
                    "name": name,
                    "score": round(score, 3),
                    "why": f"description match: {', '.join(sorted(shared))}",
                }
            )
    matches.sort(key=lambda item: (-float(item["score"]), str(item["name"])))
    return matches


def choose(
    prompt: str,
    intent: str = "",
    *,
    minimum: int = 3,
    maximum: int = 5,
    skills: Optional[Dict[str, SkillInfo]] = None,
) -> List[Dict[str, str]]:
    """Pick 3-5 skills for the live prompt.

    Fixed order: the ALWAYS trio, verification-before-completion, intent
    matches, description matches, keyword matches. Deduped, capped at
    ``maximum`` and floored at ``minimum`` (never more than the pool holds).
    """
    pool = discover() if skills is None else skills
    prompt_text = (prompt or "").lower()
    intent_key = (intent or "").strip().lower()

    picked: List[Dict[str, str]] = []

    def add(name: str, why: str) -> None:
        if (
            name in pool
            and len(picked) < maximum
            and all(item["name"] != name for item in picked)
        ):
            picked.append({"name": name, "why": why})

    for name in ALWAYS:
        add(name, "always-on baseline")

    add(VERIFY_ALWAYS, "always-on verification")

    for name in INTENT_SKILLS.get(intent_key, ()):
        add(name, f"intent: {intent_key}")

    for match in score_skills(prompt_text, pool):
        add(str(match["name"]), str(match["why"]))

    for keywords, name in KEYWORD_SKILLS:
        hit = next((keyword for keyword in keywords if keyword in prompt_text), None)
        if hit:
            add(name, f"matched '{hit}'")

    floor = min(minimum, maximum, len(pool))
    for name in sorted(pool):
        if len(picked) >= floor:
            break
        add(name, "discovered pool floor")

    return picked[:maximum]


def bootstrap(
    prompt: str,
    intent: str = "",
    *,
    skills: Optional[Dict[str, SkillInfo]] = None,
) -> Optional[Dict[str, str]]:
    """Return the first-turn ``{"context": ...}`` payload, or None if empty."""
    chosen = choose(prompt, intent, skills=skills)
    if not chosen:
        return None
    names = ", ".join(item["name"] for item in chosen)
    return {
        "context": (
            f"[superpowers-skills] load before acting: {names}. "
            "Use skill_view(name=...). "
            "On every subagent, the plugin injects its own contract."
        )
    }


# ── per-child contracts ─────────────────────────────────

_DEBUG_RE = re.compile(
    r"\b(bug|bugs|broken|fails?|failing|error|errors|traceback|crash(?:es|ed)?|"
    r"exception|regression)\b"
)
_CODE_RE = re.compile(
    r"\b(add|adds|fix|fixes|implement|implements|refactor|refactors|write|writes|"
    r"build|builds|create|creates|change|changes|patch|patches|develop|develops)\b"
)
_REVIEW_RE = re.compile(r"\b(review|reviews|audit|audits|critique|critiques)\b")
_DOCS_RE = re.compile(
    r"\b(doc|docs|documentation|prose|readme|guide|article|write-up|writeup|"
    r"summary|blog)\b"
)


def _child_selection(
    goal: str,
    skills: Optional[Dict[str, SkillInfo]] = None,
) -> List[Dict[str, str]]:
    pool = discover() if skills is None else skills
    if not pool:
        return []
    text = (goal or "").lower()

    picked: List[Dict[str, str]] = []

    def add(name: str, why: str) -> None:
        if name in pool and len(picked) < 3 and all(item["name"] != name for item in picked):
            picked.append({"name": name, "why": why})

    docs_shaped = bool(_DOCS_RE.search(text))
    if _DEBUG_RE.search(text):
        add("systematic-debugging", "bug-shaped goal")
    if docs_shaped:
        add("writing-for-agents", "doc/prose goal")
    elif _CODE_RE.search(text):
        add("test-driven-development", "code change implied")
    if _REVIEW_RE.search(text):
        add("requesting-code-review", "review-shaped goal")
    if len(picked) < 3:
        add("verification-before-completion", "verification before done")
    return picked


def child_skills(goal: str, *, skills: Optional[Dict[str, SkillInfo]] = None) -> List[str]:
    """Up to three skills suited to a subagent's goal (stable, deduped)."""
    return [item["name"] for item in _child_selection(goal, skills)]


def child_contract(goal: str, *, skills: Optional[Dict[str, SkillInfo]] = None) -> str:
    """The ``[superpowers-skills: dsh]`` block injected into a child's context."""
    selected = _child_selection(goal, skills)
    if selected:
        skills_text = ", ".join(f"{item['name']} ({item['why']})" for item in selected)
    else:
        skills_text = "none discovered"
    return (
        f"{MARKER} Skills for this subtask: {skills_text}. "
        "Load each with skill_view(name=...) before acting. "
        "Write the failing test before the implementation. "
        "To verify completion, run a real command and quote its output; "
        "no done claim without it."
    )


def inject_delegate_args(
    tool_name: str,
    args: Optional[Dict[str, object]],
    *,
    skills: Optional[Dict[str, SkillInfo]] = None,
) -> Optional[Dict[str, object]]:
    """Build the pre_tool_call ``modify`` result for ``delegate_task``.

    Single mode rewrites ``context`` only; batch mode rewrites each task's
    ``context`` while preserving goal, output_schema and every other key.
    Already-marked contexts make this a no-op (idempotent). Discovery failure
    returns None so the tool call proceeds untouched.
    """
    if tool_name != "delegate_task" or not isinstance(args, dict):
        return None

    pool = skills
    if pool is None:
        try:
            pool = discover()
        except Exception:
            logger.debug("skill_router: discovery failed for delegate_task", exc_info=True)
            return None
    if not pool:
        return None

    tasks = args.get("tasks")
    if isinstance(tasks, list):
        rewritten: List[object] = []
        modified = False
        for task in tasks:
            if not isinstance(task, dict):
                rewritten.append(task)
                continue
            context = task.get("context") or ""
            if isinstance(context, str) and MARKER in context:
                rewritten.append(task)
                continue
            contract = child_contract(str(task.get("goal") or ""), skills=pool)
            updated = dict(task)
            updated["context"] = f"{contract}\n\n{context}" if context else contract
            rewritten.append(updated)
            modified = True
        if not modified:
            return None
        return {"action": "modify", "args": {"tasks": rewritten}}

    if "goal" not in args and "context" not in args:
        return None
    context = args.get("context") or ""
    if isinstance(context, str) and MARKER in context:
        return None
    contract = child_contract(str(args.get("goal") or ""), skills=pool)
    return {
        "action": "modify",
        "args": {
            "context": f"{contract}\n\n{context}" if context else contract,
        },
    }


# ── hooks ───────────────────────────────────────────────


def on_pre_llm_call(**kwargs):
    """First real turn only: inject the skill load list, classified by intent."""
    try:
        if not kwargs.get("is_first_turn", False):
            return None
        message = kwargs.get("user_message") or ""
        if not isinstance(message, str) or not message.strip():
            return None

        intent = ""
        try:
            from .intent_router import classify_intent

            result = classify_intent(message) or {}
            intent = str(result.get("intent") or "")
        except Exception:  # intent data unavailable: keywords still route
            logger.debug("skill_router: intent classification unavailable", exc_info=True)

        return bootstrap(message, intent)
    except Exception:  # pragma: no cover - hooks must never break a turn
        logger.debug("skill_router.on_pre_llm_call failed", exc_info=True)
        return None


def on_pre_tool_call(**kwargs):
    """Every tool call: rewrite delegate_task args so children carry a contract."""
    try:
        return inject_delegate_args(
            kwargs.get("tool_name", ""),
            kwargs.get("args") if isinstance(kwargs.get("args"), dict) else {},
        )
    except Exception:  # pragma: no cover - hooks must never break a turn
        logger.debug("skill_router.on_pre_tool_call failed", exc_info=True)
        return None
