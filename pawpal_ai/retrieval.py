"""Retrieval: find the passages an answer should be built from.

Three sources, searched before the model writes anything:

- household: the owner's real pets, tasks and today's plan (from pawpal_system)
- notes: the editable care notes in knowledge/*.md
- web: Wikipedia, free and keyless, fetched with a short timeout

Every passage gets a source id (S1, S2...) so the model can cite it and the
citation can be checked afterwards.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import Callable

from pawpal_system import Owner, Pet, Scheduler

log = logging.getLogger("pawpal_ai.retrieval")

KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge"

STOPWORDS = set(
    """a an the and or but if of to in on at for with by from as is are was were be been
    being it its this that these those my our your their his her i you we they he she me
    him them do does did doing have has had can could should would will shall may might
    must how what when where which who whom why much many more most some any all each
    every so than too very just about into over under after before up down out off again
    there here not no nor only own same such get got am s t don isnt dont doesnt im ive
    please tell know want need ok okay also like really still""".split()
)


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens with stopwords removed and plurals folded."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    tokens = []
    for word in words:
        if word in STOPWORDS or len(word) < 2:
            continue
        # A crude stemmer is enough here: "walks"/"walking" should find "walk".
        if word.endswith("ies") and len(word) > 4:
            tokens.append(word[:-3] + "y")  # huskies -> husky, puppies -> puppy
            continue
        for suffix in ("ing", "es", "s"):
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                word = word[: -len(suffix)]
                break
        tokens.append(word)
    return tokens


@dataclass
class Passage:
    """One retrieved piece of evidence."""

    origin: str  # "household" | "notes" | "web"
    title: str
    text: str
    score: float
    url: str | None = None
    source_id: str = ""

    def label(self) -> str:
        """Short human-readable description of where this came from."""
        if self.origin == "web":
            return f"Wikipedia: {self.title}"
        if self.origin == "notes":
            return f"Care notes: {self.title}"
        return f"Your household: {self.title}"


@dataclass
class RetrievalResult:
    """Passages from every source, plus any source that failed and why."""

    passages: list[Passage] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)

    def by_id(self) -> dict[str, Passage]:
        return {passage.source_id: passage for passage in self.passages}

    def origins(self) -> set[str]:
        return {passage.origin for passage in self.passages}


# --- Local notes -----------------------------------------------------------


class NotesRetriever:
    """Keyword (TF-IDF) search over the markdown care notes, one chunk per heading."""

    def __init__(self, folder: Path = KNOWLEDGE_DIR):
        self.folder = Path(folder)
        self.chunks: list[tuple[str, str, list[str]]] = []  # (title, text, tokens)
        self._load()

    def _load(self) -> None:
        if not self.folder.is_dir():
            log.warning("knowledge folder missing: %s", self.folder)
            return
        for path in sorted(self.folder.glob("*.md")):
            # routines.md is machine-readable task templates, read by routines.py.
            if path.name == "routines.md":
                continue
            heading = path.stem
            body: list[str] = []
            for line in path.read_text(encoding="utf-8").splitlines() + ["## "]:
                if line.startswith("## "):
                    text = " ".join(part for part in body if not part.startswith("<!--"))
                    text = " ".join(text.split())
                    if text:
                        self.chunks.append((heading, text, tokenize(heading + " " + text)))
                    heading, body = line[3:].strip(), []
                elif not line.startswith("# "):
                    body.append(line.strip())

        # Inverse document frequency: rare words matter more than common ones.
        count = len(self.chunks) or 1
        document_frequency: dict[str, int] = {}
        for _, _, tokens in self.chunks:
            for token in set(tokens):
                document_frequency[token] = document_frequency.get(token, 0) + 1
        self.idf = {
            token: math.log(1 + count / frequency)
            for token, frequency in document_frequency.items()
        }

    def search(self, query: str, k: int = 4) -> list[Passage]:
        query_tokens = set(tokenize(query))
        if not query_tokens:
            return []
        results = []
        for title, text, tokens in self.chunks:
            heading_tokens = set(tokenize(title))
            score = 0.0
            for token in query_tokens:
                if token in tokens:
                    # Matching the heading is a much stronger signal than the body.
                    weight = 2.0 if token in heading_tokens else 1.0
                    score += weight * self.idf.get(token, 0.0)
            if score > 0:
                results.append(Passage("notes", title, text, round(score, 3)))
        results.sort(key=lambda passage: -passage.score)
        return results[:k]


# --- Household -------------------------------------------------------------


def describe_pet(pet: Pet) -> str:
    """Plain-language facts about one pet and its tasks."""
    parts = [f"{pet.name} is a {pet.species}"]
    if pet.breed:
        parts[0] += f" ({pet.breed})"
    if pet.age_years is not None:
        parts.append(f"aged {pet.age_years}" if pet.age_years else "under a year old")
    facts = ", ".join(parts) + "."
    if not pet.tasks:
        return facts + " No care tasks have been added yet."
    lines = []
    for task in pet.tasks:
        repeat = task.recurrence or "one-off"
        when = f", from {task.earliest:%H:%M}" if task.earliest else ""
        lines.append(f"{task.title} ({task.duration_minutes} min, {task.priority}, {repeat}{when})")
    return facts + " Tasks: " + "; ".join(lines) + "."


def describe_plan(plan: dict) -> str:
    """Plain-language summary of a generated plan."""
    parts = []
    if plan["entries"]:
        slots = [
            f"{entry['start_time']:%H:%M} {entry['title']} ({entry['pet'].name})"
            for entry in plan["entries"]
        ]
        parts.append("Scheduled today: " + "; ".join(slots) + ".")
    else:
        parts.append("Nothing is scheduled today.")
    if plan["skipped"]:
        left = [f"{item['task'].title} ({item['reason']})" for item in plan["skipped"]]
        parts.append("Left out: " + "; ".join(left) + ".")
    if plan["warnings"]:
        parts.append("Warnings: " + "; ".join(plan["warnings"]) + ".")
    return " ".join(parts)


class HouseholdRetriever:
    """Turns the owner's real data into passages the answer can rely on."""

    def search(
        self, query: str, owner: Owner, scheduler: Scheduler | None = None
    ) -> list[Passage]:
        if owner is None or not owner.pets:
            return []
        lowered = query.lower()
        named = [pet for pet in owner.pets if pet.name.lower() in lowered]
        # If the question names a pet, that pet is the evidence. Otherwise the
        # whole household is, but scored lower so care notes can outrank it.
        pets = named or owner.pets
        score = 3.0 if named else 1.0
        passages = [Passage("household", pet.name, describe_pet(pet), score) for pet in pets]

        if scheduler is not None:
            plan = scheduler.build_household_plan(owner, date.today())
            plan_words = {"today", "plan", "schedule", "time", "when", "busy", "left", "skip"}
            plan_score = 2.0 if plan_words & set(tokenize(query)) else 0.5
            passages.append(Passage("household", "today's plan", describe_plan(plan), plan_score))
        return passages


# --- Web (Wikipedia) -------------------------------------------------------

WIKI_API = "https://en.wikipedia.org/w/api.php"
MAX_WEB_CHARS = 1200
MIN_SECONDS_BETWEEN_CALLS = 1.0
COOLDOWN_AFTER_429 = 60.0


def user_agent() -> str:
    """Wikipedia asks automated clients to identify themselves.

    The default is generic. Set PAWPAL_WIKI_CONTACT (a URL or email) to add
    contact details, which Wikipedia rate-limits less aggressively.
    """
    contact = os.environ.get("PAWPAL_WIKI_CONTACT", "").strip()
    return f"PawPalPlus/1.0 ({contact or 'student project; pet care assistant'})"


def _http_get_json(url: str, timeout: float) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent()})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class WebUnavailable(Exception):
    """The web source is cooling down after being rate-limited."""


class WebRetriever:
    """Searches Wikipedia and returns the intro of the best-matching articles.

    Polite by design: results are cached, calls are spaced at least a second
    apart, and after an HTTP 429 the web source is skipped for a minute instead
    of retrying. `fetch` is injectable so tests never touch the network.
    """

    def __init__(
        self,
        fetch: Callable[[str, float], dict] = _http_get_json,
        timeout: float = 5.0,
        enabled: bool = True,
        min_interval: float = MIN_SECONDS_BETWEEN_CALLS,
    ):
        self.fetch = fetch
        self.timeout = timeout
        self.enabled = enabled
        self.min_interval = min_interval
        self.cache: dict[str, list[Passage]] = {}
        self._last_call = 0.0
        self._cooldown_until = 0.0

    def search(self, query: str, k: int = 2) -> list[Passage]:
        """Return up to k passages. Raises on network/format errors; caller handles."""
        query = " ".join(query.split())[:200]
        if not query:
            return []
        if query in self.cache:
            # Copies, because Retriever numbers passages in place.
            return [replace(passage) for passage in self.cache[query][:k]]

        now = time.monotonic()
        if now < self._cooldown_until:
            raise WebUnavailable(
                f"Wikipedia rate-limited us; skipping web for {self._cooldown_until - now:.0f}s"
            )
        wait = self._last_call + self.min_interval - now
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

        params = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": query,
            "gsrlimit": str(k),
            "prop": "extracts|info",
            "exintro": "1",
            "explaintext": "1",
            "inprop": "url",
            "redirects": "1",
        }
        try:
            data = self.fetch(f"{WIKI_API}?{urllib.parse.urlencode(params)}", self.timeout)
        except urllib.error.HTTPError as error:
            if error.code == 429:
                self._cooldown_until = time.monotonic() + COOLDOWN_AFTER_429
            raise
        pages = data.get("query", {}).get("pages", {})
        ordered = sorted(pages.values(), key=lambda page: page.get("index", 0))

        query_tokens = set(tokenize(query))
        passages = []
        for page in ordered:
            text = " ".join((page.get("extract") or "").split())
            if not text:
                continue
            if len(text) > MAX_WEB_CHARS:
                text = text[:MAX_WEB_CHARS].rsplit(". ", 1)[0] + "."
            overlap = len(query_tokens & set(tokenize(page.get("title", "") + " " + text)))
            score = round(0.5 + overlap / max(len(query_tokens), 1), 3)
            passages.append(
                Passage("web", page.get("title", "Wikipedia"), text, score, page.get("fullurl"))
            )
        self.cache[query] = passages
        return [replace(passage) for passage in passages[:k]]


# --- Combined ---------------------------------------------------------------


class Retriever:
    """Runs every source, tolerates failures, and numbers the results S1..Sn."""

    def __init__(
        self,
        notes: NotesRetriever | None = None,
        web: WebRetriever | None = None,
        household: HouseholdRetriever | None = None,
    ):
        self.notes = notes or NotesRetriever()
        self.web = web if web is not None else WebRetriever()
        self.household = household or HouseholdRetriever()

    def retrieve(
        self,
        query: str,
        owner: Owner | None = None,
        scheduler: Scheduler | None = None,
        web_query: str | None = None,
        k_notes: int = 4,
        k_web: int = 2,
    ) -> RetrievalResult:
        result = RetrievalResult()

        if owner is not None:
            try:
                result.passages += self.household.search(query, owner, scheduler)
            except Exception as error:  # never let one source sink the answer
                log.exception("household retrieval failed")
                result.errors["household"] = str(error)

        try:
            result.passages += self.notes.search(query, k_notes)
        except Exception as error:
            log.exception("notes retrieval failed")
            result.errors["notes"] = str(error)

        if self.web.enabled and web_query and k_web > 0:
            try:
                result.passages += self.web.search(web_query, k_web)
            except Exception as error:
                log.warning("web retrieval failed: %s", error)
                result.errors["web"] = f"{type(error).__name__}: {error}"

        for number, passage in enumerate(result.passages, 1):
            passage.source_id = f"S{number}"
        log.info(
            "retrieved %d passage(s) %s for %r; errors=%s",
            len(result.passages),
            [(p.source_id, p.origin, p.title, p.score) for p in result.passages],
            query[:80],
            result.errors or "none",
        )
        return result
