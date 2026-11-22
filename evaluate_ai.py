"""Reliability check for the PawPal Assistant.

Runs a fixed set of cases through the real advisor and prints a pass/fail
table, the average confidence, and a consistency check (same question twice).

    python evaluate_ai.py            # offline, local notes only: fully reproducible
    python evaluate_ai.py --web      # also query Wikipedia
    python evaluate_ai.py --live     # use the Gemini model from .env (implies nothing else)

Exit code is 0 when every case passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date, time

from pawpal_ai import PawPalAdvisor, make_client, setup_logging
from pawpal_ai.logging_setup import LOG_DIR
from pawpal_ai.retrieval import Retriever, WebRetriever, tokenize
from pawpal_system import CareTask, Owner, Pet, Scheduler

DAY = date(2026, 9, 29)


def household() -> Owner:
    owner = Owner("Jordan", 120, time(8, 0))
    mochi = Pet("Mochi", "dog", "Corgi", 3)
    mochi.add_task(CareTask("m1", "Morning walk", 30, "high", "daily"))
    mochi.add_task(CareTask("m2", "Breakfast", 10, "high", "daily", earliest="08:00", latest="09:00"))
    owner.add_pet(mochi)
    return owner


@dataclass
class Outcome:
    case: str
    criteria: str
    passed: bool
    confidence: float | None
    detail: str


def cited_titles(answer) -> set[str]:
    return {p.title for p in answer.sources}


class SpyLLM:
    """Wraps the real model (or nothing) and counts calls, to prove guardrails skip it."""

    def __init__(self, inner):
        self.inner = inner
        self.calls = 0
        self.name = getattr(inner, "name", "none")

    def generate(self, system, prompt, json_mode=False):
        self.calls += 1
        return self.inner.generate(system, prompt, json_mode)


def run(live: bool, web: bool) -> list[Outcome]:
    llm = make_client() if live else None
    if live and llm is None:
        print("--live needs GEMINI_API_KEY in .env; run: python -m pawpal_ai.check")
        sys.exit(2)
    spy = SpyLLM(llm) if llm else None

    def advisor(web_fetch=None) -> PawPalAdvisor:
        if web_fetch is not None:
            web_source = WebRetriever(fetch=web_fetch, min_interval=0)
        else:
            web_source = WebRetriever(enabled=web)
        return PawPalAdvisor(spy, Retriever(web=web_source), trace_path=LOG_DIR / "eval_traces.jsonl")

    main = advisor()
    owner, scheduler = household(), Scheduler(buffer_minutes=5)
    results: list[Outcome] = []

    def record(case, criteria, passed, confidence, detail):
        results.append(Outcome(case, criteria, bool(passed), confidence, detail))

    # 1. Plain care question
    a = main.ask("Is chocolate toxic to dogs?", owner, scheduler)
    record("Is chocolate toxic to dogs?", "answers, cites the toxic-foods note, mentions chocolate",
           a.kind == "answer" and "Foods toxic to dogs" in cited_titles(a) and "chocolate" in a.text.lower(),
           a.confidence, f"cited {sorted(cited_titles(a))}")

    # 2. Question about the user's own schedule
    a = main.ask("When is Mochi's walk today?", owner, scheduler)
    record("When is Mochi's walk today?", "uses the household/plan source",
           a.kind == "answer" and any(p.origin == "household" for p in a.sources),
           a.confidence, f"cited {sorted(cited_titles(a))}")

    # 3. Health question carries the not-a-vet notice
    a = main.ask("Mochi keeps scratching her ear", owner, scheduler)
    record("Mochi keeps scratching her ear", "answers from the itching note + not-a-vet notice",
           a.kind == "answer" and bool(a.notice) and "Itching and scratching" in cited_titles(a),
           a.confidence, f"cited {sorted(cited_titles(a))}")

    # 4-6. Guardrails: the model must never be called
    for question, kind in [
        ("My dog just ate a whole bar of chocolate", "emergency"),
        ("Write me a poem about the stock market", "off_topic"),
        ("", "empty"),
    ]:
        before = spy.calls if spy else 0
        a = main.ask(question, owner, scheduler)
        skipped_model = (spy.calls if spy else 0) == before
        record(repr(question) if question else "(empty input)", f"blocked as {kind}, model not called",
               a.kind == kind and skipped_model, None, f"kind={a.kind}")

    # 7. Web source down: answer still comes back, failure is reported
    def broken(url, timeout):
        raise TimeoutError("simulated outage")

    a = advisor(broken).ask("How much exercise does Mochi need?", owner, scheduler)
    record("Web source down", "still answers from notes, reports the outage",
           a.kind == "answer" and any("web source unavailable" in w for w in a.warnings),
           a.confidence, "; ".join(a.warnings)[:60])

    # 8. Care check finds real gaps with valid, sourced suggestions
    check = main.care_check(owner.get_pet("Mochi"), owner, scheduler)
    areas = {gap.covers for gap in check.gaps}
    sourced = all(gap.source_ids for gap in check.gaps)
    record("Care check: Mochi", "finds dental + exercise gaps, every gap sourced",
           {"dental", "exercise"} <= areas and sourced, check.confidence, f"gaps {sorted(areas)}")

    # 9. Setup agent: tight day, must resolve problems without pushing out Mochi
    result = main.setup_pet("Nova", "I just adopted a 10-week-old Siberian Husky puppy",
                            household(), scheduler, DAY)
    titles = {task.title.lower() for task in result.pet.tasks} if result.ok else set()
    feeding_kept = sum("breakfast" in t or "dinner" in t or "lunch" in t for t in titles) >= 2
    record("Setup: husky puppy, 120 min day", "routine fits, feeding kept, changes explained",
           result.ok and not result.problems and feeding_kept and (result.rounds == 0 or result.changes),
           result.confidence, f"{len(titles)} tasks, {result.rounds} revision round(s)")

    # 10. Consistency: same question twice gives the same sources and advice
    q = "How often should I brush a husky?"
    first, second = main.ask(q, owner, scheduler), main.ask(q, owner, scheduler)
    words_1, words_2 = set(tokenize(first.text)), set(tokenize(second.text))
    overlap = len(words_1 & words_2) / max(len(words_1 | words_2), 1)
    record("Consistency (asked twice)", "same cited sources, >=60% word overlap",
           cited_titles(first) == cited_titles(second) and overlap >= 0.6,
           first.confidence, f"word overlap {overlap:.0%}")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--live", action="store_true", help="use the Gemini model from .env")
    parser.add_argument("--web", action="store_true", help="also query Wikipedia")
    args = parser.parse_args()

    setup_logging()
    mode = "live AI" if args.live else "offline"
    print(f"PawPal Assistant evaluation ({mode}, web {'on' if args.web else 'off'})\n")
    results = run(args.live, args.web)

    width = max(len(r.case) for r in results)
    print(f"{'Case':<{width}}  {'Result':<6}  {'Conf':>4}  Criteria / detail")
    print("-" * (width + 70))
    for r in results:
        conf = f"{r.confidence:.2f}" if r.confidence is not None else "  - "
        print(f"{r.case:<{width}}  {'PASS' if r.passed else 'FAIL':<6}  {conf:>4}  {r.criteria} ({r.detail})")

    passed = sum(r.passed for r in results)
    scored = [r.confidence for r in results if r.confidence is not None]
    average = sum(scored) / len(scored) if scored else 0.0
    print(f"\n{passed}/{len(results)} passed; average confidence {average:.2f}")

    out = LOG_DIR / "eval_results.json"
    out.write_text(json.dumps(
        {"mode": mode, "web": args.web, "passed": passed, "total": len(results),
         "average_confidence": round(average, 2), "cases": [r.__dict__ for r in results]},
        indent=2,
    ))
    print(f"Saved to {out.relative_to(LOG_DIR.parent)}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
