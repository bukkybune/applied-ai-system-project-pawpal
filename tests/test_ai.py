"""Tests for the PawPal Assistant (pawpal_ai).

No network and no API key: the model is a scripted fake and Wikipedia is a
fake fetch function, so these run the same way on every machine.
"""

import json
import urllib.error
from datetime import date, time

import pytest

from pawpal_ai import advisor as advisor_module
from pawpal_ai import safety
from pawpal_ai.advisor import PawPalAdvisor, parse_profile_rules
from pawpal_ai.llm import LLMError, parse_json
from pawpal_ai.retrieval import (
    HouseholdRetriever,
    NotesRetriever,
    Retriever,
    WebRetriever,
    WebUnavailable,
    tokenize,
)
from pawpal_ai.routines import find_gaps, life_stage, parse_routines
from pawpal_ai.trace import Trace
from pawpal_system import CareTask, Owner, Pet, Scheduler

DAY = date(2026, 9, 29)


# --- Fakes -----------------------------------------------------------------


class FakeLLM:
    """Returns scripted replies chosen by a function of (system, prompt)."""

    name = "FakeLLM"

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def generate(self, system, prompt, json_mode=False):
        self.calls.append((system, prompt, json_mode))
        result = self.reply(system, prompt) if callable(self.reply) else self.reply
        if isinstance(result, Exception):
            raise result
        return result


def wiki_payload(title="Siberian Husky", extract="The Siberian Husky is a medium-sized working sled dog breed."):
    return {"query": {"pages": {"1": {"index": 1, "title": title, "extract": extract,
                                      "fullurl": f"https://en.wikipedia.org/wiki/{title}"}}}}


class FakeFetch:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def __call__(self, url, timeout):
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def offline_retriever(fetch=None):
    web = WebRetriever(fetch=fetch, min_interval=0) if fetch else WebRetriever(enabled=False)
    return Retriever(web=web)


@pytest.fixture
def household():
    owner = Owner("Jordan", 120, time(8, 0))
    mochi = Pet("Mochi", "dog", "Corgi", 3)
    mochi.add_task(CareTask("m1", "Morning walk", 30, "high", "daily"))
    mochi.add_task(CareTask("m2", "Breakfast", 10, "high", "daily", earliest="08:00", latest="09:00"))
    owner.add_pet(mochi)
    return owner


# --- Retrieval ----------------------------------------------------------------


def test_tokenize_folds_plurals_and_drops_stopwords():
    assert tokenize("How often do Huskies need walking?") == ["often", "husky", "walk"]


def test_notes_search_ranks_the_matching_section_first():
    results = NotesRetriever().search("is chocolate toxic to dogs", k=3)
    assert results[0].title == "Foods toxic to dogs"
    assert all(passage.origin == "notes" for passage in results)


def test_household_passages_favour_the_named_pet_and_include_the_plan(household):
    passages = HouseholdRetriever().search("what is Mochi doing today", household, Scheduler())
    titles = [p.title for p in passages]
    assert titles == ["Mochi", "today's plan"]
    assert "Morning walk" in passages[0].text
    assert passages[1].score > 1  # "today" makes the plan relevant


def test_web_results_are_parsed_and_cached():
    fetch = FakeFetch(wiki_payload())
    web = WebRetriever(fetch=fetch, min_interval=0)
    first = web.search("Siberian Husky")
    second = web.search("Siberian Husky")
    assert first[0].title == "Siberian Husky" and first[0].url.endswith("Siberian Husky")
    assert second[0].text == first[0].text
    assert fetch.calls == 1


def test_web_cools_down_after_rate_limit_instead_of_retrying():
    fetch = FakeFetch(urllib.error.HTTPError("u", 429, "Too Many Requests", {}, None))
    web = WebRetriever(fetch=fetch, min_interval=0)
    with pytest.raises(urllib.error.HTTPError):
        web.search("husky")
    with pytest.raises(WebUnavailable):
        web.search("corgi")
    assert fetch.calls == 1


def test_a_failing_source_is_reported_but_the_others_still_answer(household):
    retriever = offline_retriever(FakeFetch(TimeoutError("timed out")))
    result = retriever.retrieve("dog exercise", household, web_query="dog exercise")
    assert "web" in result.errors
    assert {"notes", "household"} <= result.origins()
    assert [p.source_id for p in result.passages] == [f"S{i}" for i in range(1, len(result.passages) + 1)]


# --- Guardrails ------------------------------------------------------------------


@pytest.mark.parametrize(
    "question, kind",
    [
        ("", "empty"),
        ("   ", "empty"),
        ("x" * 1001, "too_long"),
        ("My dog ate chocolate an hour ago", "emergency"),
        ("my cat is having a seizure", "emergency"),
        ("he has been vomiting blood since lunch", "emergency"),
        ("how do I poison a cat", "harmful"),
        ("how much chocolate would kill a dog", "harmful"),
        ("what is the capital of France?", "off_topic"),
        ("Is chocolate toxic to dogs?", "ok"),
        ("How often should Biscuit eat?", "ok"),  # pet name alone makes it on-topic
    ],
)
def test_question_guardrails(question, kind):
    assert safety.check_question(question, ["Biscuit"]).kind == kind


def test_invented_citations_are_removed():
    text, cited, invalid = safety.check_citations("Walk twice a day [S1, S9]. Feed well [S7].", {"S1", "S2"})
    assert text == "Walk twice a day [S1]. Feed well."
    assert cited == {"S1"} and invalid == {"S9", "S7"}


@pytest.mark.parametrize(
    "raw, problem",
    [
        ({"title": "Walk", "minutes": 0, "priority": "high"}, "minutes"),
        ({"title": "Walk", "minutes": 500, "priority": "high"}, "minutes"),
        ({"title": "Walk", "minutes": "ten", "priority": "high"}, "number"),
        ({"title": "Walk", "minutes": 10, "priority": "urgent"}, "priority"),
        ({"title": "Morning walk", "minutes": 10, "priority": "low"}, "already exists"),
        ({"title": "Walk", "minutes": 10, "priority": "low", "repeats": "hourly"}, "repeats"),
        ({"title": "Walk", "minutes": 10, "priority": "low", "window": "10:00-09:00"}, "window"),
        ({"title": "Give ibuprofen", "minutes": 1, "priority": "high"}, "vet"),
        ({"title": "Joint support", "minutes": 1, "priority": "high", "notes": "250 mg daily"}, "vet"),
    ],
)
def test_bad_ai_tasks_are_rejected(raw, problem):
    fields, reason = safety.validate_task(raw, {"morning walk"})
    assert fields is None and problem in reason


def test_valid_ai_task_becomes_care_task_fields():
    fields, _ = safety.validate_task(
        {"title": "Brush", "minutes": 10, "priority": "low", "repeats": "weekly:sat,sun",
         "window": "18:00-19:00"},
        set(),
    )
    task = CareTask(task_id="x", **fields)
    assert task.weekdays == {"sat", "sun"} and task.earliest == time(18, 0)


def test_parse_json_tolerates_code_fences_and_rejects_garbage():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    with pytest.raises(LLMError):
        parse_json("sorry, I can't do that")


# --- Ask PawPal --------------------------------------------------------------------


def test_offline_answer_is_built_from_cited_sources(household):
    answer = PawPalAdvisor(retriever=offline_retriever()).ask("Is chocolate toxic to dogs?", household)
    assert answer.kind == "answer" and answer.mode == "offline"
    assert "chocolate" in answer.text.lower()
    assert any(p.title == "Foods toxic to dogs" for p in answer.sources)
    assert 0 < answer.confidence <= 1


def test_emergency_never_reaches_the_model(household):
    llm = FakeLLM("should not be called")
    answer = PawPalAdvisor(llm, offline_retriever()).ask("my puppy ate grapes", household)
    assert answer.kind == "emergency" and llm.calls == []
    assert "vet" in answer.text


def test_live_answer_keeps_valid_citations_and_drops_invented_ones(household):
    llm = FakeLLM("Chocolate is toxic to dogs [S2]. Mochi loves it [S99].")
    good = PawPalAdvisor(FakeLLM("Chocolate is toxic to dogs [S2]."), offline_retriever())
    answer = PawPalAdvisor(llm, offline_retriever()).ask("Is chocolate toxic to dogs?", household)
    assert "[S99]" not in answer.text and answer.mode == "live"
    assert any("S99" in warning for warning in answer.warnings)
    assert answer.confidence < good.ask("Is chocolate toxic to dogs?", household).confidence


def test_live_answer_without_citations_is_replaced_by_the_sources(household):
    llm = FakeLLM("Sure, chocolate is totally fine for dogs.")
    answer = PawPalAdvisor(llm, offline_retriever()).ask("Is chocolate toxic to dogs?", household)
    assert answer.mode == "offline"
    assert "fine for dogs" not in answer.text
    assert any("cited no sources" in warning for warning in answer.warnings)


def test_model_failure_falls_back_to_offline(household):
    llm = FakeLLM(LLMError("quota exceeded"))
    answer = PawPalAdvisor(llm, offline_retriever()).ask("How much exercise does a dog need?", household)
    assert answer.kind == "answer" and answer.mode == "offline"
    assert any("quota exceeded" in warning for warning in answer.warnings)


def test_prompt_contains_the_household_and_notes(household):
    llm = FakeLLM("Mochi walks at 08:00 [S1].")
    PawPalAdvisor(llm, offline_retriever()).ask("When does Mochi walk today?", household, Scheduler())
    prompt = llm.calls[0][1]
    assert "Morning walk" in prompt and "[S1]" in prompt and "Question:" in prompt


def test_health_answers_carry_a_not_a_vet_notice(household):
    answer = PawPalAdvisor(retriever=offline_retriever()).ask("Mochi keeps scratching her ear", household)
    assert answer.notice == safety.HEALTH_NOTICE


# --- Care check -------------------------------------------------------------------


def test_rule_gaps_find_missing_areas_and_short_exercise(household):
    gaps = {gap.covers: gap for gap in find_gaps(household.get_pet("Mochi"))}
    assert {"dental", "grooming", "exercise"} <= set(gaps)
    assert "feeding" not in gaps  # Breakfast covers it
    assert "30 min" in gaps["exercise"].issue


def test_high_energy_breeds_get_a_higher_exercise_target():
    husky = Pet("Nova", "dog", "Siberian Husky", 4)
    husky.add_task(CareTask("n1", "Walk", 50, "high", "daily"))
    gaps = {gap.covers: gap for gap in find_gaps(husky)}
    assert "about 60 min" in gaps["exercise"].issue


def test_life_stages():
    assert life_stage("dog", 0) == "dog-puppy"
    assert life_stage("cat", 0) == "cat-kitten"
    assert life_stage("dog", 9) == "dog-senior"
    assert life_stage("cat", 9) == "cat-adult"
    assert life_stage("rabbit", 2) == "other"
    assert set(parse_routines()) >= {"dog-adult", "dog-puppy", "cat-adult", "other"}


def test_offline_care_check_cites_the_right_notes(household):
    check = PawPalAdvisor(retriever=offline_retriever()).care_check(household.get_pet("Mochi"))
    by_id = {p.source_id: p for p in check.sources}
    grooming = next(gap for gap in check.gaps if gap.covers == "grooming")
    assert by_id[grooming.source_ids[0]].title == "Grooming dogs"
    assert check.mode == "offline" and check.confidence == 0.8


def test_ai_care_check_ignores_unsourced_gaps_but_keeps_rule_gaps(household):
    reply = json.dumps({"gaps": [
        {"covers": "dental", "issue": "No tooth care.", "source_ids": ["S1"],
         "task": {"title": "Teeth brushing", "minutes": 5, "priority": "medium", "repeats": "daily"}},
        {"covers": "supplements", "issue": "Give fish oil.", "source_ids": ["S42"],
         "task": {"title": "Fish oil", "minutes": 1, "priority": "high", "repeats": "daily"}},
    ]})
    check = PawPalAdvisor(FakeLLM(reply), offline_retriever()).care_check(household.get_pet("Mochi"))
    areas = {gap.covers: gap.origin for gap in check.gaps}
    assert areas["dental"] == "agreed"
    assert "supplements" not in areas
    assert areas["grooming"] == "rules"  # the AI skipped it; the rule check kept it
    assert any("no valid sources" in warning for warning in check.warnings)


# --- New pet setup (agent) ---------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("I adopted a 10-week-old Siberian Husky puppy", ("dog", "Siberian Husky", 0)),
        ("a 3 year old cat", ("cat", None, 3)),
        ("rescued a 12 yr old labrador", ("dog", "Labrador", 12)),
        ("a maine coon", ("cat", "Maine Coon", None)),
        ("a pet rabbit", ("other", None, None)),
        ("3 month old domesti shorthair", ("cat", None, 0)),  # typo + no "cat" word
        ("a domestic long-haired rescue, 2 years", ("cat", None, 2)),
    ],
)
def test_profile_rules(text, expected):
    profile = parse_profile_rules(text)
    assert (profile["species"], profile["breed"], profile["age_years"]) == expected


def test_agent_checks_its_draft_and_revises_until_it_fits(household):
    advisor = PawPalAdvisor(retriever=offline_retriever())
    result = advisor.setup_pet("Nova", "a 10-week-old husky puppy", household,
                               Scheduler(buffer_minutes=5), DAY)
    steps = [step.step for step in result.trace.steps]
    assert result.ok and result.profile["stage"] == "dog-puppy"
    assert steps.count("check") == result.rounds + 1
    assert result.rounds >= 1 and "revise" in steps
    assert result.problems == []
    # Nothing is dropped silently, and essential care survives the cuts.
    assert result.changes
    titles = {task.title for task in result.pet.tasks}
    assert {"Puppy breakfast", "Puppy dinner", "Toilet break after meals"} <= titles
    # Mochi's existing tasks are never pushed out of the day.
    assert not any(p["kind"] == "pushed_out_existing" for p in result.problems)


def test_accepted_routine_is_planned_by_the_real_scheduler(household):
    result = PawPalAdvisor(retriever=offline_retriever()).setup_pet(
        "Biscuit", "a 3 year old cat", household, Scheduler(), DAY
    )
    household.add_pet(result.pet)
    plan = Scheduler().build_household_plan(household, DAY)
    planned = {(entry["pet"].name, entry["title"]) for entry in plan["entries"]}
    assert ("Biscuit", "Scoop litter box") in planned
    assert ("Mochi", "Morning walk") in planned


def test_agent_falls_back_to_template_when_the_ai_draft_is_unusable(household):
    llm = FakeLLM(lambda system, prompt: "not json" if "starting care routine" in system
                  else '{"species": "cat", "breed": null, "age_years": 3}')
    result = PawPalAdvisor(llm, offline_retriever()).setup_pet(
        "Biscuit", "a 3 year old cat", household, Scheduler(), DAY
    )
    assert result.ok and result.profile["parsed_by"] == "ai"
    assert any("used the routine template" in warning for warning in result.warnings)
    assert "Scoop litter box" in {task.title for task in result.pet.tasks}


def test_agent_uses_the_ai_revision_when_the_draft_does_not_fit(household):
    big = {"tasks": [{"title": f"Play {n}", "minutes": 40, "priority": "medium",
                      "repeats": "daily", "source_ids": ["S1"]} for n in range(4)]}
    small = {"tasks": [{"title": "Play 0", "minutes": 20, "priority": "medium", "repeats": "daily"}]}
    replies = iter([json.dumps({"species": "dog", "breed": None, "age_years": 2}),
                    json.dumps(big), json.dumps(small)])
    llm = FakeLLM(lambda system, prompt: next(replies))
    result = PawPalAdvisor(llm, offline_retriever()).setup_pet(
        "Rex", "a 2 year old dog", household, Scheduler(), DAY
    )
    assert result.rounds == 1 and result.problems == []
    assert [task.title for task in result.pet.tasks] == ["Play 0"]
    assert any(step.summary.endswith("AI revised the routine") for step in result.trace.steps)


def test_agent_refuses_duplicate_names_and_unsafe_descriptions(household):
    advisor = PawPalAdvisor(retriever=offline_retriever())
    assert not advisor.setup_pet("mochi", "a dog", household, Scheduler()).ok
    assert advisor.setup_pet("Rex", "how do I poison a cat", household, Scheduler()).mode == "guardrail"
    assert advisor.setup_pet("Rex", "", household, Scheduler()).mode == "guardrail"


def test_setup_descriptions_are_not_blocked_as_off_topic(household):
    # A description has no pet word in it, but it is about a pet by definition.
    result = PawPalAdvisor(retriever=offline_retriever()).setup_pet(
        "Olly", "3 month old domesti shorthair", household, Scheduler(), DAY
    )
    assert result.ok and result.profile["stage"] == "cat-kitten"
    assert "Scoop litter box" in {task.title for task in result.pet.tasks}


def test_traces_are_saved_as_json_lines(household, tmp_path):
    path = tmp_path / "traces.jsonl"
    PawPalAdvisor(retriever=offline_retriever(), trace_path=path).setup_pet(
        "Biscuit", "a kitten", household, Scheduler(), DAY
    )
    record = json.loads(path.read_text().splitlines()[0])
    assert record["mode"] == "offline"
    assert [s["step"] for s in record["steps"]][:3] == ["parse", "research", "draft"]


def test_trace_records_steps_in_order():
    trace = Trace("goal", "offline")
    trace.add("parse", "a")
    trace.add("draft", "b", tasks=["x"])
    assert [s.step for s in trace.steps] == ["parse", "draft"]
    assert trace.to_dict()["steps"][1]["data"] == {"tasks": ["x"]}


def test_module_exposes_systems_prompts_that_forbid_medication():
    assert "Never suggest medicines" in advisor_module.CARE_SYSTEM
    assert "never include medicines" in advisor_module.SETUP_SYSTEM


# --- Model client ------------------------------------------------------------------


def test_no_key_means_offline_mode(monkeypatch):
    from pawpal_ai import llm

    monkeypatch.setattr(llm, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert llm.make_client() is None


def test_gemini_errors_become_llm_errors_so_the_app_can_fall_back(monkeypatch):
    errors = pytest.importorskip("google.genai.errors")
    from pawpal_ai.llm import GeminiClient

    client = GeminiClient("fake-key", "gemini-test")
    quota = errors.APIError(429, {"error": {"code": 429, "message": "quota exceeded"}})

    def refuse(**kwargs):
        raise quota

    monkeypatch.setattr(client._client.models, "generate_content", refuse)
    with pytest.raises(LLMError, match="429"):
        client.generate("system", "prompt")


def test_gemini_request_uses_low_temperature_and_json_mode(monkeypatch):
    pytest.importorskip("google.genai")
    from pawpal_ai.llm import GeminiClient

    client = GeminiClient("fake-key", "gemini-test")
    seen = {}

    class Reply:
        text = '{"ok": true}'

    def capture(**kwargs):
        seen.update(kwargs)
        return Reply()

    monkeypatch.setattr(client._client.models, "generate_content", capture)
    assert client.generate("be careful", "hello", json_mode=True) == '{"ok": true}'
    assert seen["model"] == "gemini-test"
    assert seen["config"].temperature == 0.2
    assert seen["config"].response_mime_type == "application/json"
    assert seen["config"].system_instruction == "be careful"


@pytest.mark.parametrize(
    "breed, species, query",
    [
        ("Chihuahua", "dog", "Chihuahua dog"),  # alone, Wikipedia finds the Mexican state
        ("Maine Coon", "cat", "Maine Coon cat"),
        (None, "cat", "cat"),
        ("Holland Lop", "other", "Holland Lop"),
    ],
)
def test_web_searches_name_the_species_to_avoid_ambiguous_breeds(breed, species, query):
    assert advisor_module.web_topic(breed, species) == query


def test_safety_net_fixes_what_the_ai_revisions_could_not(household):
    # Replays a real live failure: the model kept a breakfast window that ends
    # when the owner's day starts, and a routine big enough to push Mochi out.
    stubborn = {"tasks": [
        {"title": "Puppy breakfast", "minutes": 10, "priority": "high", "repeats": "daily",
         "window": "07:00-08:00", "source_ids": ["S1"]},
        {"title": "Long play", "minutes": 90, "priority": "medium", "repeats": "daily",
         "source_ids": ["S1"]},
    ]}
    llm = FakeLLM(lambda system, prompt: json.dumps(stubborn) if "starting care routine" in system
                  else '{"species": "dog", "breed": "Siberian Husky", "age_years": 0}')
    result = PawPalAdvisor(llm, offline_retriever()).setup_pet(
        "Nova", "a husky puppy", household, Scheduler(buffer_minutes=5), DAY
    )
    assert result.rounds == 2  # the AI got its two rounds first
    assert result.problems == []
    assert any("Safety net" in step.summary for step in result.trace.steps)
    breakfast = next(t for t in result.pet.tasks if t.title == "Puppy breakfast")
    assert not breakfast.is_anchored  # the impossible window was removed


def test_draft_prompt_tells_the_model_when_the_day_starts(household):
    llm = FakeLLM(lambda system, prompt: '{"tasks": []}' if "starting care routine" in system
                  else '{"species": "cat", "breed": null, "age_years": 3}')
    PawPalAdvisor(llm, offline_retriever()).setup_pet("Biscuit", "a cat", household, Scheduler(), DAY)
    draft_prompt = next(prompt for system, prompt, _ in llm.calls if "starting care routine" in system)
    assert "starts at 08:00" in draft_prompt and "Morning walk (Mochi" in draft_prompt


ALL_HIGH_PUPPY = {"tasks": [  # copied from a real live run where every task was high priority
    {"title": t, "minutes": 10, "priority": "high", "repeats": "daily", "window": w, "source_ids": ["S1"]}
    for t, w in [("Puppy breakfast", "08:50-09:10"), ("Toilet break after breakfast", "09:00-09:30"),
                 ("Puppy lunch", "12:00-13:00"), ("Toilet break after lunch", "12:10-13:10"),
                 ("Puppy dinner", "18:00-19:00"), ("Toilet break after dinner", "18:10-19:10")]
]}


def _stubborn_llm():
    return FakeLLM(lambda system, prompt: json.dumps(ALL_HIGH_PUPPY) if "starting care routine" in system
                   else '{"species": "dog", "breed": "Siberian Husky", "age_years": 0}')


def test_all_high_priority_routine_is_shortened_instead_of_pushing_out_another_pet(household):
    result = PawPalAdvisor(_stubborn_llm(), offline_retriever()).setup_pet(
        "Nova", "a husky puppy", household, Scheduler(buffer_minutes=5), DAY
    )
    assert result.problems == []
    assert len(result.pet.tasks) == 6  # essential care shortened, never dropped
    assert any(change.startswith("shortened") for change in result.changes)
    plan = result.plan
    assert ("Mochi", "Morning walk") in {(e["pet"].name, e["title"]) for e in plan["entries"]}


def test_a_day_too_short_for_essential_care_is_reported_not_hidden(household):
    household.available_minutes = 50
    result = PawPalAdvisor(_stubborn_llm(), offline_retriever()).setup_pet(
        "Nova", "a husky puppy", household, Scheduler(buffer_minutes=5), DAY
    )
    assert result.problems and result.confidence < 0.8
    assert len(result.pet.tasks) == 6


@pytest.mark.parametrize(
    "data, expected",
    [
        ({"gaps": [{"a": 1}]}, [{"a": 1}]),
        ([{"a": 1}], [{"a": 1}]),               # bare list
        ({"Gaps": [{"a": 1}]}, [{"a": 1}]),     # different capitalisation
        ({"result": [{"a": 1}]}, [{"a": 1}]),   # one list under another name
        ({"x": [{"a": 1}], "y": [{"b": 2}]}, None),  # ambiguous: refuse to guess
        ("text", None),
    ],
)
def test_model_json_lists_are_found_in_common_shapes(data, expected):
    assert advisor_module.find_list(data, "gaps") == expected


def test_care_check_accepts_a_bare_list_from_the_model(household):
    reply = json.dumps([{"covers": "dental", "issue": "No tooth care.", "source_ids": ["S1"],
                         "task": {"title": "Teeth brushing", "minutes": 5, "priority": "medium",
                                  "repeats": "daily"}}])
    check = PawPalAdvisor(FakeLLM(reply), offline_retriever()).care_check(household.get_pet("Mochi"))
    assert check.mode == "live"
    assert {gap.covers: gap.origin for gap in check.gaps}["dental"] == "agreed"
