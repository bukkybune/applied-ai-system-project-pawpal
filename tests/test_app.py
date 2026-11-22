"""Smoke tests for the Streamlit app, run headless with Streamlit's AppTest.

The advisor is pre-loaded offline with the web switched off, so no key and no
network are needed.
"""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from pawpal_ai import PawPalAdvisor  # noqa: E402
from pawpal_ai.retrieval import Retriever, WebRetriever  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture
def app():
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["advisor"] = PawPalAdvisor(retriever=Retriever(web=WebRetriever(enabled=False)))
    at.run()
    assert not at.exception
    return at


def tab(at, label):
    return next(t for t in at.tabs if label in t.label)


def test_app_starts_with_four_tabs_in_offline_mode(app):
    assert [t.label for t in app.tabs] == ["📅 Schedule", "🐣 New pet setup", "🩺 Care check", "💬 Ask PawPal"]
    assert any("Offline mode" in info.value for info in app.info)


def test_ask_tab_answers_with_sources(app):
    app.chat_input[0].set_value("Is chocolate toxic to dogs?").run()
    assert not app.exception
    answer = app.session_state["chat"][-1]["answer"]
    assert answer.kind == "answer" and answer.sources


def test_care_check_accept_adds_the_task_to_the_pet(app):
    care = tab(app, "Care check")
    care.button[0].click().run()  # "Check care" for Mochi
    assert not app.exception
    before = len(app.session_state["owner"].get_pet("Mochi").tasks)
    accept = next(b for b in tab(app, "Care check").button if b.label == "Accept")
    accept.click().run()
    assert not app.exception
    assert len(app.session_state["owner"].get_pet("Mochi").tasks) == before + 1


def test_setup_then_accept_adds_a_scheduled_pet(app):
    setup = tab(app, "New pet setup")
    setup.text_input[0].input("Biscuit")
    setup.text_input[1].input("a 3 year old cat")
    setup.button[0].click().run()
    assert not app.exception
    result = app.session_state["setup_result"]
    assert result.ok and result.pet.tasks

    accept = next(b for b in tab(app, "New pet setup").button if "Accept" in b.label)
    accept.click().run()
    assert not app.exception
    biscuit = app.session_state["owner"].get_pet("Biscuit")
    assert biscuit is not None and len(biscuit.tasks) == len(result.pet.tasks)
    assert len({t.task_id for t in biscuit.tasks}) == len(biscuit.tasks)


def test_schedule_shows_without_a_button_and_task_buttons_work(app):
    from pawpal_system import CareTask

    mochi = app.session_state["owner"].get_pet("Mochi")
    mochi.add_task(CareTask("w1", "Morning walk", 30, "high", "daily"))
    mochi.add_task(CareTask("b1", "Brush coat", 10, "low"))
    app.run()
    assert not app.exception
    timetable = tab(app, "Schedule").dataframe[0].value
    assert list(timetable["Task"]) == ["Morning walk", "Brush coat"]

    done = next(b for b in tab(app, "Schedule").button if b.key == "done_w1")
    done.click().run()
    assert mochi.tasks[0].is_complete
    remove = next(b for b in tab(app, "Schedule").button if b.key == "remove_b1")
    remove.click().run()
    assert [t.task_id for t in mochi.tasks] == ["w1"]


def test_schedule_greets_the_owner_by_name(app):
    app.sidebar.text_input[0].input("Dorcas").run()
    assert not app.exception
    assert "Hi Dorcas, here's today's schedule" in tab(app, "Schedule").subheader[0].value
