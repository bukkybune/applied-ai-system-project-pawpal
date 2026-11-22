from dataclasses import replace
from datetime import date, time, timedelta

import streamlit as st

from pawpal_ai import PawPalAdvisor, make_client, setup_logging
from pawpal_ai.logging_setup import LOG_DIR
from pawpal_ai.safety import validate_task
from pawpal_system import CareTask, Owner, Pet, Scheduler

st.set_page_config(page_title="PawPal+", page_icon="🐾", layout="wide")

st.title("🐾 PawPal+")
st.caption("Plan a day of pet care around the time you actually have, with an AI assistant that looks things up first.")


# --- Session state -------------------------------------------------------
# Streamlit re-runs this script on every interaction, so the owner is created
# once and kept in the session vault. Everything else hangs off it.
if "owner" not in st.session_state:
    st.session_state.owner = Owner(
        name="Jordan", available_minutes=120, preferred_start_time=time(8, 0)
    )
    st.session_state.owner.add_pet(Pet(name="Mochi", species="dog", breed="Corgi", age_years=3))

if "next_task_id" not in st.session_state:
    st.session_state.next_task_id = 1

if "advisor" not in st.session_state:
    setup_logging()
    st.session_state.advisor = PawPalAdvisor(
        make_client(), trace_path=LOG_DIR / "agent_traces.jsonl"
    )
for key, default in (("chat", []), ("setup_result", None), ("care_results", {})):
    if key not in st.session_state:
        st.session_state[key] = default

owner = st.session_state.owner
advisor = st.session_state.advisor
today = date.today()


def new_task_id() -> str:
    """Hand out a unique task id, since CareTask requires one."""
    task_id = f"t{st.session_state.next_task_id}"
    st.session_state.next_task_id += 1
    return task_id


def mode_banner() -> None:
    if advisor.mode == "live":
        st.success(f"🤖 {advisor.mode_label}", icon="✅")
    else:
        st.info(
            f"📴 {advisor.mode_label}. Add a free GEMINI_API_KEY to a .env file to turn on "
            "the AI model (see README → Setup).",
        )


def show_sources(passages, cited_ids=None, title="Sources") -> None:
    if not passages:
        return
    with st.expander(f"{title} ({len(passages)})"):
        for passage in passages:
            used = cited_ids is None or passage.source_id in cited_ids
            link = f" — [open]({passage.url})" if passage.url else ""
            mark = "" if used else " *(retrieved, not cited)*"
            st.markdown(f"**[{passage.source_id}] {passage.label()}**{link}{mark}")
            st.caption(passage.text[:400] + ("…" if len(passage.text) > 400 else ""))


def show_warnings(warnings) -> None:
    for warning in warnings:
        st.caption(f"⚠️ {warning}")


def repeats_label(task: CareTask) -> str:
    if task.recurrence == "weekly" and task.weekdays:
        return "weekly: " + ", ".join(sorted(task.weekdays))
    if task.recurrence == "interval":
        return f"every {task.interval_days} days"
    return task.recurrence or "one-off"


# --- Owner and day constraints -------------------------------------------
with st.sidebar:
    st.header("Your day")
    owner.name = st.text_input("Owner name", value=owner.name)
    owner.available_minutes = st.number_input(
        "Minutes available", min_value=5, max_value=600, value=owner.available_minutes, step=5
    )
    owner.preferred_start_time = st.time_input("Start at", value=owner.preferred_start_time)

    st.header("Scheduling style")
    buffer_minutes = st.slider("Gap between tasks (min)", 0, 30, 5, step=5)
    allow_partial = st.checkbox(
        "Shorten tasks to fit", value=False, help="Book a task for less than its full length."
    )
    cap_per_pet = st.checkbox("Cap time per pet", value=False)
    max_per_pet = (
        st.number_input("Minutes per pet", min_value=5, max_value=300, value=60, step=5)
        if cap_per_pet
        else None
    )

    st.header("Filters")
    only_pets = st.multiselect("Only these pets", [pet.name for pet in owner.pets], default=[])
    priority_floor = st.selectbox("Minimum priority", ["any", "low", "medium", "high"])

    st.header("AI assistant")
    st.caption(advisor.mode_label)
    advisor.retriever.web.enabled = st.checkbox(
        "Look things up on Wikipedia", value=advisor.retriever.web.enabled,
        help="Turn off to use only your care notes and household data.",
    )

scheduler = Scheduler(
    buffer_minutes=buffer_minutes,
    allow_partial=allow_partial,
    max_minutes_per_pet=max_per_pet,
)

tab_schedule, tab_setup, tab_check, tab_ask = st.tabs(
    ["📅 Schedule", "🐣 New pet setup", "🩺 Care check", "💬 Ask PawPal"]
)


# ======================================================================
# Schedule tab (the original PawPal+ planner), laid out schedule-first
# ======================================================================
with tab_schedule:
    # --- Today's schedule: always current, no button to press ----------
    # Planning is instant, so rebuild it on every rerun; the timetable can
    # then never be out of date with the task list below it.
    plan = scheduler.build_household_plan(
        owner,
        today,
        only_pets=only_pets or None,
        min_priority=None if priority_floor == "any" else priority_floor,
    )
    name = owner.name.strip()
    st.subheader(f"👋 Hi {name}, here's today's schedule" if name else "Today's schedule")
    pet_names = ", ".join(pet.name for pet in owner.pets)
    st.caption(
        f"{today:%A %d %B} · {owner.available_minutes} minutes from "
        f"{owner.preferred_start_time:%H:%M}" + (f" · for {pet_names}" if pet_names else "")
    )

    if not owner.pets:
        st.info("No pets yet. Add one below, or try the 🐣 New pet setup tab.")
    elif not any(pet.tasks for pet in owner.pets):
        st.info("No tasks yet. Add one below, or run a 🩺 Care check for suggestions.")
    else:
        if plan["entries"]:
            booked = sum(entry["scheduled_minutes"] for entry in plan["entries"])
            used = plan["minutes_used"]
            cols = st.columns(4)
            count = len(plan["entries"])
            cols[0].metric("Scheduled", f"{count} task{'' if count == 1 else 's'}")
            cols[1].metric("On tasks", f"{booked} min")
            cols[2].metric("On gaps", f"{used - booked} min")
            cols[3].metric("Spare", f"{owner.available_minutes - used} min")

        for warning in plan["warnings"]:
            st.warning(warning, icon="⚠️")

        if plan["entries"]:
            st.dataframe(
                [
                    {
                        "Time": f"{entry['start_time']:%H:%M}–{entry['end_time']:%H:%M}",
                        "Task": entry["title"],
                        "Pet": entry["pet"].name,
                        "Min": entry["scheduled_minutes"],
                        "Priority": entry["priority"],
                        # Flag anchored tasks that could not get the time they asked
                        # for, so a shifted appointment is obvious at a glance.
                        "Note": (
                            f"moved from {entry['task'].earliest:%H:%M}"
                            if entry["task"].earliest
                            and entry["task"].earliest != entry["start_time"]
                            else entry["notes"] or ""
                        ),
                    }
                    for entry in plan["entries"]
                ],
                hide_index=True,
                use_container_width=True,
            )
            if not plan["skipped"]:
                st.success("Everything on the list fits in the day.")
        else:
            st.error("Nothing fits in the time available.")

        cols = st.columns(2)
        if plan["skipped"]:
            with cols[0].expander(f"Left out ({len(plan['skipped'])})"):
                for item in plan["skipped"]:
                    st.write(f"**{item['task'].title}** ({item['pet'].name}) — {item['reason']}")
        with cols[1].expander("Why this plan?"):
            st.text(scheduler.explain(plan))

    # --- Task list: compact rows with icon buttons ------------------------
    st.divider()
    st.subheader("Tasks")
    for pet in owner.pets:
        tasks = pet.list_tasks()
        done = len(pet.filter_tasks(status="complete", day=today))
        breed = f", {pet.breed}" if pet.breed else ""
        st.markdown(f"**{pet.name}** ({pet.species}{breed}) — {len(tasks)} task(s), {done} done today")
        if not tasks:
            st.caption("No tasks yet.")

        # Show them in the order the day will actually run, not insertion order.
        for task in scheduler.sort_by_time(tasks):
            cols = st.columns([12, 1, 1], vertical_alignment="center")
            when = f"`{task.earliest:%H:%M}` " if task.earliest else ""
            mark = "~~" if task.is_complete_on(today) else ""
            repeat = f" · ↻ {repeats_label(task)}" if task.recurrence else ""
            cols[0].markdown(f"{when}{mark}{task.summary()}{mark}{repeat}")

            if not task.is_complete_on(today):
                if cols[1].button("✓", key=f"done_{task.task_id}", help="Mark done for today"):
                    # Goes through the pet so a dated recurring task can queue
                    # up its next occurrence.
                    follow_up = pet.mark_task_complete(task.task_id, today)
                    if follow_up:
                        st.toast(f"Next {follow_up.title} due {follow_up.due_date}")
                    st.rerun()

            if cols[2].button("🗑", key=f"remove_{task.task_id}", help="Remove this task"):
                pet.remove_task(task.task_id)
                st.rerun()

    tomorrow = today + timedelta(days=1)
    due_tomorrow = scheduler.tasks_due_on(owner, tomorrow)
    if due_tomorrow:
        with st.expander(f"Due tomorrow ({len(due_tomorrow)})"):
            for pet, task in due_tomorrow:
                st.write(f"{pet.name}: {task.title}")

    # --- Forms, folded away until needed ----------------------------------
    st.divider()
    with st.expander("➕ Add a pet"):
        with st.form("add_pet", clear_on_submit=True):
            cols = st.columns([2, 1, 2])
            pet_name = cols[0].text_input("Pet name")
            pet_species = cols[1].selectbox("Species", ["dog", "cat", "other"])
            pet_breed = cols[2].text_input("Breed (optional)")

            if st.form_submit_button("Add pet"):
                if not pet_name.strip():
                    st.error("Give the pet a name.")
                else:
                    try:
                        owner.add_pet(
                            Pet(
                                name=pet_name.strip(),
                                species=pet_species,
                                breed=pet_breed.strip() or None,
                            )
                        )
                        st.toast(f"Added {pet_name.strip()}.")
                        st.rerun()
                    except ValueError as error:
                        st.error(str(error))

    with st.expander("➕ Add a care task"):
        if not owner.pets:
            st.caption("Add a pet first, then you can give it tasks.")
        else:
            with st.form("add_task", clear_on_submit=True):
                cols = st.columns([2, 3])
                task_pet_name = cols[0].selectbox("For", [pet.name for pet in owner.pets])
                task_title = cols[1].text_input("Task")

                cols = st.columns([1, 1, 2])
                task_duration = cols[0].number_input("Minutes", min_value=1, max_value=240, value=20)
                task_priority = cols[1].selectbox("Priority", ["high", "medium", "low"])
                task_notes = cols[2].text_input("Notes (optional)", placeholder="e.g. with food")

                cols = st.columns([1, 1, 2])
                repeat = cols[0].selectbox("Repeats", ["one-off", "daily", "weekly", "interval"])
                interval_days = cols[1].number_input(
                    "Every N days", min_value=1, max_value=365, value=30
                )
                weekdays = cols[2].multiselect(
                    "On these days", ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
                )

                fixed_time = st.checkbox("This task has a fixed time window")
                cols = st.columns([1, 1])
                earliest = cols[0].time_input("Not before", value=time(9, 0))
                latest = cols[1].time_input("Finish by", value=time(10, 0))

                if st.form_submit_button("Add task"):
                    if not task_title.strip():
                        st.error("Give the task a title.")
                    else:
                        pet = owner.get_pet(task_pet_name)
                        try:
                            pet.add_task(
                                CareTask(
                                    task_id=new_task_id(),
                                    title=task_title.strip(),
                                    duration_minutes=int(task_duration),
                                    priority=task_priority,
                                    notes=task_notes.strip() or None,
                                    recurrence=None if repeat == "one-off" else repeat,
                                    weekdays=set(weekdays) if weekdays else None,
                                    interval_days=int(interval_days),
                                    start_date=today if repeat == "interval" else None,
                                    earliest=earliest if fixed_time else None,
                                    latest=latest if fixed_time else None,
                                )
                            )
                            st.toast(f"Added {task_title.strip()} for {pet.name}.")
                            st.rerun()
                        except ValueError as error:
                            st.error(str(error))

    with st.expander("How the scheduler works"):
        st.markdown(
            """
Tasks from **all** of your pets compete for the same block of time. The scheduler
sorts them by priority (shortest first within a priority), fills the day until the
time runs out, and tells you what it left out and why.

- Tasks with a **time window** claim their slot first, because they cannot be moved.
- The **gap between tasks** counts against your available time, so a plan can never
  overrun the day.
- If two tasks want the same moment, you get a **warning** and one is shifted — the
  timetable itself never double-books you.
"""
        )


# ======================================================================
# New pet setup tab (agent: research -> draft -> schedule -> check -> revise)
# ======================================================================
with tab_setup:
    st.subheader("Set up a new pet")
    st.caption(
        "Describe your new pet. PawPal researches its needs, drafts a routine, runs it "
        "through the scheduler alongside your other pets, fixes anything that doesn't fit, "
        "and then asks you to approve it."
    )
    mode_banner()

    with st.form("setup_pet"):
        cols = st.columns([1, 3])
        new_name = cols[0].text_input("Name", placeholder="Nova")
        description = cols[1].text_input(
            "Describe your pet", placeholder="I just adopted a 10-week-old Siberian Husky puppy"
        )
        if st.form_submit_button("Research and plan", type="primary"):
            with st.spinner("Researching, drafting and checking the routine…"):
                st.session_state.setup_result = advisor.setup_pet(
                    new_name, description, owner, scheduler, today
                )

    result = st.session_state.setup_result
    if result is not None:
        if not result.ok:
            (st.error if result.mode == "guardrail" else st.warning)(result.message)
        else:
            profile = result.profile
            age = profile["age_years"]
            age_text = "age unknown" if age is None else ("under 1 year" if age == 0 else f"{age} years")
            st.markdown(
                f"#### Proposed routine for {result.pet.name}\n"
                f"{(profile['breed'] or profile['species']).title()} · {age_text} · "
                f"treated as **{profile['stage'].replace('-', ' ')}** · confidence "
                f"**{result.confidence:.0%}**"
            )
            st.dataframe(
                [
                    {
                        "Task": task.title,
                        "Min": task.duration_minutes,
                        "Priority": task.priority,
                        "Repeats": repeats_label(task),
                        "Window": (
                            f"{task.earliest:%H:%M}-{task.latest:%H:%M}" if task.earliest and task.latest else ""
                        ),
                    }
                    for task in result.pet.tasks
                ],
                hide_index=True,
                use_container_width=True,
            )
            if result.changes:
                st.warning(
                    "**To fit your available time I changed the first draft:**\n"
                    + "\n".join(f"- {change}" for change in result.changes)
                    + "\n\nRaise *Minutes available* in the sidebar and plan again to keep more."
                )
            for problem in result.problems:
                st.error(f"Still unresolved: {problem}")

            if result.plan:
                with st.expander("Preview: your whole day with this routine added"):
                    st.dataframe(
                        [
                            {
                                "Start": entry["start_time"].strftime("%H:%M"),
                                "Task": entry["title"],
                                "Pet": entry["pet"].name,
                                "Min": entry["scheduled_minutes"],
                            }
                            for entry in result.plan["entries"]
                        ],
                        hide_index=True,
                        use_container_width=True,
                    )
            show_sources(result.sources, title="What I looked up")
            with st.expander(f"How I got here ({len(result.trace.steps)} steps)"):
                for step in result.trace.steps:
                    st.markdown(f"**{step.step}** — {step.summary}")
            show_warnings(result.warnings)

            cols = st.columns([1, 1, 4])
            if cols[0].button("✅ Accept and add", type="primary"):
                pet = result.pet
                pet.tasks = [replace(task, task_id=new_task_id()) for task in pet.tasks]
                try:
                    owner.add_pet(pet)
                    st.session_state.setup_result = None
                    st.toast(f"Added {pet.name} with {len(pet.tasks)} tasks. See the Schedule tab.")
                    st.rerun()
                except ValueError as error:
                    st.error(str(error))
            if cols[1].button("Discard"):
                st.session_state.setup_result = None
                st.rerun()


# ======================================================================
# Care check tab (RAG gap finder)
# ======================================================================
with tab_check:
    st.subheader("Care check")
    st.caption(
        "Compares a pet's tasks with care guidance for its species, breed and age, and "
        "suggests tasks for anything missing. Nothing is added until you accept it."
    )
    mode_banner()

    if not owner.pets:
        st.info("Add a pet first.")
    else:
        cols = st.columns([2, 1, 3])
        check_name = cols[0].selectbox("Pet", [pet.name for pet in owner.pets], key="check_pet")
        check_pet = owner.get_pet(check_name)
        cols[1].write("")
        if cols[1].button("Check care", type="primary"):
            with st.spinner("Looking up care guidance…"):
                st.session_state.care_results[check_name] = advisor.care_check(check_pet, owner, scheduler)

        check = st.session_state.care_results.get(check_name)
        if check is not None:
            if not check.gaps:
                st.success(f"No gaps found in {check.pet_name}'s routine for a {check.stage.replace('-', ' ')}.")
            else:
                st.markdown(
                    f"**{len(check.gaps)} gap(s)** for {check.pet_name} "
                    f"({check.stage.replace('-', ' ')}) · confidence **{check.confidence:.0%}**"
                )
            by_id = {p.source_id: p for p in check.sources}
            origin_text = {"rules": "rule check", "ai": "AI", "agreed": "rule check + AI"}
            for index, gap in enumerate(list(check.gaps)):
                with st.container(border=True):
                    suggestion = gap.suggestion
                    st.markdown(f"**{gap.covers.title()}** — {gap.issue}")
                    st.write(
                        f"Suggested: **{suggestion.get('title')}**, {suggestion.get('minutes')} min, "
                        f"{suggestion.get('priority')} priority, {suggestion.get('repeats') or 'one-off'}"
                    )
                    st.caption(
                        "Found by " + origin_text.get(gap.origin, gap.origin) + " · sources: "
                        + ", ".join(f"[{i}] {by_id[i].label()}" for i in gap.source_ids if i in by_id)
                    )
                    if st.button("Accept", key=f"accept_gap_{check_name}_{index}"):
                        existing = {task.title.lower() for task in check_pet.tasks}
                        fields, problem = validate_task(suggestion, existing)
                        if fields is None:
                            st.error(problem)
                        else:
                            if fields.get("recurrence") == "interval":
                                fields["start_date"] = today
                            check_pet.add_task(CareTask(task_id=new_task_id(), **fields))
                            check.gaps.remove(gap)
                            st.toast(f"Added {fields['title']} for {check_pet.name}.")
                            st.rerun()
            show_sources(check.sources)
            show_warnings(check.warnings)


# ======================================================================
# Ask PawPal tab (RAG chat)
# ======================================================================
with tab_ask:
    st.subheader("Ask PawPal")
    st.caption(
        "Ask about care, health or your schedule. Answers are built from your care notes, "
        "Wikipedia and your own pets' tasks, and every fact cites its source. "
        "PawPal is not a vet."
    )
    mode_banner()
    st.caption("Try: *How much exercise does Mochi need?* · *Is chocolate toxic to dogs?* · *When is Mochi's walk today?*")

    for message in st.session_state.chat:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.write(message["text"])
                continue
            answer = message["answer"]
            if answer.kind == "emergency":
                st.error(answer.text, icon="🚨")
            elif answer.kind in ("off_topic", "empty", "too_long", "no_sources", "harmful"):
                st.warning(answer.text)
            else:
                st.markdown(answer.text)
                if answer.notice:
                    st.info(answer.notice)
                st.caption(f"{'AI' if answer.mode == 'live' else 'Offline'} · confidence {answer.confidence:.0%}")
            show_sources(answer.retrieved, {p.source_id for p in answer.sources})
            show_warnings(answer.warnings)

    question = st.chat_input("Ask about your pets…")
    if question:
        with st.spinner("Looking it up…"):
            answer = advisor.ask(question, owner, scheduler)
        st.session_state.chat.append({"role": "user", "text": question})
        st.session_state.chat.append({"role": "assistant", "answer": answer})
        st.rerun()

    if st.session_state.chat and st.button("Clear chat"):
        st.session_state.chat = []
        st.rerun()
