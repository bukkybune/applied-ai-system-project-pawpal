# AI Interactions Log

> **Stretch features only.** Only fill in the sections that apply to stretch features you attempted. If you did not attempt a stretch feature, leave its section blank or delete it. This file is not required for the core project.

---

## PawPal setup agent: intermediate reasoning traces (Agentic Workflow Enhancement)

The **New pet setup** tab runs a multi-step agent (`PawPalAdvisor.setup_pet` in
[pawpal_ai/advisor.py](pawpal_ai/advisor.py)). Its decision chain is:

1. **parse**: read species, breed and age from the owner's description (model in live
   mode, rules offline)
2. **research**: retrieve care notes, Wikipedia and the routine template for that life
   stage
3. **draft**: propose tasks, each tied to source ids
4. **schedule (tool call)**: run the real `Scheduler` on a copy of the household with the
   new pet added
5. **check**: find tasks left out, clashes, and other pets' tasks pushed out of the day
6. **revise**: fix the problems (model in live mode, rules offline), up to 2 rounds, then
   schedule and check again
7. **result**: the routine, every change made, any unresolved problems, and a confidence
   score, for the human to accept or discard

Every run is appended to `logs/agent_traces.jsonl`. Two committed sample runs are in
[docs/agent_traces_sample.jsonl](docs/agent_traces_sample.jsonl). One of them, reformatted:

**Goal:** set up Nova: "I just adopted a 10-week-old Siberian Husky puppy". Household:
Mochi the Corgi with a 30-minute walk and breakfast; 120 minutes available; 5-minute gaps;
offline mode, Wikipedia on.

```
parse     Understood: Siberian Husky, stage dog puppy
          profile = {species: dog, breed: Siberian Husky, age_years: 0, parsed_by: rules}

research  Retrieved 7 source(s)
          S1 Care notes: High-energy dog breeds     S5 Care notes: Daily needs of adult dogs
          S2 Care notes: Feeding young animals      S6 Wikipedia: Siberian Husky
          S3 Care notes: Toilet training puppies    S7 Care notes: Suggested routine (dog-puppy)
          S4 Care notes: Dental care for dogs

draft     Drafted 8 task(s) (template)              free_minutes = 75
          Puppy breakfast 10 [high] 07:00-08:30 · Puppy lunch 10 [high] 12:00-13:30 ·
          Puppy dinner 10 [high] 17:00-18:30 · Toilet break after meals 10 [high] ·
          Short gentle walk 15 [medium] · Training session 10 [high] ·
          Socialisation and play 15 [medium] · Handling practice 5 [medium]

schedule  Ran the scheduler: 7 slot(s) for the day  minutes_used = 120 of 120

check     3 problem(s)
          new_task_left_out: Handling practice  (needs 5 min, only 0 min left)
          new_task_left_out: Short gentle walk  (needs 15 min, only 0 min left)
          new_task_left_out: Socialisation and play

revise    Round 1: rule-based fixes
          # (annotation, not in the log) the rules shorten medium tasks by a third,
          # then drop the lowest-priority ones, keeping high-priority care

schedule  Ran the scheduler: 7 slot(s) for the day  minutes_used = 120 of 120
check     No problems found

changes   dropped Short gentle walk (15 min) to fit your time
          dropped Socialisation and play (15 min) to fit your time
          dropped Handling practice (paws, ears, teeth) (5 min) to fit your time

result    5 task(s) proposed, 0 unresolved problem(s), confidence 0.80
```

**What the trace shows:** the agent did not trust its first draft. It tested the draft with
the scheduler, found three tasks that could not fit, revised, and tested again before
showing anything. It also shows the agent's limit: it fixed the *time* problem by dropping
the puppy's only walk. The trace and the app's "changes" box make that visible, so the
owner can raise their available minutes instead.

### A live run (Gemini on), with typos and a busy model

**Goal:** set up titi: "7 moth old chuwawa" (typed quickly, with two typos). Live mode,
`gemini-3.5-flash-lite`, Wikipedia on, 2026-11-19.

```
parse     Understood: Chihuahua, stage dog puppy                       t=10.5s
          profile = {species: dog, breed: Chihuahua, age_years: 0, parsed_by: ai}

research  Retrieved 7 source(s)                                         t=10.9s
          S1 Feeding young animals   S4 Daily needs of adult dogs   S6 Wikipedia: Chihuahua (state)
          S2 Toilet training puppies S5 Exercise for puppies        S7 Suggested routine (dog-puppy)
          S3 Dental care for dogs

draft     AI draft rejected (Gemini API error 503: UNAVAILABLE, "This model is
          currently experiencing high demand"); falling back to template  t=34.2s
draft     Drafted 8 task(s) (template)                 free_minutes = 120

schedule  Ran the scheduler: 8 slot(s) for the day     minutes_used = 120 of 120
check     No problems found
result    8 task(s) proposed, 0 unresolved problem(s), confidence 0.80
```

**What this live trace shows, compared with the offline one above:**

- **The model is better at messy input.** "7 moth old chuwawa" has two typos. The model
  turned it into *Chihuahua, under a year, puppy* (`parsed_by: ai`). The offline rules would
  not have matched "moth" as months or "chuwawa" as a breed, so they would have produced an
  unknown "other" pet.
- **The fallback did its job.** Gemini's free tier was overloaded, so drafting failed with
  a 503 error. Instead of crashing or showing nothing, the agent logged why and used the
  routine template, and the result still went through the scheduler and the check.
- **It exposed a retrieval bug.** Source S6 was *Chihuahua (state)*, the Mexican state,
  not the dog. Searching Wikipedia for a breed name on its own is ambiguous. The fix:
  searches now add the species ("Chihuahua dog"), which finds *Chihuahua (dog breed)*,
  and a test checks this.
- **It was slow.** About 34 seconds end to end, almost all of it waiting on the busy free
  tier. That's why the Gemini client now gives up after 45 seconds with one retry instead
  of hanging.

---

## Agent Workflow (SF7)

> Document your experience using an AI agent (e.g., Cursor Agent, Claude, Copilot) to make multi-step changes autonomously.

**What task did you give the agent?**

I used Claude (Claude Code in VS Code) to turn PawPal+ into an applied-AI project. After
choosing the direction (new pet setup assistant, care-gap checker and care advice, as
tabs, on a free model), I asked it to carry out the approved plan.

**What did the agent do?**

- Read `pawpal_system.py` and `app.py` to reuse the existing scheduler instead of
  rewriting it.
- Created the `pawpal_ai/` package (retrieval, safety, routines, advisor, llm, trace,
  check) and the `knowledge/` care notes.
- Wrapped the app in tabs and added the three AI tabs.
- Wrote `tests/test_ai.py` and `tests/test_app.py` with a fake model and fake Wikipedia,
  and ran the suite after each change.
- Installed `google-genai` and checked the client code against the installed SDK's real
  signatures.
- Built `evaluate_ai.py`, then drafted the README, the model card and this file.
- Committed in steps. I had those commits undone, because I wanted to set up Gemini and
  make my own edits before anything was committed.

**What did you have to verify or fix manually?**

- The agent filled in an architecture diagram before I asked for one; I had it emptied.
- Its first plan wasn't what I wanted; I asked for ideas before any plan.
- I checked the plan against the assignment's requirements before approving it, which led
  to adding a consistency check and clearer setup steps.
- Things the agent caught and fixed itself after reading real outputs (not tests):
  Wikipedia rate-limiting (HTTP 429), a care-check gap citing the wrong note, and
  "huskies" not matching "husky".

**What I checked myself, in the browser with my own key:**

  - **A kitten was refused as "off-topic".** My first real test of New pet setup was
    "3 month old domesti shorthair". The app answered "I can only help with pet care",
    because the off-topic guardrail looks for words like "cat" or "kitten" and found none.
    All the automated tests had passed, because they only used tidy inputs. I reported it,
    and it was fixed: the setup tab no longer runs the off-topic check, while the emergency
    and harm checks still run. "Shorthair", "longhair", "DSH" and "moggy" now count as cat,
    and my exact input became a test.
  - **The Schedule tab was the wrong way round.** The add-pet and add-task forms took up
    most of the screen, and the actual timetable was at the bottom behind a button. The
    Done/Remove buttons were also squashed into one letter per line. I asked for a
    schedule-first layout: the timetable is now at the top and updates on its own, the
    task list uses small ✓/🗑 buttons, and the forms are folded away.
  - **The owner's name was asked for but never used.** The sidebar asks for your name, but
    nothing in the app used it. Now the Schedule tab greets you ("👋 Hi Dorcas, here's
    today's schedule"), with the date, your available minutes and your pets underneath.
  - **Live mode setup.** Getting Gemini running needed a newer Python than my Mac's
    built-in one, a different model name (Google had retired `gemini-2.5-flash` for new
    users), and a lighter model (`gemini-3.5-flash-lite`) while the others were overloaded.
  - **My live trace revealed a bug.** When Claude reviewed the trace from my live run, it
    noticed that Wikipedia had returned the Mexican state for "Chihuahua". Searches now
    include the species.
