# Model card: PawPal Assistant

## Overview

| | |
|---|---|
| **System** | PawPal Assistant, the AI layer of PawPal+ (`pawpal_ai/`) |
| **Built on** | PawPal+ (Module 2), a rule-based pet-care scheduler |
| **Model** | Google Gemini, free tier, via `google-genai` (default `gemini-3.8-flash`, set with `PAWPAL_MODEL`); temperature 0.2. Runs in offline mode with no model when there is no key. |
| **Techniques** | Retrieval-augmented generation over three sources; an agentic setup workflow (draft → schedule → check → AI revise → rule-based safety net); rule-based checks and a test/evaluation harness |
| **Data sources** | The user's own pets, tasks and plan; `knowledge/*.md` (short care notes written for this project); Wikipedia article introductions |
| **Intended use** | Everyday pet-care planning and general care information for dog and cat owners |
| **Not for** | Diagnosis, treatment, medication or dosing decisions, or emergencies. The app sends those to a vet. |

## Limitations and biases

- **The knowledge base is small and general.** The care notes are about thirty short
  sections, written in plain English from widely accepted guidance. They focus on dogs and
  cats. Rabbits, birds, reptiles and other pets get only a generic "other" routine. The
  notes reflect common Western pet-keeping norms (for example indoor cats and leash
  walking), which don't fit everywhere.
- **The notes were drafted with an AI assistant** and checked by me, not by a vet. They
  point to a vet for anything medical, but they are not veterinary-reviewed.

- **Breed generalisations.** "High-energy breeds need 60+ minutes" is a population-level
  rule. An individual dog may be older, injured or just calmer, and the care check will
  still flag it. Wikipedia breed intros describe breed history more than care.
- **Age is whole years only.** The original `Pet` model stores `age_years` as an integer,
  so a 10-week puppy and an 11-month adolescent are both "under a year old" and get the same
  routine.
- **The agent only fixes what the scheduler can see.** It detects tasks that don't fit,
  clashes, and other pets' tasks being pushed out. It can't judge care quality. In testing,
  fitting a husky puppy into a 120-minute day made it **drop the puppy's only walk** and
  keep the high-priority meals, because the rule-based fixer drops the lowest-priority
  tasks first. The app shows this in a yellow "changes" box, but it is a real weakness.
- **Keyword guardrails are blunt.** Emergency, harm and off-topic detection are regular
  expressions in English. They miss unusual phrasing (an emergency described without any of
  the listed words) and can over-trigger. A question about a pet species not in the word
  list (say, "my gecko") can be declined as off-topic in Ask PawPal. This already
  happened once in testing with "domesti shorthair" (see below).
- **Offline answers are quotes, not explanations.** Without a model, answers are the
  best-matching sentences from the sources. They are accurate to the sources, but can be
  choppy, and can include a closely related fact (a cat fact in a dog answer).
- **The confidence score is a heuristic,** not a calibrated probability. It combines
  evidence strength (50%), citation validity (30%) and source diversity (20%).
- **The live model needed a safety net.** The automated tests use a scripted fake model,
  so I also ran the evaluator against real Gemini (`python evaluate_ai.py --live`,
  `gemini-3.5-flash-lite`, 2026-11-22). The first run scored **9/10, average confidence
  0.70**. Everything passed except the new-pet setup for a husky puppy. Gemini gave
  "Puppy breakfast" a time window ending at 08:00, the exact minute the owner's day
  starts, so it could never fit. Its revisions also pushed Mochi's morning walk out of the
  day. After its two revision rounds, 2 problems were left; the agent's check caught and
  reported them (confidence dropped to 0.40) instead of hiding them. Two fixes followed:
  the prompt now tells the model when the day starts and what is already booked, and if
  the AI's revisions still leave problems, the rule-based fixes run once as a safety net.
  A test replays the failure.

  The second live run was **9/10 again, average confidence 0.73**, but for a different
  reason, which the saved trace made easy to see. This time Gemini drafted six tasks, all
  high priority (three meals and three toilet breaks), with gaps between them needing 85
  minutes when only 70 were free. The safety net ran but couldn't act: its rules only
  shortened or dropped *medium and low* priority tasks, so nothing changed and Mochi's
  walk was still pushed out (1 problem, confidence 0.68). I added a last-resort step:
  shorten essential tasks (never below 5 minutes, always shown to the owner) rather than
  push another pet's care out of the day. If even that can't fit, the problem is reported
  instead of cutting further. Two more tests replay this run.

  **After both fixes, the third live run passed 10/10, average confidence 0.77.** The
  trace shows how: Gemini's own two revisions *still* pushed Mochi's walk out, and it was
  the safety net that resolved it, by shortening the puppy's six essential tasks from 10 to
  6 minutes each. So the AI alone did not solve the hardest case; the AI plus a
  deterministic check did. One honest caveat: that run still reported confidence 0.90,
  because the score doesn't yet count "the safety net had to step in" against the AI.
- **Privacy.** In live mode, pet names, tasks and questions are sent to Google. Google's
  terms allow free-tier data to be used to improve its products. Offline mode sends nothing
  except short search terms to Wikipedia.

## Misuse and how it is prevented

| Risk | Prevention in the code |
|---|---|
| Treating PawPal as a vet | Emergencies are detected before any model call and answered with "contact a vet now". Health answers carry a "not a vet, not a diagnosis" notice. Prompts forbid diagnosis. |
| AI recommending medicines or doses | Prompts forbid it, **and** `validate_task` rejects any AI-proposed task mentioning medicines, tablets, doses or `mg`, so a model that ignores its prompt still can't add one. |
| Asking how to hurt an animal | "How do I poison/kill/hurt…" style questions are refused before retrieval. |
| Made-up facts | The model only sees numbered sources. Citations are checked in code: invented ones are removed, and an answer with no valid citation is replaced by the sources themselves. |
| Prompt injection through web text | Sources are labelled as reference material, not instructions. Nothing the model writes changes the schedule without validation **and** a human pressing Accept. |
| Over-using Wikipedia | Results are cached, requests are spaced one second apart, and there's a 60-second cool-down after a rate-limit response. |
| Leaking the API key | The key lives in a git-ignored `.env`, is never logged, and the checker never prints it. |

## What surprised me while testing

- **Tests passing didn't mean the answers were good.** All the new tests passed on the
  first run. Reading actual outputs revealed problems the tests couldn't see: an offline
  answer about an itchy ear included an unrelated poisoning tip, and the care check cited
  *Dental care* as the source for a *grooming* gap (that note mentions "brushing" teeth).
  Both were fixed by tightening how passages and sentences are chosen.
- **The free web source pushed back.** Wikipedia returned HTTP 429 (too many requests)
  after a burst of test queries. The app kept answering from the local notes, which showed
  the fallback works, but it led to adding caching and a cool-down.
- **Small text details matter.** "Huskies" didn't match "husky" until the word-stemmer
  handled `-ies` endings.
- **My own first real test broke a guardrail.** When I tried the New pet setup tab with a
  kitten, I typed "3 month old domesti shorthair" (with a typo, and without the word
  "cat"). The off-topic guardrail looks for pet words like "cat" or "kitten", found none,
  and refused it with "I can only help with pet care". The automated tests had all passed,
  because they only used tidy inputs like "a 3 year old cat". Two fixes followed: the
  setup tab no longer runs the off-topic check, since a description of a new pet is about
  a pet by definition, while the safety checks for emergencies and harm still run; and the
  offline parser now reads "shorthair", "longhair", "DSH" and "moggy" as a cat. I added my
  exact input as a test so the bug can't come back. The lesson: real people type breed
  names, abbreviations and typos that a keyword list will never fully cover.
- **The agent's "successful" fix wasn't always the right one.** It made the puppy's
  routine fit by dropping the walk, which is technically correct but not what a good owner
  would choose.
- **Evaluation summary (offline):** 10/10 cases passed, average confidence 0.73, with 100%
  word overlap when the same question was asked twice.

## Collaboration with AI

I built this with Claude (Claude Code in VS Code) as a coding assistant. I set the
direction and made the decisions. Claude proposed ideas and plans, wrote most of the code
and tests, and ran them.

**Suggestions I accepted**
- Adding the AI as **tabs** next to the existing scheduler instead of a single chatbot, so
  that anything changing the schedule goes through review cards and an Accept button.
- An **offline mode**, so the app, tests and evaluator run for anyone without a key.
- Adding a **consistency check** to the evaluator after comparing the plan against the
  assignment requirements.

**Suggestions I rejected or changed**
- **The direction of the project was mine; the plan was a collaboration.** Claude's first
  plan was a single RAG question-answering advisor. I rejected it, because I wanted to
  explore what the project could become before committing to anything. I asked for ideas
  instead and chose to combine a new-pet setup assistant with a care-gap checker and
  health advice, built on the scheduler I already had. I also decided it should be tabs
  rather than one chatbot. We then worked out the plan together, and I checked it against
  the assignment's requirements before approving it.
- Claude recommended the **Claude API** as the model. I chose a free option instead
  (Gemini's free tier) because the project needed to cost nothing.

**What I learned about working with an AI assistant:**

- **It moves fast, so I have to steer.** Claude would happily fill a file or commit work
  before I'd decided what I wanted. I learned to say what I wanted first, ask for options
  before plans, and say clearly when something should wait. For example, I had it undo
  its commits so I could set up Gemini and make my own edits first.
- **Its confidence isn't proof.** It wrote 100+ tests that all passed, and the app still
  broke on my first real input. My own testing in the browser found things its tests
  didn't: the "domesti shorthair" refusal, a layout that hid the schedule, and a name field
  that was never used. Using the app myself mattered as much as the test suite.
- **It's most useful when I give it real evidence.** Pasting real terminal output,
  screenshots and error messages, instead of describing problems, led to precise fixes:
  the wrong Python version, a retired model name, and the setup agent's failures, which
  it traced from the logs.
