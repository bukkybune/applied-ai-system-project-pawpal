"""Routine templates and rule-based care-gap detection.

knowledge/routines.md holds a starting routine per profile (dog-adult,
cat-kitten, ...). Comparing a pet's real tasks against its profile gives a
deterministic list of gaps that works with or without a language model, and
that the model's own suggestions are checked against.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from pawpal_system import Pet

from .retrieval import KNOWLEDGE_DIR, tokenize

# Words in a task title that show which care area it covers.
COVERS_KEYWORDS = {
    "feeding": {"breakfast", "dinner", "lunch", "supper", "meal", "feed", "food", "kibble"},
    "exercise": {"walk", "run", "jog", "hike", "fetch", "exercise", "park", "play", "swim"},
    "enrichment": {"play", "train", "puzzle", "game", "enrich", "sniff", "toy", "brain"},
    "dental": {"teeth", "tooth", "dental", "handl"},
    "grooming": {"brush", "groom", "bath", "comb", "coat", "nail"},
    "parasite": {"flea", "tick", "worm", "parasite", "deworm"},
    "litter": {"litter"},
    "water": {"water", "fountain"},
    "toilet": {"toilet", "potty", "outside"},
    "training": {"train", "obedience", "class"},
    "health": {"check", "weigh", "weight", "vet", "mobility", "lump", "appetite"},
    "habitat": {"habitat", "cage", "tank", "hutch", "clean"},
}

HIGH_ENERGY_TARGET = 60  # minutes of exercise a day, from knowledge/dogs.md

# Recognised in free text when there is no model to parse it. Longest first so
# "siberian husky" wins over "husky".
KNOWN_BREEDS = sorted(
    [
        "siberian husky", "husky", "corgi", "pembroke welsh corgi", "labrador retriever",
        "labrador", "golden retriever", "german shepherd", "border collie",
        "australian shepherd", "belgian malinois", "poodle", "beagle", "bulldog",
        "french bulldog", "dachshund", "chihuahua", "pug", "boxer", "rottweiler",
        "jack russell", "vizsla", "weimaraner", "dalmatian", "german shorthaired pointer",
        "shih tzu", "yorkshire terrier", "cocker spaniel", "greyhound", "great dane",
        "siamese", "persian", "maine coon", "ragdoll", "bengal", "sphynx",
        "british shorthair", "tabby",
    ],
    key=len,
    reverse=True,
)


@dataclass
class Gap:
    """A care area the pet's routine is missing, with a task that would fill it."""

    covers: str
    issue: str
    suggestion: dict
    source_ids: list[str] = field(default_factory=list)
    origin: str = "rules"  # "rules" | "ai"


def parse_routines(path: Path = KNOWLEDGE_DIR / "routines.md") -> dict[str, list[dict]]:
    """Read every `## profile` section of routines.md into task dicts."""
    routines: dict[str, list[dict]] = {}
    current = None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            routines[current] = []
        elif current and line.startswith("- ") and "|" in line:
            parts = [part.strip() for part in line[2:].split("|")]
            if len(parts) < 5:
                continue
            title, minutes, priority, repeats, covers = parts[:5]
            routines[current].append(
                {
                    "title": title,
                    "minutes": int(minutes),
                    "priority": priority,
                    "repeats": repeats,
                    "covers": covers,
                    "window": parts[5] if len(parts) > 5 and parts[5] else None,
                }
            )
    return routines


def high_energy_breeds(path: Path = KNOWLEDGE_DIR / "dogs.md") -> set[str]:
    match = re.search(r"<!--\s*high-energy:(.*?)-->", Path(path).read_text(encoding="utf-8"))
    if not match:
        return set()
    return {name.strip().lower() for name in match.group(1).split(",") if name.strip()}


def life_stage(species: str, age_years: int | None) -> str:
    species = (species or "other").lower()
    if species not in ("dog", "cat"):
        return "other"
    if age_years is None:
        return f"{species}-adult"
    if age_years < 1:
        return f"{species}-puppy" if species == "dog" else "cat-kitten"
    senior_from = 8 if species == "dog" else 11
    return f"{species}-senior" if age_years >= senior_from else f"{species}-adult"


def is_high_energy(pet: Pet) -> bool:
    if pet.species != "dog" or not pet.breed:
        return False
    breed = pet.breed.lower()
    return any(name in breed for name in high_energy_breeds())


def task_covers(title: str) -> set[str]:
    """Which care areas a task title looks after."""
    tokens = set(tokenize(title))
    areas = set()
    for area, keywords in COVERS_KEYWORDS.items():
        if any(token.startswith(keyword) for token in tokens for keyword in keywords):
            areas.add(area)
    return areas


def exercise_target(pet: Pet, template: list[dict]) -> int:
    planned = sum(item["minutes"] for item in template if item["covers"] == "exercise")
    if is_high_energy(pet) and life_stage(pet.species, pet.age_years) == "dog-adult":
        return max(planned, HIGH_ENERGY_TARGET)
    return planned


def find_gaps(pet: Pet, routines: dict[str, list[dict]] | None = None) -> list[Gap]:
    """Compare a pet's tasks with the template for its life stage."""
    routines = routines or parse_routines()
    stage = life_stage(pet.species, pet.age_years)
    template = routines.get(stage, routines.get("other", []))

    covered: dict[str, int] = {}  # area -> minutes of tasks covering it
    for task in pet.tasks:
        for area in task_covers(task.title):
            covered[area] = covered.get(area, 0) + task.duration_minutes
    existing = {task.title.lower() for task in pet.tasks}

    gaps: list[Gap] = []
    seen_areas: set[str] = set()
    for item in template:
        area = item["covers"]
        if area in seen_areas or area in covered or area == "exercise":
            continue
        seen_areas.add(area)
        if item["title"].lower() in existing:
            continue
        gaps.append(
            Gap(
                covers=area,
                issue=f"No {area} task in {pet.name}'s routine.",
                suggestion=dict(item),
            )
        )

    # Exercise is judged by minutes, not just presence.
    target = exercise_target(pet, template)
    have = covered.get("exercise", 0)
    if target and have < target:
        shortfall = target - have
        reason = " (a high-energy breed)" if is_high_energy(pet) else ""
        template_walks = [item for item in template if item["covers"] == "exercise"]
        base = next(
            (item for item in template_walks if item["title"].lower() not in existing),
            template_walks[0] if template_walks else None,
        )
        title = base["title"] if base and base["title"].lower() not in existing else "Extra play or run"
        gaps.append(
            Gap(
                covers="exercise",
                issue=(
                    f"{pet.name} gets {have} min of exercise a day; about {target} min is "
                    f"suggested for a {stage.replace('-', ' ')}{reason}."
                ),
                suggestion={
                    "title": title,
                    "minutes": min(max(shortfall, 10), 60),
                    "priority": "medium" if have else "high",
                    "repeats": "daily",
                    "covers": "exercise",
                    "window": None,
                },
            )
        )
    return gaps


def routine_text(stage: str, routines: dict[str, list[dict]] | None = None) -> str:
    """The template as a citable passage."""
    routines = routines or parse_routines()
    items = routines.get(stage, [])
    lines = [
        f"{item['title']}: {item['minutes']} min, {item['priority']} priority, "
        f"{item['repeats']} ({item['covers']})"
        for item in items
    ]
    return f"Suggested starting routine for a {stage.replace('-', ' ')}: " + "; ".join(lines) + "."
