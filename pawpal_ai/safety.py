"""Guardrails that run before and after the model.

Before: reject empty or huge input, decline questions that are not about pet
care, and route emergencies straight to "see a vet" without asking a model.
After: strip citations that point at sources that were never retrieved, and
validate any task the AI proposes before it can reach the scheduler.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from pawpal_system import PRIORITY_ORDER, WEEKDAYS, as_time

MAX_QUESTION_CHARS = 1000
MAX_TASK_MINUTES = 240

EMERGENCY_MESSAGE = (
    "This sounds like it could be an emergency. Please contact your vet or an "
    "emergency animal clinic right now rather than waiting for advice here. If your pet "
    "may have eaten something poisonous, keep the packaging and call a vet or a pet "
    "poison helpline. Do not make your pet vomit unless a vet tells you to."
)

HEALTH_NOTICE = (
    "PawPal is not a vet and this is not a diagnosis. If you are worried about your "
    "pet's health, contact your vet."
)

# Things that are urgent no matter how they are phrased.
_URGENT = [
    r"\bseizur", r"\bconvuls", r"\bcollaps", r"\bunconscious", r"\bnot breathing\b",
    r"\b(can'?t|cannot|struggling to|trouble) breath", r"\bchoking\b",
    r"\bhit by a (car|vehicle)\b", r"\bbleeding (heavily|a lot|badly|won'?t stop)",
    r"\bwon'?t stop bleeding\b", r"\bheat ?stroke\b", r"\bbroken (leg|bone)\b",
    r"\b(can'?t|cannot|not able to) (pee|urinate)\b", r"\bbloat(ed)? .*retch",
    r"\bretching\b.*\bbloat", r"\b(vomit\w*|throw\w* up|pooping|diarrh\w*) blood",
    r"\bblood in (the |his |her |their )?(vomit|stool|poo|pee|urine)", r"\bpale gums\b",
    r"\bswollen (belly|abdomen|stomach)\b",
]

# Questions asking how to hurt an animal are refused outright.
_HARM = re.compile(
    r"\bhow (to|do i|can i|could i|would i|much \w+ (would|will|to)) "
    r"(\w+ )?(poison|kill|hurt|harm|injure|drown|starve|overdose)",
    re.IGNORECASE,
)

# AI-proposed tasks must be everyday care; medicines and doses come from a vet.
_MEDICATION = re.compile(
    r"\b(ibuprofen|paracetamol|acetaminophen|aspirin|naproxen|antibiotic\w*|steroid\w*|"
    r"dose|dosage|\d+\s?mg|tablets?|pills?|supplement\w*|medicat\w*|medicine)\b",
    re.IGNORECASE,
)

HARM_MESSAGE = "I can't help with that. If an animal is in danger, contact a vet or your local animal welfare service."

# Eating something toxic is urgent; asking whether something is toxic is not.
_INGESTED = r"\b(ate|eaten|eat some|swallowed|got into|licked|chewed|drank)\b"
_TOXIC = (
    r"\b(chocolate|grapes?|raisins?|xylitol|onions?|garlic|macadamia|antifreeze|"
    r"rat poison|poison|lil(y|ies)|ibuprofen|paracetamol|acetaminophen|naproxen|"
    r"medication|pills?|bleach|alcohol)\b"
)

HEALTH_WORDS = re.compile(
    r"\b(sick|ill\b|vomit|diarrh|symptom|pain|limp|itch|scratch|rash|infect|ears?\b|eyes?\b|"
    r"cough|sneez|letharg|appetite|not eating|weight|lump|swell|bleed|blood|medic|meds\b|"
    r"pill|dose|vaccin|flea|tick|worm|allerg|toxic|poison|vet|teeth|tooth)",
    re.IGNORECASE,
)

PET_WORDS = re.compile(
    r"\b(pets?|dogs?|cats?|pupp(y|ies)|kittens?|kitty|doggo|canine|feline|rabbit|"
    r"hamster|guinea pig|bird|parrot|fish|reptile|breed|walk|leash|lead|litter|fe(ed|d)|"
    r"food|meal|treat|kibble|vet|groom|brush|bath|nail|claw|fur|coat|shed|flea|tick|"
    r"worm|vaccin|neuter|spay|train|toilet|potty|crate|bark|meow|purr|chew|toy|play|"
    r"exercise|husky|corgi|labrador|retriever|shepherd|collie|poodle|terrier|beagle|"
    r"bulldog|spaniel|dachshund|chihuahua|pug|boxer|rottweiler|siamese|persian|"
    r"maine coon|ragdoll|bengal|sphynx|tabby|routine|schedule|task|plan|med|dose|pill)",
    re.IGNORECASE,
)


@dataclass
class SafetyCheck:
    ok: bool
    kind: str  # "ok" | "empty" | "too_long" | "harmful" | "emergency" | "off_topic"
    message: str = ""


def check_question(
    text: str, pet_names: list[str] | None = None, require_topic: bool = True
) -> SafetyCheck:
    """Decide whether a question should reach retrieval and the model at all.

    `require_topic=False` skips only the off-topic check, for inputs that are
    about a pet by definition (a new pet's description). Safety checks always run.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return SafetyCheck(False, "empty", "Please type a question about your pet's care.")
    if len(cleaned) > MAX_QUESTION_CHARS:
        return SafetyCheck(
            False,
            "too_long",
            f"Please keep questions under {MAX_QUESTION_CHARS} characters.",
        )

    lowered = cleaned.lower()
    if _HARM.search(lowered):
        return SafetyCheck(False, "harmful", HARM_MESSAGE)
    if any(re.search(pattern, lowered) for pattern in _URGENT) or (
        re.search(_INGESTED, lowered) and re.search(_TOXIC, lowered)
    ):
        return SafetyCheck(False, "emergency", EMERGENCY_MESSAGE)

    names = [name.lower() for name in (pet_names or [])]
    if require_topic and not PET_WORDS.search(lowered) and not any(name in lowered for name in names):
        return SafetyCheck(
            False,
            "off_topic",
            "I can only help with pet care: routines, feeding, exercise, grooming, "
            "health questions and your PawPal schedule.",
        )
    return SafetyCheck(True, "ok")


def is_health_question(text: str) -> bool:
    return bool(HEALTH_WORDS.search(text or ""))


CITATION = re.compile(r"\[(S\d+(?:\s*,\s*S\d+)*)\]")


def check_citations(text: str, valid_ids: set[str]) -> tuple[str, set[str], set[str]]:
    """Remove citations to sources that do not exist.

    Returns (cleaned text, valid ids cited, invalid ids that were removed).
    """
    cited: set[str] = set()
    invalid: set[str] = set()

    def fix(match: re.Match) -> str:
        ids = [part.strip() for part in match.group(1).split(",")]
        keep = [source_id for source_id in ids if source_id in valid_ids]
        cited.update(keep)
        invalid.update(source_id for source_id in ids if source_id not in valid_ids)
        return f"[{', '.join(keep)}]" if keep else ""

    cleaned = CITATION.sub(fix, text)
    cleaned = re.sub(r"[ \t]+([.,;])", r"\1", cleaned)
    return cleaned.strip(), cited, invalid


def _parse_repeats(value) -> dict:
    """'daily' | 'weekly:sat,sun' | 'interval:30' | None -> CareTask fields."""
    if value in (None, "", "none", "one-off", "once"):
        return {"recurrence": None}
    text = str(value).strip().lower()
    kind, _, detail = text.partition(":")
    if kind == "daily":
        return {"recurrence": "daily"}
    if kind == "weekly":
        days = {day.strip()[:3] for day in detail.split(",") if day.strip()}
        if days - set(WEEKDAYS):
            raise ValueError(f"unknown weekday in {value!r}")
        return {"recurrence": "weekly", "weekdays": days or None}
    if kind == "interval":
        days = int(detail)
        if days <= 0:
            raise ValueError("interval must be positive")
        return {"recurrence": "interval", "interval_days": days}
    raise ValueError(f"repeats must be daily, weekly:<days> or interval:<n>, got {value!r}")


def validate_task(raw: dict, existing_titles: set[str]) -> tuple[dict | None, str]:
    """Check an AI-proposed task. Returns (CareTask keyword args, "") or (None, reason).

    Nothing the model writes reaches the scheduler without passing through here.
    """
    if not isinstance(raw, dict):
        return None, "not an object"
    title = str(raw.get("title", "")).strip()
    if not title or len(title) > 60:
        return None, "title missing or too long"
    if title.lower() in existing_titles:
        return None, f"{title!r} already exists"
    if _MEDICATION.search(title) or _MEDICATION.search(str(raw.get("notes") or "")):
        return None, f"{title}: medicines and doses must come from your vet, not PawPal"

    try:
        minutes = int(raw.get("minutes"))
    except (TypeError, ValueError):
        return None, f"{title}: minutes must be a number"
    if not 1 <= minutes <= MAX_TASK_MINUTES:
        return None, f"{title}: minutes must be 1-{MAX_TASK_MINUTES}"

    priority = str(raw.get("priority", "")).strip().lower()
    if priority not in PRIORITY_ORDER:
        return None, f"{title}: priority must be low, medium or high"

    try:
        fields = {"title": title, "duration_minutes": minutes, "priority": priority}
        fields.update(_parse_repeats(raw.get("repeats")))
        window = raw.get("window")
        if window:
            start, _, end = str(window).partition("-")
            fields["earliest"] = as_time(start.strip())
            fields["latest"] = as_time(end.strip()) if end.strip() else None
            if fields["latest"] and fields["latest"] <= fields["earliest"]:
                return None, f"{title}: window ends before it starts"
    except ValueError as error:
        return None, f"{title}: {error}"

    notes = str(raw.get("notes") or "").strip()
    if notes:
        fields["notes"] = notes[:120]
    return fields, ""
