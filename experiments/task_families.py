"""Small semantic task vocabulary used by task-conditioned pilot models."""

TASK_FAMILIES = (
    "unknown",
    "neighbor_matching",
    "political_leaning",
    "bot_detection",
    "page_category",
    "account_suspension",
)
TASK_FAMILY_TO_ID = {name: idx for idx, name in enumerate(TASK_FAMILIES)}

CLASSIFICATION_FAMILY_BY_DATASET = {
    "covid_political": "political_leaning",
    "election2020": "political_leaning",
    "facebook_page_reference": "page_category",
    "twibot20": "bot_detection",
    "ukr_rus_suspended": "account_suspension",
}


def resolve_task_family(task_name: str, dataset: str) -> str:
    if task_name == "neighbor_matching":
        return "neighbor_matching"
    if task_name == "classification":
        return CLASSIFICATION_FAMILY_BY_DATASET.get(dataset, "unknown")
    return "unknown"


def parse_seen_families(value) -> set[str]:
    if isinstance(value, str):
        families = {item.strip() for item in value.split(",") if item.strip()}
    else:
        families = set(value or ())
    unknown = families - set(TASK_FAMILIES)
    if unknown:
        raise ValueError(f"Unknown task families: {sorted(unknown)}")
    return families


def effective_task_family(task_name: str, dataset: str, seen_families) -> str:
    family = resolve_task_family(task_name, dataset)
    seen = parse_seen_families(seen_families)
    return family if family in seen else "unknown"
