"""Shared utility functions used across all dataset preprocessing scripts."""


def normalize_user_id(val):
    """Convert any user-id value to int, or return None on failure."""
    try:
        return int(val)
    except Exception:
        return None


def normalize_handle(h):
    """Lowercase + strip a screen-name; return None for empty/sentinel values."""
    if h is None:
        return None
    s = str(h).strip().lower()
    return s if s and s not in {"nan", "none", "<na>"} else None
