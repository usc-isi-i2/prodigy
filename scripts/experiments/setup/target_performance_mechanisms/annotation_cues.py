"""Published-cue audit, not a political classifier or annotation reconstruction."""
import re


SOURCE_URL = "https://arxiv.org/pdf/2207.08349v4"
# Retweet-BERT, Table 3, visually checked. Only set membership is used; we do
# not adopt the table's political assignments (including its BackTheBlue row).
PUBLISHED_CUES = (
    "Resist", "FBR", "TheResistance", "Resistance", "Biden2020", "VoteBlue",
    "VoteBlueNoMatterWho", "Bernie2020", "BlueWave", "BackTheBlue",
    "NotMyPresident", "NeverTrump", "Resister", "VoteBlue2020", "ImpeachTrump",
    "BlueWave2020", "YangGang", "MAGA", "KAG", "Trump2020", "WWG1WGA",
    "QAnon", "Trump", "KAG2020", "Conservative", "BuildTheWall", "AmericaFirst",
    "TheGreatAwakening", "TrumpTrain",
)
DETECTORS = ("original_hashtag", "original_or_cleaned_surface")
SUBSETS = (
    "all", "query_center_absent", "query_center_present", "query_subgraph_absent",
    "query_subgraph_present", "supports_absent", "supports_present", "all_input_absent",
)


def cue_flags(raw_profile, cleaned_profile):
    """Ignore labels and predictions; original hashtags plus a broader sensitivity.

    The released cleaned profiles have no hash signs. Digits can also be absent,
    so the secondary detector accepts both literal tokens and digit-deleted
    aliases from the SAME fixed published vocabulary. Not all political language
    is covered; a match need not indicate the route that generated the label.
    """
    lexicon = {x.casefold() for x in PUBLISHED_CUES}
    aliases = lexicon | {re.sub(r"[0-9]", "", x) for x in lexicon}
    tags = {m.casefold() for m in re.findall(r"(?<!\w)[#＃](\w+)", str(raw_profile))}
    words = {m.casefold() for m in re.findall(r"\w+", str(cleaned_profile))}
    original = bool(tags & lexicon)
    return original, original or bool(words & aliases)


def join_original_profiles(small, full):
    """Every input row must have one exact complete-row match, not fuzzy bio text."""
    import numpy as np
    if "raw_profile" not in full or "profile" not in small:
        raise ValueError("Original and cleaned profiles required")
    columns = list(small.columns)
    if not set(columns) <= set(full.columns):
        raise ValueError("Original table does not contain all released fields")
    if full.duplicated(columns).any():
        raise ValueError("Ambiguous full-row provenance join")
    joined = small.assign(_audit_row=np.arange(len(small))).merge(
        full[columns + ["raw_profile"]], on=columns, how="left", sort=False,
        validate="many_to_one", indicator=True,
    ).sort_values("_audit_row")
    if len(joined) != len(small) or not joined._merge.eq("both").all():
        raise ValueError("Incomplete original-profile provenance join")
    return joined.raw_profile.tolist()


def subset_masks(center_flags, subgraph_flags, support_flags):
    import numpy as np
    c, q, s = [np.asarray(x, dtype=bool) for x in (center_flags, subgraph_flags, support_flags)]
    if c.shape != q.shape or c.shape != s.shape or (c & ~q).any():
        raise ValueError("Invalid cue containment or occurrence alignment")
    return dict(zip(SUBSETS, (np.ones(c.shape, bool), ~c, c, ~q, q, ~s, s, ~q & ~s)))
