"""Variable-width neighbor-matching episode schedule."""

# Twenty entries encode the preregistered distribution exactly: 2/3/5/10/30-way
# with probabilities 30/25/20/15/10 percent.
VARIABLE_NM_WAYS = [2] * 6 + [3] * 5 + [5] * 4 + [10] * 3 + [30] * 2


def episode_n_way(task: str, variable_nm_way: bool):
    if task == "neighbor_matching":
        return VARIABLE_NM_WAYS if variable_nm_way else 30
    return 2
