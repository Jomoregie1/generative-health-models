# ---- constants (tune if you like) ----
MIN_HEAD_SKIP   = 2        # never fewer than 2 head steps
ALPHABAR_FLOOR  = 5e-4     # ensures MIN_HEAD_SKIP given a typical ᾱ grid


# ---- tiny helpers ----
def _first_index_where_ge(sorted_array, value):
    """
    Return the smallest index i such that sorted_array[i] >= value.
    Expects the array to be sorted in ascending order.
    """
    try:
        import numpy as np
        return int(np.searchsorted(np.asarray(sorted_array), float(value), side="left"))
    except Exception:
        # pure-Python fallback
        from bisect import bisect_left
        return int(bisect_left(sorted_array, float(value)))


def schedule_head_skip(ddim_alpha_bar_start, timesteps_alpha_bar):
    """
    Enforce an ᾱ floor and a minimum head-skip.

    Parameters
    ----------
    ddim_alpha_bar_start : float
        Your current ᾱ-start threshold (may be 0 or very small).
    timesteps_alpha_bar : 1D array-like (ascending)
        ᾱ values for each *kept* inference step, sorted ascending along your
        inference order. If yours is descending, reverse it before passing.

    Returns
    -------
    ab_start : float
        The floored ᾱ start actually used.
    head_skip : int
        Number of initial steps to drop, clamped to [MIN_HEAD_SKIP, len(grid)-1].
    """
    # Respect an explicit floor in alpha_bar space
    ab_start = max(float(ddim_alpha_bar_start), float(ALPHABAR_FLOOR))

    n = len(timesteps_alpha_bar)
    if n == 0:
        # Degenerate, but stay safe
        return ab_start, MIN_HEAD_SKIP

    # Recompute first kept index with the floored alpha_bar
    first_keep = _first_index_where_ge(timesteps_alpha_bar, ab_start)

    # Clamp so we never skip *all* steps, and never fewer than MIN_HEAD_SKIP
    max_head_skip = max(0, n - 1)  # leave at least 1 step to run
    head_skip = max(MIN_HEAD_SKIP, min(first_keep, max_head_skip))
    return ab_start, head_skip