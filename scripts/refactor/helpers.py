
# ---------------------------------------------------------------------------
# CPU budget & parallelization helpers
# ---------------------------------------------------------------------------

def _get_cpu_budget(total_cpus=None):
    """Determine available CPU count from env vars or os.cpu_count().

    Checks ``SLURM_CPUS_PER_TASK`` and ``OMP_NUM_THREADS`` before falling
    back to :func:`os.cpu_count`.
    """
    if total_cpus is not None:
        return total_cpus
    for var in ("SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS"):
        val = os.environ.get(var)
        if val and val.isdigit():
            return int(val)
    return os.cpu_count() or 1


def _set_estimator_njobs(estimator, n_jobs):
    """Set ``n_jobs`` on an estimator that supports the parameter."""
    try:
        params = estimator.get_params()
    except Exception:
        return
    if "n_jobs" in params:
        estimator.set_params(n_jobs=n_jobs)


def _suppress_fold_warnings():
    """Suppress expected, harmless warnings from fold-local feature ops."""
    import warnings as _w
    _w.filterwarnings(
        "ignore", message=r"Features .* are constant", category=UserWarning,
    )
    _w.filterwarnings(
        "ignore", message=r"invalid value encountered in divide",
        category=RuntimeWarning,
    )
    _w.filterwarnings(
        "ignore", message=r"X does not have valid feature names",
        category=UserWarning,
    )
