"""Storage helpers for self-play tables.

Standardizes on pickle (protocol 4) for all persistent artifacts. Phase 6
table builders should use these helpers so the on-disk format is uniform.

Layout convention (recommended):

    artifacts/
        <run_name>/
            metadata.pkl            # dict: timestamp, n_games, seed, policy descriptions
            <collector_name>.pkl    # one file per collector

`load_table` works for any pickle file but provides a stable name-based
lookup for the recommended layout.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Iterable, Mapping

from .collectors import Collector


PROTOCOL = 4


def save_collectors(
    collectors: Iterable[Collector],
    out_dir: Path | str,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Save each collector's `result()` to `<name>.pkl` under `out_dir`.

    Also writes `metadata.pkl` if provided. Creates the directory.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if metadata is not None:
        with (out / "metadata.pkl").open("wb") as f:
            pickle.dump(dict(metadata), f, protocol=PROTOCOL)
    for c in collectors:
        with (out / f"{c.name}.pkl").open("wb") as f:
            pickle.dump(c.result(), f, protocol=PROTOCOL)


def save_collector(c: Collector, path: Path | str) -> None:
    """Save a single collector's result to an explicit path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        pickle.dump(c.result(), f, protocol=PROTOCOL)


def load_table(path: Path | str) -> Any:
    """Load a single pickled table."""
    with Path(path).open("rb") as f:
        return pickle.load(f)


def load_run(out_dir: Path | str) -> dict[str, Any]:
    """Load every `*.pkl` under `out_dir` into a dict keyed by stem."""
    out = Path(out_dir)
    tables: dict[str, Any] = {}
    for f in sorted(out.glob("*.pkl")):
        tables[f.stem] = load_table(f)
    return tables


__all__ = ["save_collectors", "save_collector", "load_table", "load_run"]
