"""Welford's online mean / variance.

A small numerically-stable accumulator. Used by tables that record EV
estimates over many self-play observations.

Supports `merge(other)` so collectors that store one Welford per key can
combine results across parallel workers.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Welford:
    """Online mean / sample-variance accumulator.

    Per Welford's algorithm:
        n     : count
        mean  : running mean
        M2    : running sum of squared deviations from mean
                (variance = M2 / (n-1) for sample, M2 / n for population)
    """

    n: int = 0
    mean: float = 0.0
    M2: float = 0.0

    def push(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def merge(self, other: "Welford") -> None:
        """Chan's parallel algorithm."""
        if other.n == 0:
            return
        if self.n == 0:
            self.n = other.n
            self.mean = other.mean
            self.M2 = other.M2
            return
        delta = other.mean - self.mean
        n_total = self.n + other.n
        self.mean += delta * (other.n / n_total)
        self.M2 += other.M2 + delta * delta * self.n * other.n / n_total
        self.n = n_total

    @property
    def variance(self) -> float:
        return self.M2 / self.n if self.n > 0 else 0.0

    @property
    def sample_variance(self) -> float:
        return self.M2 / (self.n - 1) if self.n > 1 else 0.0

    @property
    def stddev(self) -> float:
        return self.variance ** 0.5

    @property
    def stderr(self) -> float:
        """Standard error of the mean: sqrt(variance / n)."""
        if self.n == 0:
            return 0.0
        return (self.sample_variance / self.n) ** 0.5

    def __repr__(self) -> str:
        return (
            f"Welford(n={self.n}, mean={self.mean:.4f}, "
            f"stderr={self.stderr:.4f})"
        )


__all__ = ["Welford"]
