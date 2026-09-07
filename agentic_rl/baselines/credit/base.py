"""Interface for baseline credit assigners."""

from abc import ABC, abstractmethod


class CreditAssigner(ABC):
    name = "abstract"

    def __init__(self, config=None):
        self.config = dict(config or {})
        self.delta = float(self.config.get("delta", 1e-4))

    @abstractmethod
    def compute(self, episode_group):
        """Return ``(per_episode_advantages, diagnostics)``."""
        raise NotImplementedError
