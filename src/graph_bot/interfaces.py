from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from .datatypes import ReasoningNode, ReasoningTree


class AbstractValidator(ABC):
    """
    Abstract base class for reasoning validators.
    Decouples specific logic (Game24, Math, Code) from the core loop.
    """

    @abstractmethod
    def validate(self, node: ReasoningNode) -> float:
        """
        Validate a reasoning node and return a score (0.0 to 1.0).
        """
        pass

    @abstractmethod
    def get_failure_reason(self, node: ReasoningNode) -> Optional[str]:
        """
        Return a human-readable reason for validation failure, if any.
        """
        pass


class AbstractDistiller(ABC):
    """
    Abstract base class for knowledge distillers.
    Handles compression of traces into reusable templates.
    """

    @abstractmethod
    def distill_trace(self, tree: ReasoningTree) -> List[ReasoningNode]:
        """
        Extract reusable template nodes from a successful reasoning tree.
        """
        pass

    @abstractmethod
    def distill_query(self, query: str) -> str:
        """
        Normalize or cluster a query to find relevant existing templates.
        """
        pass
