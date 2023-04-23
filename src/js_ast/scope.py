from __future__ import annotations
from ast import Tuple
import copy

from typing import Optional


class Scope:
    def __init__(
        self,
        variables: set = set(),
        functions: dict = {},
        classes: set = set(),
        parent: Optional[Scope] = None,
    ) -> None:
        self.variables = variables
        self.functions = functions
        self.classes = classes
        self.parent = parent

    def available_variables(self) -> set[str]:
        return self.variables | (
            self.parent.available_variables() if self.parent else set()
        )

    def available_functions(self) -> dict:
        return self.functions | (
            self.parent.available_functions() if self.parent else {}
        )

    def available_classes(self) -> set[str]:
        return self.classes | (
            self.parent.available_classes() if self.parent else set()
        )

    def __repr__(self) -> str:
        return f"Scope({self.available_variables()}, {self.available_functions()}, {self.available_classes()})"

    def __deepcopy__(self, _memo):
        return self.__class__(**copy.deepcopy({k: v for k, v in self.__dict__.items()}))


class BlockScope(Scope):
    pass
