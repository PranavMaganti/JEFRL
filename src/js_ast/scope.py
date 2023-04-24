from __future__ import annotations

import copy
from ast import Tuple
from enum import Enum
from typing import Optional


class ScopeType(Enum):
    BLOCK = 1
    FUNCTION = 2
    CLASS = 3


class Scope:
    def __init__(
        self,
        variables: set = set(),
        functions: dict = {},
        classes: set = set(),
        parent: Optional[Scope] = None,
        type: ScopeType = ScopeType.BLOCK,
    ) -> None:
        self.variables = variables
        self.functions = functions
        self.classes = classes
        self.parent = parent
        self.type = type

    # Get all available variables in the current scope and all parent scopes
    def available_variables(self) -> set[str]:
        return self.variables | (
            self.parent.available_variables() if self.parent else set()
        )

    # Get all available functions in the current scope and all parent scopes
    def available_functions(self) -> dict:
        return self.functions | (
            self.parent.available_functions() if self.parent else {}
        )

    # Get all available classes in the current scope and all parent scopes
    def available_classes(self) -> set[str]:
        return self.classes | (
            self.parent.available_classes() if self.parent else set()
        )

    def __repr__(self) -> str:
        return f"Scope({self.available_variables()}, {self.available_functions()}, {self.available_classes()})"

    def __deepcopy__(self, _memo):
        return self.__class__(**copy.deepcopy({k: v for k, v in self.__dict__.items()}))
