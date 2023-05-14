from __future__ import annotations

from enum import Enum
from typing import Optional


class ScopeType(Enum):
    GLOBAL = 0
    BLOCK = 1
    FUNCTION = 2
    CLASS = 3


class Scope:
    def __init__(
        self,
        variables: Optional[set[str]] = None,
        functions: Optional[dict[str, int]] = None,
        classes: Optional[set[str]] = None,
        parent: Optional[Scope] = None,
        scope_type: ScopeType = ScopeType.BLOCK,
    ) -> None:
        self.variables: set[str] = variables if variables is not None else set()
        self.functions: dict[str, int] = functions if functions is not None else {}
        self.classes: set[str] = classes if classes is not None else set()
        self.parent = parent
        self.scope_type = scope_type

        self.parent_variables: set[str] = (
            self.parent.available_variables() if self.parent else set()
        )
        self.parent_functions: dict[str, int] = (
            self.parent.available_functions() if self.parent else {}
        )
        self.parent_classes: set[str] = (
            self.parent.available_classes() if self.parent else set()
        )

    # Get all available variables in the current scope and all parent scopes
    def available_variables(self) -> set[str]:
        return self.variables | self.parent_variables

    # Get all available functions in the current scope and all parent scopes
    def available_functions(self) -> dict[str, int]:
        return {**self.functions, **self.parent_functions}

    # Get all available classes in the current scope and all parent scopes
    def available_classes(self) -> set[str]:
        return self.classes | self.parent_classes

    def __repr__(self) -> str:
        return f"Scope({self.available_variables()}, {self.available_functions()}, {self.available_classes()}, {self.parent}, {self.scope_type})"

    # def __deepcopy__(self):
    #     return self.__class__(
    #         **copy.deepcopy(
    #             {
    #                 "variables": self.variables,
    #                 "functions": self.functions,
    #                 "classes": self.classes,
    #                 "parent": self.parent,
    #                 "scope_type": self.scope_type,
    #             }
    #         )
    #     )
