from collections import deque
import copy
import random
from typing import Any

from js_ast.mutation import add
from js_ast.mutation import remove
from js_ast.mutation import replace
from js_ast.nodes import BlockStatement
from js_ast.nodes import ClassBody
from js_ast.nodes import FunctionDeclaration
from js_ast.nodes import Node
from js_ast.nodes import Program

from utils.js_engine import Coverage


class ProgramState:
    def __init__(self, program: Node, coverage_data: Coverage, original_file: str = ""):
        self.program = program
        self.coverage_data = coverage_data
        self.target_node: Node = program
        self.context_node: deque[Node] = deque([program])

        self.original_file = original_file
        self.history: list[Node] = []

    def move_up(self) -> bool:
        if self.target_node.parent is None:
            return False

        self.target_node = self.target_node.parent
        if len(self.context_node) > 1 and self.target_node == self.context_node[-1]:
            self.context_node.pop()

        return True

    def move_down(self) -> bool:
        children = self.target_node.children()
        if not children:
            return False

        if self.target_node != self.context_node[-1] and is_context_node(
            self.target_node
        ):
            self.context_node.append(self.target_node)

        self.target_node = random.choice(children)

        return True

    def replace(self, subtrees: dict[str, list[Node]]) -> Node:
        return replace(subtrees, self.target_node)

    def add(self, subtrees: dict[str, list[Node]]) -> Node:
        return add(subtrees, self.target_node)

    def remove(self) -> Node:
        return remove(self.target_node)

    def generate_target_code(self) -> str:
        return self.target_node.generate_code()

    def generate_context_code(self) -> str:
        return self.context_node[-1].generate_code()

    def generate_program_code(self) -> str:
        return self.program.generate_code()

    def __str__(self):
        return self.__repr__()

    def __deepcopy__(self, _memo: dict[int, Any]):
        new = self.__class__(
            copy.deepcopy(self.program, _memo),
            self.coverage_data,
            self.original_file,
        )
        new.history = list(self.history)

        return new


def is_context_node(node: Node) -> bool:
    return (
        isinstance(node, Program)
        or isinstance(node, ClassBody)
        or isinstance(node, FunctionDeclaration)
        or (
            isinstance(node, BlockStatement)
            and node.parent is not None
            and not isinstance(node.parent, FunctionDeclaration)
        )
    )
