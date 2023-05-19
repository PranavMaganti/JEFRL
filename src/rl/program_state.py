import logging
import random
from collections import deque

from js_ast.mutation import add, remove, replace
from js_ast.nodes import BlockStatement, ClassBody, FunctionDeclaration, Node, Program
from utils.js_engine import CoverageData


class ProgramState:
    def __init__(
        self, program: Node, coverage_data: CoverageData, original_file: str = ""
    ):
        self.program = program
        self.coverage_data = coverage_data
        self.target_node: Node = program
        self.context_node: deque[Node] = deque([program])

        self.original_file = original_file
        self.history = []

    def move_up(self) -> bool:
        if self.target_node.parent is None:
            return False

        self.target_node = self.target_node.parent
        if self.target_node == self.context_node[-1]:
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

    def generate_code(self, node: Node) -> str:
        try:
            return escodegen.generate(node)  # type: ignore
        except Exception:
            logging.error("Error generating code")
            return ""

    def generate_target_code(self) -> str:
        return self.generate_code(self.target_node)

    def generate_context_code(self) -> str:
        return self.generate_code(self.context_node[-1])

    def generate_program_code(self) -> str:
        return self.generate_code(self.program)

    def __str__(self):
        return self.__repr__()


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
