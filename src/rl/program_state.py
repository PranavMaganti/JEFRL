from collections import deque
import copy
import random
from typing import Any, Optional

from js_ast.mutation import add
from js_ast.mutation import modify
from js_ast.mutation import remove
from js_ast.mutation import replace
from js_ast.nodes import BlockStatement
from js_ast.nodes import ClassBody
from js_ast.nodes import FunctionDeclaration
from js_ast.nodes import Node
from js_ast.nodes import Program
from rl.fuzzing_action import FuzzingAction

from utils.js_engine import ExecutionData


class ProgramState:
    __slots__ = [
        "original_program",
        "program",
        "exec_data",
        "target_node",
        "context_node",
        "action_history",
    ]

    def __init__(self, program: Node, exec_data: ExecutionData):
        self.original_program = copy.deepcopy(program)
        self.program = program
        # self.coverage = coverage
        self.exec_data = exec_data
        self.target_node: Node = program
        self.context_node: deque[Node] = deque([program])

        self.action_history: list[FuzzingAction] = []

    def move_up(self) -> bool:
        self.action_history.append(FuzzingAction.MOVE_UP)
        if self.target_node.parent is None:
            return False

        self.target_node = self.target_node.parent
        if len(self.context_node) > 1 and self.target_node == self.context_node[-1]:
            self.context_node.pop()

        return True

    def move_down(self) -> bool:
        self.action_history.append(FuzzingAction.MOVE_DOWN)
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
        self.action_history.append(FuzzingAction.REPLACE)
        return replace(subtrees, self.target_node, self.program)

    def add(self, subtrees: dict[str, list[Node]]) -> Node:
        self.action_history.append(FuzzingAction.ADD)
        return add(subtrees, self.target_node, self.program)

    def remove(self) -> Node:
        self.action_history.append(FuzzingAction.REMOVE)
        return remove(self.target_node, self.program)

    def modify(self) -> bool:
        self.action_history.append(FuzzingAction.MODIFY)
        return modify(self.target_node)

    def get_target_node(self) -> Node:
        return self.target_node

    def get_context_node(self) -> Node:
        return self.context_node[-1]

    def generate_program_code(self) -> Optional[str]:
        return self.program.generate_code()

    def __str__(self):
        return self.__repr__()

    def __deepcopy__(self, _memo: dict[int, Any]):
        return self.__class__(
            copy.deepcopy(self.program, _memo),
            self.exec_data,
        )


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
