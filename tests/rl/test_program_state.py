from collections import deque

from js_ast.nodes import BinaryExpression
from js_ast.nodes import BlockStatement
from js_ast.nodes import ClassBody
from js_ast.nodes import ExpressionStatement
from js_ast.nodes import FunctionDeclaration
from js_ast.nodes import Identifier
from js_ast.nodes import Program
from rl.program_state import is_context_node
from rl.program_state import ProgramState

from utils.js_engine import Coverage
from utils.js_engine import ExecutionData
from utils.js_engine import JSError


class TestIsContextNode:
    def test_program_is_context_node(self):
        node = Program(body=[], sourceType="script")

        assert is_context_node(node)

    def test_class_body_is_context_node(self):
        node = ClassBody(body=[])

        assert is_context_node(node)

    def test_function_declaration_is_context_node(self):
        node = FunctionDeclaration(
            id=Identifier(name="foo"),
            params=[],
            body=BlockStatement(body=[]),
            generator=False,
        )

        assert is_context_node(node)

    def test_block_statement_is_context_node(self):
        node = BlockStatement(body=[])
        parent = Program(body=[node], sourceType="script")

        assert is_context_node(node)

    def test_block_statement_in_function_is_not_context_node(self):
        node = BlockStatement(body=[])
        parent = FunctionDeclaration(
            id=Identifier(name="foo"),
            params=[],
            body=node,
            generator=False,
        )

        assert not is_context_node(node)


class TestProgramState:
    def test_move_up_root_node(self):
        node = Program(body=[], sourceType="script")
        state = ProgramState(
            node, exec_data=ExecutionData(Coverage(), JSError.NoError, "")
        )

        assert not state.move_up()

    def test_move_up_context_node(self):
        node = BlockStatement(body=[])
        parent = FunctionDeclaration(
            id=Identifier("foo"),
            params=[],
            body=node,
            generator=False,
        )
        root = Program(body=[parent], sourceType="script")
        state = ProgramState(root, ExecutionData(Coverage(), JSError.NoError, ""))

        state.target_node = node
        state.context_node = deque([root, parent])

        assert state.move_up()
        assert state.target_node == parent
        assert len(state.context_node) == 1
        assert state.context_node[0] == root

    def test_move_up_normal_node(self):
        expr = BinaryExpression(
            left=Identifier(name="a"),
            right=Identifier(name="b"),
            operator="*",
        )
        node = ExpressionStatement(expression=expr, directive="")
        parent = BlockStatement(body=[node])
        root = Program(body=[parent], sourceType="script")

        state = ProgramState(root, ExecutionData(Coverage(), JSError.NoError, ""))

        state.target_node = expr
        state.context_node = deque([root, parent])

        assert state.move_up()
        assert state.target_node == node
        assert len(state.context_node) == 2
        assert state.context_node[0] == root
        assert state.context_node[1] == parent

    def test_move_down_no_children(self):
        node = BlockStatement(body=[])
        state = ProgramState(node, ExecutionData(Coverage(), JSError.NoError, ""))

        assert not state.move_down()

    def test_move_down_context_node(self):
        node = FunctionDeclaration(
            id=Identifier("foo"),
            params=[],
            body=BlockStatement(body=[]),
            generator=False,
        )
        parent = Program(body=[node], sourceType="script")
        state = ProgramState(parent, ExecutionData(Coverage(), JSError.NoError, ""))

        assert state.move_down()
        assert state.target_node == node
        assert len(state.context_node) == 1
        assert state.context_node[0] == parent

        assert state.move_down()
        assert state.target_node == node.body
        assert len(state.context_node) == 2
        assert state.context_node[0] == parent
        assert state.context_node[1] == node

    def test_move_down_normal_node(self):
        expr = BinaryExpression(
            left=Identifier(name="a"),
            right=Identifier(name="b"),
            operator="*",
        )
        node = ExpressionStatement(expression=expr, directive="")
        parent = Program(body=[node], sourceType="script")
        state = ProgramState(parent, ExecutionData(Coverage(), JSError.NoError, ""))

        assert state.move_down()
        assert state.target_node == node
        assert len(state.context_node) == 1
        assert state.context_node[0] == parent


if __name__ == "__main__":
    pytest.main()
