import pytest

from js_ast.nodes import (
    BinaryExpression,
    BlockStatement,
    CallExpression,
    ExpressionStatement,
    Identifier,
    Literal,
)
from utils.loader import sanitise_ast


class TestSanitiseAst:
    def test_remove_assert_calls(self):
        node = ExpressionStatement(
            expression=CallExpression(
                callee=Identifier(name="assert"),
                arguments=[Literal(value=True, raw="true")],
                optional=False,
            ),
            directive="",
        )
        root = BlockStatement(body=[node])

        sanitise_ast(root)

        assert root.body == []

    def test_keep_other_calls(self):
        node = ExpressionStatement(
            expression=CallExpression(
                callee=Identifier(name="foo"),
                arguments=[Literal(value=2, raw="2")],
                optional=False,
            ),
            directive="",
        )
        root = BlockStatement(body=[node])
        sanitise_ast(root)

        assert node in root.body

    def test_keep_other_nodes(self):
        node = ExpressionStatement(
            expression=BinaryExpression(
                left=Identifier(name="y"),
                operator="*",
                right=Literal(value=2, raw="2"),
            ),
            directive="",
        )

        root = BlockStatement(body=[node])
        sanitise_ast(root)

        assert node in root.body


if __name__ == "__main__":
    pytest.main()
