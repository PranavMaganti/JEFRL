import pytest

from js_ast.mutation import replace
from js_ast.nodes import (
    BinaryExpression,
    BlockStatement,
    Identifier,
    Literal,
    Node,
    VariableDeclaration,
    VariableDeclarator,
)
from js_ast.scope import Scope, ScopeType


class TestReplace:
    def test_replace_no_parent(self):
        subtrees = {"Identifier": [Identifier(name="y")]}
        target = Identifier(name="x")
        target.parent = None

        new_node = replace(subtrees, target)

        assert new_node == target

    def test_replace_no_subtrees(self):
        subtrees = {}
        target = Identifier(name="x")
        target.parent = BinaryExpression(
            left=target,
            operator="+",
            right=Literal(value=1, raw="1"),
            scope=Scope(scope_type=ScopeType.GLOBAL),
        )

        new_node = replace(subtrees, target)

        assert new_node == target

    def test_replace_no_match(self):
        subtrees = {"Identifier": [Identifier(name="y")]}
        target = Identifier(name="x")
        target.parent = BinaryExpression(
            left=Literal(value=1, raw="1"),
            operator="+",
            right=Literal(value=2, raw="2"),
            scope=Scope(scope_type=ScopeType.GLOBAL),
        )

        with pytest.raises(ValueError):
            replace(subtrees, target)

    def test_replace_list(self):
        subtrees = {
            "BinaryExpression": [
                BinaryExpression(
                    left=Identifier(name="x"),
                    operator="/",
                    right=Identifier(name="y"),
                )
            ]
        }
        target = BinaryExpression(
            left=Identifier(name="x"),
            operator="+",
            right=Literal(value=1, raw="1"),
        )
        parent = BlockStatement(
            body=[
                VariableDeclaration(
                    kind="let",
                    declarations=[
                        VariableDeclarator(id=Identifier(name="x"), init=None),
                        VariableDeclarator(id=Identifier(name="y"), init=None),
                    ],
                ),
                target,
            ]
        )
        target.parent = parent

        new_node = replace(subtrees, target)

        assert new_node.parent == parent
        assert new_node in parent.body

        assert new_node.type == "BinaryExpression"
        assert new_node.left.type == "Identifier"
        assert new_node.left.name == "x"
        assert new_node.operator == "/"
        assert new_node.right.type == "Identifier"
        assert new_node.right.name == "y"

        for child in parent.traverse():
            assert hasattr(child, "scope")

    def test_replace_nested(self):
        subtrees = {"Identifier": [Identifier(name="y")]}
        target = Identifier(name="x")
        parent = BinaryExpression(
            left=target,
            operator="*",
            right=Literal(value=2, raw="2"),
            scope=Scope(scope_type=ScopeType.GLOBAL),
        )
        root = BlockStatement(
            body=[
                VariableDeclaration(
                    kind="let",
                    declarations=[
                        VariableDeclarator(id=Identifier(name="x"), init=None),
                        VariableDeclarator(id=Identifier(name="y"), init=None),
                    ],
                ),
                BinaryExpression(
                    left=Identifier(name="a"),
                    operator="+",
                    right=parent,
                ),
            ]
        )

        parent.parent = root
        target.parent = parent

        new_node = replace(subtrees, target)

        assert new_node.parent == parent
        assert new_node == parent.left

        assert new_node.type == "Identifier"
        assert new_node.name == "y"

        for child in root.traverse():
            assert hasattr(child, "scope")


if __name__ == "__main__":
    pytest.main()
