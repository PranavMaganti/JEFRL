#  Test that the function correctly replaces a node with a subtree
from ast.nodes import (
    BinaryExpression,
    BlockStatement,
    Expression,
    ExpressionStatement,
    Identifier,
    Literal,
    Node,
    SpreadElement,
    Statement,
    VariableDeclarator,
)
from utils.mutation import replace


# When target is the root node, return the original node
def test_replace_root_node():
    node = Statement()
    assert replace({"Statement": []}, node) == node


# When target is a field of parent, replace the field with the subtree
def test_replace_field():
    node = BinaryExpression(
        left=Identifier(name="a"),
        right=Identifier(name="b"),
        operator="+",
    )
    subtrees: dict[str, list[Node]] = {"Literal": [Literal(value=1, raw="1")]}
    new_node = replace(subtrees, node)

    assert new_node.left == Identifier(name="a", parent=new_node)
    assert new_node.right == Literal(value=1, raw="1", parent=new_node)
    assert new_node.operator == "+"
