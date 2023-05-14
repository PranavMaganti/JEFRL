#  Test that the function correctly replaces a node with a subtree
from js_ast.analysis import scope_analysis
from js_ast.mutation import replace
from js_ast.nodes import (
    BinaryExpression,
    CallExpression,
    Identifier,
    Literal,
    Node,
    Statement,
)
