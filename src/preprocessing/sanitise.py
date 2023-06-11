# Remove all assert statements from the AST
from typing import Any

from js_ast.nodes import CallExpression
from js_ast.nodes import ExpressionStatement
from js_ast.nodes import MemberExpression
from js_ast.nodes import Node
from js_ast.nodes import ThrowStatement


def sanitise_node(node: Node) -> bool:
    if isinstance(node, ExpressionStatement):
        if isinstance(node.expression, CallExpression):
            if node.expression.callee.name and (
                "assert" in node.expression.callee.name
                or "print" in node.expression.callee.name
            ):
                return True

            if (
                isinstance(node.expression.callee, MemberExpression)
                and node.expression.callee.object.name == "console"
                and node.expression.callee.property.name == "log"
            ):
                return True

    if isinstance(node, ThrowStatement):
        return True

    return False


def sanitise_ast(ast: Node):
    for node in ast.traverse():
        for field in node.fields:
            val = getattr(node, field)
            if isinstance(val, list):
                new_body: list[Any] = []
                v: Any
                for v in val:
                    if sanitise_node(v):
                        continue

                    new_body.append(v)

                setattr(node, field, new_body)
