# Remove all assert statements from the AST
from typing import Any
from js_ast.nodes import CallExpression, ExpressionStatement, Node


def sanitise_ast(ast: Node):
    for node in ast.traverse():
        for field in node.fields:
            val = getattr(node, field)
            if isinstance(val, list):
                new_body: list[Any] = []
                v: Any
                for v in val:
                    if (
                        isinstance(v, ExpressionStatement)
                        and isinstance(v.expression, CallExpression)
                        and v.expression.callee.name
                        and "assert" in v.expression.callee.name
                        # and "assert" in v.expression.callee.name
                    ):
                        continue

                    new_body.append(v)

                setattr(node, field, new_body)
