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


def test_replace():
    program = {
        "type": "Program",
        "sourceType": "script",
        "body": [
            {
                "type": "FunctionDeclaration",
                "id": {"type": "Identifier", "name": "f"},
                "params": [{"type": "Identifier", "name": "x"}],
                "generator": False,
                "async": False,
                "expression": False,
                "body": {
                    "type": "BlockStatement",
                    "body": [],
                },
            },
            {
                "type": "ExpressionStatement",
                "expression": {
                    "type": "CallExpression",
                    "callee": {"type": "Identifier", "name": "f"},
                    "arguments": [{"type": "Literal", "value": 10, "raw": "10"}],
                },
            },
        ],
    }

    program = Node.from_dict(program)
    target = program.body[2].expression
    scope_analysis(program)

    subtrees = {
        "CallExpression": [CallExpression(False, Identifier("k"), [Literal(1, "1")])]
    }
    replace(subtrees, target)
