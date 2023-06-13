from typing import Any

import esprima
from js_ast.fragmentise import node_to_frags
from js_ast.nodes import Node


class TestMakeFrags:
    def test_single_child_node(self):
        node = esprima.parseScript("42")
        node = Node.from_dict(node.toDict())

        frag_seq: list[dict[str, Any]] = []
        node_types: set[str] = set()
        node_to_frags(node, frag_seq, node_types)

        assert len(frag_seq) == 3
        print(frag_seq)
        assert frag_seq[0] == {
            "type": "Program",
            "sourceType": "script",
            "body": [{"type": "ExpressionStatement"}],
        }
        assert frag_seq[1] == {
            "type": "ExpressionStatement",
            "expression": {"type": "Literal"},
            "directive": None,
        }
        assert frag_seq[2] == {
            "type": "Literal",
            "value": 42,
            "raw": "42",
            "regex": None,
            "bigint": None,
        }

    def test_multiple_child_node(self):
        node = esprima.parseScript("x + y")
        node = Node.from_dict(node.toDict())

        frag_seq: list[dict[str, Any]] = []
        node_types: set[str] = set()
        node_to_frags(node, frag_seq, node_types)

        assert len(frag_seq) == 5
        assert frag_seq[0] == {
            "type": "Program",
            "body": [{"type": "ExpressionStatement"}],
            "sourceType": "script",
        }
        assert frag_seq[1] == {
            "type": "ExpressionStatement",
            "expression": {"type": "BinaryExpression"},
            "directive": None,
        }
        assert frag_seq[2] == {
            "type": "BinaryExpression",
            "operator": "+",
            "left": {"type": "Identifier"},
            "right": {"type": "Identifier"},
        }
        assert frag_seq[3] == {"type": "Identifier", "name": "x"}
        assert frag_seq[4] == {"type": "Identifier", "name": "y"}

    def test_nested_node(self):
        node = esprima.parseScript("if (x) { y } else { z }")
        node = Node.from_dict(node.toDict())

        frag_seq: list[dict[str, Any]] = []
        node_types: set[str] = set()
        node_to_frags(
            node,
            frag_seq,
            node_types,
        )

        print(frag_seq)

        assert len(frag_seq) == 9

        assert frag_seq[0] == {
            "type": "Program",
            "sourceType": "script",
            "body": [{"type": "IfStatement"}],
        }

        assert frag_seq[1] == {
            "type": "IfStatement",
            "test": {"type": "Identifier"},
            "consequent": {"type": "BlockStatement"},
            "alternate": {"type": "BlockStatement"},
        }

        assert frag_seq[2] == {
            "type": "Identifier",
            "name": "x",
        }

        assert frag_seq[3] == {
            "type": "BlockStatement",
            "body": [{"type": "ExpressionStatement"}],
        }

        assert frag_seq[4] == {
            "type": "ExpressionStatement",
            "expression": {"type": "Identifier"},
            "directive": None,
        }

        assert frag_seq[5] == {
            "type": "Identifier",
            "name": "y",
        }

        assert frag_seq[6] == {
            "type": "BlockStatement",
            "body": [{"type": "ExpressionStatement"}],
        }

        assert frag_seq[7] == {
            "type": "ExpressionStatement",
            "expression": {"type": "Identifier"},
            "directive": None,
        }

        assert frag_seq[8] == {
            "type": "Identifier",
            "name": "z",
        }
