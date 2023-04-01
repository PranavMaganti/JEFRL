"""Trasnforms AST dictionary into a tree of Node objects."""
from __future__ import annotations

import abc
import json
from typing import Any, Generator, Optional, Union


class UnknownNodeTypeError(Exception):
    """Raised if we encounter a node with an unknown type."""

    pass


class Node(abc.ABC):
    """Abstract Node class which defines node operations"""

    node_type: str
    loc: Optional[dict[str, int]]

    @property
    @abc.abstractmethod
    def fields(self) -> list[str]:  # type: ignore
        """list of field names associated with this node type, in canonical order."""

    def children(self) -> list[Node]:
        """list of node children"""
        return []

    def __init__(self, data: dict[str, Any], parent: Optional[Node]) -> None:
        """Sets one attribute in the Node for each field (e.g. self.body)."""
        self.parent = parent
        self.node_type = data["type"]
        self.loc = data["loc"] if "loc" in data else None

        for field in self.fields:
            setattr(self, field, objectify(data.get(field), self))

    def dict(self) -> dict[str, Any]:
        """Transform the Node back into an Esprima-compatible AST dictionary."""
        result: dict[str, Any] = {"type": self.node_type}
        for field in self.fields:
            val = getattr(self, field)
            if isinstance(val, Node):
                result[field] = val.dict()
            elif isinstance(val, list):
                result[field] = [x.dict() for x in val]
            else:
                result[field] = val
        return result

    def traverse(self) -> Generator[Node, None, None]:
        """Pre-order traversal of this node and all of its children."""
        yield self
        for field in self.fields:
            val = getattr(self, field)
            if isinstance(val, Node):
                yield from val.traverse()
            elif isinstance(val, list):
                for node in val:
                    yield from node.traverse()

    @property
    def type(self) -> str:
        """The name of the node type, e.g. 'Identifier'."""
        return self.__class__.__name__

    def __repr__(self) -> str:
        """String representation of the node."""
        return json.dumps(self.dict(), indent=4)

    def __str__(self) -> str:
        return self.__repr__()


def objectify(
    data: Union[None, dict[str, Any], list[dict[str, Any]]],
    parent: Optional[Node] = None,
) -> Union[None, dict[str, Any], list[Any], Node]:
    """Recursively transform AST data into a Node object."""
    if not isinstance(data, (dict, list)):
        # Data is a basic type (None, string, number)
        return data

    if isinstance(data, dict):
        if "type" not in data:
            # Literal values can be empty dictionaries, for example.
            return data
        # Transform the type into the appropriate class.
        node_class = globals().get(data["type"])
        if not node_class:
            raise UnknownNodeTypeError(data["type"])
        return node_class(data, parent)
    else:
        # Data is a list of nodes.
        return [objectify(x, parent) for x in data if x is not None]


class Expression(Node):
    pass


class Statement(Node):
    pass


class Pattern(Node):
    pass


# --- AST spec: https://github.com/estree/estree/blob/master/es5.md ---
class Identifier(Expression, Pattern):
    @property
    def fields(self):
        return ["name"]


class Literal(Expression):
    @property
    def fields(self):
        return ["value"]


class RegExpLiteral(Literal):
    @property
    def fields(self):
        return ["regex"]


class Program(Node):
    @property
    def fields(self):
        return ["body"]

    def children(self) -> list[Node]:
        return getattr(self, "body")


class Function(Node):
    @property
    def fields(self):
        return ["id", "params", "body"]

    def children(self) -> list[Node]:
        return [getattr(self, "id"), getattr(self, "body")] + getattr(self, "params")


# Statements
class ExpressionStatement(Statement):
    @property
    def fields(self):
        return ["expression"]

    def children(self) -> list[Node]:
        return [getattr(self, "expression")]


class Directive(ExpressionStatement):
    @property
    def fields(self):
        return ["expression", "directive"]

    def children(self) -> list[Node]:
        return [getattr(self, "expression")]


class BlockStatement(Statement):
    @property
    def fields(self):
        return ["body"]

    def children(self) -> list[Node]:
        return getattr(self, "body")


class FunctionBody(BlockStatement):
    @property
    def fields(self):
        return ["body"]

    def children(self) -> list[Node]:
        return getattr(self, "body")


class EmptyStatement(Statement):
    @property
    def fields(self):
        return []


class DebuggerStatement(Statement):
    @property
    def fields(self):
        return []


class WithStatement(Statement):
    @property
    def fields(self):
        return ["object", "body"]

    def children(self) -> list[Node]:
        return [getattr(self, "object"), getattr(self, "body")]


# Control Flow
class ReturnStatement(Statement):
    @property
    def fields(self):
        return ["argument"]

    def children(self) -> list[Node]:
        return [getattr(self, "argument")]


class LabeledStatement(Statement):
    @property
    def fields(self):
        return ["label", "body"]


class BreakStatement(Statement):
    @property
    def fields(self):
        return ["label"]


class ContinueStatement(Statement):
    @property
    def fields(self):
        return ["label"]


# Choice
class IfStatement(Statement):
    @property
    def fields(self):
        return ["test", "consequent", "alternate"]

    def children(self) -> list[Node]:
        return [
            getattr(self, "test"),
            getattr(self, "consequent"),
            getattr(self, "alternate"),
        ]


class SwitchStatement(Statement):
    @property
    def fields(self):
        return ["discriminant", "cases"]

    def children(self) -> list[Node]:
        return [getattr(self, "discriminant")] + getattr(self, "cases")


class SwitchCase(Node):
    @property
    def fields(self):
        return ["test", "consequent"]

    def children(self) -> list[Node]:
        return [getattr(self, "test")] + getattr(self, "consequent")


# Exceptions
class ThrowStatement(Statement):
    @property
    def fields(self):
        return ["argument"]

    def children(self) -> list[Node]:
        return [getattr(self, "argument")]


class TryStatement(Statement):
    @property
    def fields(self):
        return ["block", "handler", "finalizer"]

    def children(self) -> list[Node]:
        return [
            getattr(self, "block"),
            getattr(self, "handler"),
            getattr(self, "finalizer"),
        ]


class CatchClause(Node):
    @property
    def fields(self):
        return ["param", "body"]

    def children(self) -> list[Node]:
        return getattr(self, "body")


# Loops
class WhileStatement(Statement):
    @property
    def fields(self):
        return ["test", "body"]

    def children(self) -> list[Node]:
        return [getattr(self, "test"), getattr(self, "body")]


class DoWhileStatement(Statement):
    @property
    def fields(self):
        return ["body", "test"]

    def children(self) -> list[Node]:
        return [getattr(self, "body"), getattr(self, "test")]


class ForStatement(Statement):
    @property
    def fields(self):
        return ["init", "test", "update", "body"]

    def children(self) -> list[Node]:
        return [
            getattr(self, "init"),
            getattr(self, "test"),
            getattr(self, "update"),
            getattr(self, "body"),
        ]


class ForInStatement(Statement):
    @property
    def fields(self):
        return ["left", "right", "body"]

    def children(self) -> list[Node]:
        return [
            getattr(self, "left"),
            getattr(self, "right"),
            getattr(self, "body"),
        ]


class ForOfStatement(Node):
    @property
    def fields(self):
        return ["left", "right", "body"]

    def children(self) -> list[Node]:
        return [
            getattr(self, "left"),
            getattr(self, "right"),
            getattr(self, "body"),
        ]


# Declarations
class Declaration(Statement):
    pass


class FunctionDeclaration(Function, Declaration):
    @property
    def fields(self):
        return ["id"]


class VariableDeclaration(Declaration):
    @property
    def fields(self):
        return ["declarations"]

    def children(self) -> list[Node]:
        return getattr(self, "declarations")


class VariableDeclarator(Node):
    @property
    def fields(self):
        return ["id", "init"]

    def children(self) -> list[Node]:
        return [getattr(self, "init")]


# Expressions
class ThisExpression(Expression):
    @property
    def fields(self):
        return []


class ArrayExpression(Expression):
    @property
    def fields(self):
        return ["elements"]

    def children(self) -> list[Node]:
        return getattr(self, "elements")


class ObjectExpression(Expression):
    @property
    def fields(self):
        return ["properties"]

    def children(self) -> list[Node]:
        return getattr(self, "properties")


class Property(Node):
    @property
    def fields(self):
        return ["key", "value", "kind"]

    def children(self) -> list[Node]:
        return [getattr(self, "key"), getattr(self, "value")]


class FunctionExpression(Function, Expression):
    pass


class UnaryExpression(Expression):
    @property
    def fields(self):
        return ["operator", "prefix", "argument"]

    def children(self) -> list[Node]:
        return [getattr(self, "argument")]


class UpdateExpression(Expression):
    @property
    def fields(self):
        return ["operator", "argument", "prefix"]

    def children(self) -> list[Node]:
        return [getattr(self, "argument")]


class BinaryExpression(Expression):
    @property
    def fields(self):
        return ["operator", "left", "right"]

    def children(self) -> list[Node]:
        return [getattr(self, "left"), getattr(self, "right")]


class AssignmentExpression(Expression):
    @property
    def fields(self):
        return ["operator", "left", "right"]

    def children(self) -> list[Node]:
        return [getattr(self, "left"), getattr(self, "right")]


class LogicalExpression(Expression):
    @property
    def fields(self):
        return ["operator", "left", "right"]

    def children(self) -> list[Node]:
        return [getattr(self, "left"), getattr(self, "right")]


class MemberExpression(Expression, Pattern):
    @property
    def fields(self):
        return ["object", "property", "computed"]

    def children(self) -> list[Node]:
        return [getattr(self, "object"), getattr(self, "property")]


class ConditionalExpression(Expression):
    @property
    def fields(self):
        return ["test", "consequent", "alternate"]

    def children(self) -> list[Node]:
        return [
            getattr(self, "test"),
            getattr(self, "consequent"),
            getattr(self, "alternate"),
        ]


class CallExpression(Expression):
    @property
    def fields(self):
        return ["callee", "arguments"]

    def children(self) -> list[Node]:
        return [getattr(self, "callee")] + getattr(self, "arguments")


class NewExpression(Expression):
    @property
    def fields(self):
        return ["callee", "arguments"]

    def children(self) -> list[Node]:
        return [getattr(self, "callee")] + getattr(self, "arguments")


class SequenceExpression(Expression):
    @property
    def fields(self):
        return ["expressions"]

    def children(self) -> list[Node]:
        return getattr(self, "expressions")


class Super(Node):
    @property
    def fields(self):
        return []


class SpreadElement(Node):
    @property
    def fields(self):
        return ["argument"]

    def children(self) -> list[Node]:
        return [getattr(self, "argument")]


class ArrowFunctionExpression(Node):
    @property
    def fields(self):
        return ["params", "body"]

    def children(self) -> list[Node]:
        return getattr(self, "body")


class YieldExpression(Node):
    @property
    def fields(self):
        return ["argument", "delegate"]

    def children(self) -> list[Node]:
        return [getattr(self, "argument")]


# Classes
class ClassBody(Node):
    @property
    def fields(self):
        return ["body"]

    def children(self) -> list[Node]:
        return getattr(self, "body")


class ClassDeclaration(Node):
    @property
    def fields(self):
        return ["id", "superClass", "body"]

    def children(self) -> list[Node]:
        return [getattr(self, "body")]


class MethodDefinition(Node):
    @property
    def fields(self):
        return ["key", "value", "kind"]

    def children(self) -> list[Node]:
        return [getattr(self, "key"), getattr(self, "value")]


class ClassExpression(Node):
    @property
    def fields(self):
        return ["id", "superClass", "body"]

    def children(self) -> list[Node]:
        return [getattr(self, "body")]


class MetaProperty(Node):
    @property
    def fields(self):
        return ["meta", "property"]

    def children(self) -> list[Node]:
        return [getattr(self, "meta"), getattr(self, "property")]


# Patterns
class ObjectPattern(Node):
    @property
    def fields(self):
        return ["properties"]

    def children(self) -> list[Node]:
        return getattr(self, "properties")


class ArrayPattern(Node):
    @property
    def fields(self):
        return ["elements"]

    def children(self) -> list[Node]:
        return getattr(self, "elements")


class RestElement(Node):
    @property
    def fields(self):
        return ["argument"]

    def children(self) -> list[Node]:
        return [getattr(self, "argument")]


class AssignmentPattern(Node):
    @property
    def fields(self):
        return ["left", "right"]

    def children(self) -> list[Node]:
        return [getattr(self, "left"), getattr(self, "right")]


# Template Literals
class TemplateLiteral(Node):
    @property
    def fields(self):
        return ["quasis", "expressions"]

    def children(self) -> list[Node]:
        return getattr(self, "quasis") + getattr(self, "expressions")


class TaggedTemplateExpression(Node):
    @property
    def fields(self):
        return ["tag", "quasi"]

    def children(self) -> list[Node]:
        return [getattr(self, "tag"), getattr(self, "quasi")]


class TemplateElement(Node):
    @property
    def fields(self):
        return ["value", "tail"]

    def children(self) -> list[Node]:
        return []
