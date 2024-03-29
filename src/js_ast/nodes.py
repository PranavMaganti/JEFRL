"""Trasnforms AST dictionary into a tree of Node objects."""
from __future__ import annotations

import abc
import copy
import dataclasses
from dataclasses import dataclass
from dataclasses import field
import json
import logging
import re
from typing import Any, Generator, Optional, Union

import js_ast.escodegen as escodegen
from js_ast.scope import Scope


class UnknownNodeTypeError(Exception):
    """Raised if we encounter a node with an unknown type."""

    pass


estree_field_map = {
    "isAsync": "async",
    "awaitAllowed": "await",
}

context_fields = {"parent", "scope", "end_scope", "origin_file"}

# Set of children fields that should not be children
non_child_fields = {"id"}


@dataclass(kw_only=True, slots=True)
class Node(metaclass=abc.ABCMeta):
    """Abstract Node class which defines node operations"""

    # loc: Optional[dict[str, int]]
    origin_file: Optional[str] = None
    parent: Optional[Node] = None
    scope: Optional[Scope] = None
    end_scope: Optional[Scope] = None

    def __post_init__(self) -> None:
        """Set the parent of each child node."""
        for field in self.fields:
            val = getattr(self, field)
            if isinstance(val, Node):
                val.parent = self
            elif isinstance(val, list):
                node: Any
                for node in val:
                    if isinstance(node, Node):
                        node.parent = self

    @property
    def type(self) -> str:
        """The name of the node type, e.g. 'Identifier'."""
        return self.__class__.__name__

    @property
    def fields(self) -> list[str]:
        """list of node fields"""
        return ["type"] + [
            f.name for f in dataclasses.fields(self) if f.name not in context_fields
        ]

    def traverse(self) -> Generator[Node, None, None]:
        """Pre-order traversal of this node and all of its children."""
        yield self
        for node in self.children():
            yield from node.traverse()

    def to_dict(self) -> dict[str, Any]:
        """Transform the Node back into an Esprima-compatible AST dictionary."""
        result: dict[str, Any] = {"type": self.type}
        for field in self.fields:
            data_field = estree_field_map[field] if field in estree_field_map else field

            val = getattr(self, field)
            if isinstance(val, Node):
                result[data_field] = val.to_dict()
            elif isinstance(val, list):
                result[data_field] = [
                    x.to_dict() if isinstance(x, Node) else x for x in val
                ]
            elif val is not None:
                result[data_field] = val
        return result

    @staticmethod
    def from_dict(
        data: Union[None, dict[str, Any], list[dict[str, Any]]],
        origin_file: Optional[str] = None,
    ) -> Union[None, dict[str, Any], list[Any], Node]:
        """Recursively transform AST data into a Node object."""
        if not isinstance(data, (dict, list)):
            # Data is a basic type (None, string, number)
            return data

        if isinstance(data, dict):
            if "type" not in data:
                # Literal values can be empty dictionaries, for example.
                return data

            if data["type"] == "Literal" and hasattr(data, "regex") and data["regex"]:
                return Literal(
                    value=re.compile(data["regex"]["pattern"]),
                    raw=data["raw"],
                    regex=data["regex"],
                )

            # Transform the type into the appropriate class.
            node_class = globals().get(data["type"])
            if not node_class:
                print(data)
                raise UnknownNodeTypeError(data["type"])

            fields = [
                f.name
                for f in dataclasses.fields(node_class)
                if f.name not in context_fields
            ]
            if data["type"] == "FunctionDeclaration":
                assert data["params"] is not None
            params: dict[str, Any] = {"origin_file": origin_file}

            for field in fields:
                data_field = (
                    estree_field_map[field] if field in estree_field_map else field
                )

                if data_field not in data:
                    params[field] = None
                else:
                    params[field] = Node.from_dict(data[data_field])

            if data["type"] == "FunctionDeclaration":
                assert params["params"] is not None

            return node_class(**params)
        else:
            # Data is a list of nodes.
            return [Node.from_dict(x) for x in data]

    def children(self) -> list[Node]:
        children: list[Node] = []

        for field in self.fields:
            if field in non_child_fields:
                continue

            val = getattr(self, field)
            if isinstance(val, Node):
                children.append(val)
            elif isinstance(val, list):
                node: Any
                for node in val:
                    if isinstance(node, Node):
                        children.append(node)

        return children

    def root(self) -> Node:
        """Return the root node of the tree."""
        node = self
        while node.parent:
            node = node.parent
        return node

    def generate_code(self) -> Optional[str]:
        try:
            return escodegen.generate(self)  # type: ignore
        except Exception:
            print(self)
            logging.error("Error generating code")
            return None

    def __repr__(self) -> str:
        """String representation of the node."""
        return json.dumps(self.to_dict(), indent=4)

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), indent=4)

    def __getattr__(self, name: str):
        return None

    def __iter__(self):
        return self.__iter__

    def __deepcopy__(self, _memo: dict[int, Any]):
        return self.__class__(
            **copy.deepcopy(
                {
                    k.name: getattr(self, k.name)
                    for k in dataclasses.fields(self)
                    if k.name not in {"parent", "scope", "end_scope"}
                }
            )
        )

    def __getstate__(self):
        return {
            slot.name: getattr(self, slot.name) for slot in dataclasses.fields(self)
        }

    def __setstate__(self, d):
        for slot in d:
            setattr(self, slot, d[slot])


@dataclass(slots=True)
class Pattern(Node):
    pass


@dataclass(slots=True)
class Expression(Node):
    pass


@dataclass(slots=True)
class Identifier(Expression, Pattern):
    name: str


@dataclass(slots=True)
class Literal(Expression):
    value: Any
    raw: str
    regex: Optional[dict[str, Any]] = None
    bigint: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        if self.regex:
            return {
                "type": "Literal",
                "raw": self.raw,
                "regex": self.regex,
                "bigint": self.bigint,
            }

        return {
            "type": "Literal",
            "value": self.value,
            "raw": self.raw,
            "bigint": self.bigint,
        }


@dataclass(slots=True)
class Program(Node):
    sourceType: str
    body: list[Union[Statement, ImportOrExportDeclaration]]


@dataclass(kw_only=True, slots=True)
class Function(Node):
    id: Optional[Identifier] = None
    params: list[Pattern] = field(default_factory=list)
    generator: bool
    isAsync: bool = False


# Statements
@dataclass(slots=True)
class Statement(Node):
    pass


@dataclass(slots=True)
class ExpressionStatement(Statement):
    expression: Expression
    directive: str


@dataclass(slots=True)
class BlockStatement(Statement):
    body: list[Statement] = field(default_factory=list)


@dataclass(slots=True)
class StaticBlock(BlockStatement):
    pass


@dataclass(slots=True)
class FunctionBody(BlockStatement):
    body: list[Union[ExpressionStatement, Statement]]


@dataclass(slots=True)
class EmptyStatement(Statement):
    pass


@dataclass(slots=True)
class DebuggerStatement(Statement):
    pass


@dataclass(slots=True)
class WithStatement(Statement):
    object: Expression
    body: Statement


# Control Flow
@dataclass(slots=True)
class ReturnStatement(Statement):
    argument: Optional[Expression]


@dataclass(slots=True)
class LabeledStatement(Statement):
    label: Identifier
    body: Statement


@dataclass(slots=True)
class BreakStatement(Statement):
    label: Optional[Identifier]


@dataclass(slots=True)
class ContinueStatement(Statement):
    label: Optional[Identifier]


# Choice
@dataclass(slots=True)
class IfStatement(Statement):
    test: Expression
    consequent: Statement
    alternate: Optional[Statement]


@dataclass(slots=True)
class SwitchStatement(Statement):
    discriminant: Expression
    cases: list[SwitchCase] = field(default_factory=list)


@dataclass(slots=True)
class SwitchCase(Node):
    test: Optional[Expression]
    consequent: list[Statement] = field(default_factory=list)


# Exceptions
@dataclass(slots=True)
class ThrowStatement(Statement):
    argument: Expression


@dataclass(slots=True)
class TryStatement(Statement):
    block: BlockStatement
    handler: Optional[CatchClause]
    finalizer: Optional[BlockStatement]
    param: Optional[Pattern]


@dataclass(slots=True)
class CatchClause(Node):
    param: Pattern
    body: BlockStatement


# Loops
@dataclass(slots=True)
class WhileStatement(Statement):
    test: Expression
    body: Statement


@dataclass(slots=True)
class DoWhileStatement(Statement):
    body: Statement
    test: Expression


@dataclass(kw_only=True, slots=True)
class ForStatement(Statement):
    init: Optional[Union[VariableDeclaration, Expression]]
    test: Optional[Expression]
    update: Optional[Expression]
    body: Statement


@dataclass(slots=True)
class ForInStatement(Statement):
    left: Union[VariableDeclaration, Pattern]
    right: Expression
    body: Statement


@dataclass(slots=True)
class ForOfStatement(ForInStatement):
    awaitAllowed: bool


# Declarations
@dataclass(slots=True)
class Declaration(Statement):
    pass


@dataclass(kw_only=True, slots=True)
class FunctionDeclaration(Function, Declaration):
    id: Identifier
    expression: bool = False
    body: Union[FunctionBody, Expression]


@dataclass(kw_only=True, slots=True)
class VariableDeclaration(Declaration):
    declarations: list[VariableDeclarator]
    kind: str


@dataclass(slots=True)
class VariableDeclarator(Node):
    id: Pattern
    init: Optional[Expression]


# Expressions
@dataclass(slots=True)
class ThisExpression(Expression):
    pass


@dataclass(slots=True)
class ArrayExpression(Expression):
    elements: list[Optional[Union[Expression, SpreadElement]]] = field(
        default_factory=list
    )


@dataclass(slots=True)
class ObjectExpression(Expression):
    properties: list[Union[Property, SpreadElement]] = field(default_factory=list)


@dataclass(slots=True)
class Property(Node):
    key: Expression
    value: Expression
    kind: str
    method: bool
    shorthand: bool
    computed: bool


@dataclass(slots=True)
class FunctionExpression(Function, Expression):
    expression: bool
    body: Union[FunctionBody, Expression]


# Unary operations
@dataclass(slots=True)
class UnaryExpression(Expression):
    operator: str
    prefix: bool
    argument: Expression


@dataclass(slots=True)
class UpdateExpression(Expression):
    operator: str
    argument: Expression
    prefix: bool


# Binary operations
@dataclass(slots=True)
class BinaryExpression(Expression):
    operator: str
    left: Union[Expression, PrivateIdentifier]
    right: Expression


@dataclass(slots=True)
class AssignmentExpression(Expression):
    operator: str
    left: Pattern
    right: Expression


@dataclass(slots=True)
class LogicalExpression(Expression):
    operator: str
    left: Expression
    right: Expression


@dataclass(slots=True)
class ChainExpression(Expression):
    expression: Expression


@dataclass(slots=True)
class ChainElement(Node):
    optional: bool


@dataclass(slots=True)
class MemberExpression(ChainElement):
    object: Union[Expression, Super]
    property: Union[Expression, PrivateIdentifier]
    computed: bool


@dataclass(slots=True)
class ConditionalExpression(Expression):
    test: Expression
    consequent: Expression
    alternate: Expression


@dataclass(slots=True)
class CallExpression(ChainElement):
    callee: Union[Expression, Super]
    arguments: list[Union[Expression, SpreadElement]] = field(default_factory=list)


@dataclass(slots=True)
class NewExpression(Expression):
    callee: Expression
    arguments: list[Union[Expression, SpreadElement]] = field(default_factory=list)


@dataclass(slots=True)
class SequenceExpression(Expression):
    expressions: list[Expression] = field(default_factory=list)


@dataclass(slots=True)
class Super(Node):
    pass


@dataclass(slots=True)
class SpreadElement(Node):
    argument: Expression


@dataclass(kw_only=True, slots=True)
class ArrowFunctionExpression(Function, Expression):
    body: Union[FunctionBody, Expression]
    expression: bool
    generator: bool = False


@dataclass(kw_only=True, slots=True)
class YieldExpression(Expression):
    argument: Optional[Expression]
    delegate: bool


@dataclass(slots=True)
class AwaitExpression(Expression):
    argument: Expression


@dataclass(slots=True)
class ImportExpression(Expression):
    source: Expression


# Template Literals
@dataclass(slots=True)
class TemplateLiteral(Expression):
    quasis: list[TemplateElement] = field(default_factory=list)
    expressions: list[Expression] = field(default_factory=list)


@dataclass(slots=True)
class TaggedTemplateExpression(Expression):
    tag: Expression
    quasi: TemplateLiteral


@dataclass(slots=True)
class TemplateElement(Node):
    tail: bool
    value: dict[str, Any]


# Patterns
@dataclass(kw_only=True, slots=True)
class AssignmentProperty(Property):
    kind: str = "init"
    method: bool = False


@dataclass(slots=True)
class ObjectPattern(Pattern):
    properties: list[Union[AssignmentProperty, RestElement]] = field(
        default_factory=list
    )


@dataclass(slots=True)
class ArrayPattern(Pattern):
    elements: list[Optional[Pattern]] = field(default_factory=list)


@dataclass(slots=True)
class RestElement(Pattern):
    argument: Pattern


@dataclass(slots=True)
class AssignmentPattern(Pattern):
    left: Pattern
    right: Expression


# Classes
@dataclass(slots=True)
class Class(Node):
    id: Optional[Identifier]
    superClass: Optional[Expression]
    body: ClassBody


@dataclass(slots=True)
class ClassBody(Node):
    body: list[Union[MethodDefinition, PropertyDefinition, StaticBlock]] = field(
        default_factory=list
    )


@dataclass(slots=True)
class PrivateIdentifier(Node):
    name: str


@dataclass(slots=True)
class PropertyDefinition(Node):
    key: Union[Expression, PrivateIdentifier]
    value: Optional[Expression]
    kind: str
    computed: bool
    static: bool


@dataclass(slots=True)
class MethodDefinition(Node):
    key: Union[Expression, PrivateIdentifier]
    value: FunctionExpression
    kind: str
    computed: bool
    static: bool


@dataclass(slots=True)
class ClassDeclaration(Class, Declaration):
    id: Identifier


@dataclass(slots=True)
class ClassExpression(Class, Expression):
    pass


@dataclass(slots=True)
class MetaProperty(Expression):
    meta: Identifier
    property: Identifier


# Modules
@dataclass(slots=True)
class ImportOrExportDeclaration(Node):
    pass


@dataclass(slots=True)
class ModuleSpecifier(Node):
    local: Identifier


@dataclass(slots=True)
class Import(Node):
    pass


@dataclass(kw_only=True, slots=True)
class ImportDeclaration(ImportOrExportDeclaration):
    specifiers: list[
        Union[ImportSpecifier, ImportDefaultSpecifier, ImportNamespaceSpecifier]
    ] = field(default_factory=list)
    source: Literal


@dataclass(slots=True)
class ImportSpecifier(ModuleSpecifier):
    imported: Union[Identifier, Literal]


@dataclass(slots=True)
class ImportDefaultSpecifier(ModuleSpecifier):
    pass


@dataclass(slots=True)
class ImportNamespaceSpecifier(ModuleSpecifier):
    pass


@dataclass(kw_only=True, slots=True)
class ExportNamedDeclaration(ImportOrExportDeclaration):
    declaration: Optional[Declaration]
    specifiers: list[ExportSpecifier] = field(default_factory=list)
    source: Optional[Literal]


@dataclass(slots=True)
class ExportSpecifier(ModuleSpecifier):
    local: Union[Identifier, Literal]
    exported: Union[Identifier, Literal]


@dataclass(slots=True)
class ExportDefaultDeclaration(ImportOrExportDeclaration):
    declaration: Union[FunctionDeclaration, ClassDeclaration, Expression]


@dataclass(slots=True)
class ExportAllDeclaration(ImportOrExportDeclaration):
    source: Literal
    exported: Optional[Union[Identifier, Literal]]
