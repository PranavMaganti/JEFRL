"""Trasnforms AST dictionary into a tree of Node objects."""
from __future__ import annotations

import abc
import copy
import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, Generator, Optional, Union

from js_ast.scope import Scope


class UnknownNodeTypeError(Exception):
    """Raised if we encounter a node with an unknown type."""

    pass


estree_field_map = {
    "isAsync": "async",
    "awaitAllowed": "await",
}

context_fields = {"parent", "scope", "end_scope"}

# Set of children fields that should not be children
non_child_fields = {"id"}


@dataclass(kw_only=True)
class Node(abc.ABC):
    """Abstract Node class which defines node operations"""

    # loc: Optional[dict[str, int]]
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
        return [
            f.name for f in dataclasses.fields(self) if f.name not in context_fields
        ]

    def traverse(self) -> Generator[Node, None, None]:
        """Pre-order traversal of this node and all of its children."""
        yield self
        for field in self.fields:
            val = getattr(self, field)
            if isinstance(val, Node):
                yield from val.traverse()
            elif isinstance(val, list):
                node: Any
                for node in val:
                    if isinstance(node, Node):
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
        data: Union[None, dict[str, Any], list[dict[str, Any]]]
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

            fields = [
                f.name
                for f in dataclasses.fields(node_class)
                if f.name not in context_fields
            ]
            params = {}

            for field in fields:
                data_field = (
                    estree_field_map[field] if field in estree_field_map else field
                )

                if data_field not in data:
                    params[field] = None
                else:
                    params[field] = Node.from_dict(data[data_field])

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
        """Return the root node of the tre_getattr__(self, name: str):
        return Nonee."""
        node = self
        while node.parent:
            node = node.parent
        return node

    def __repr__(self) -> str:
        """String representation of the node."""
        return json.dumps(self.to_dict(), indent=4)

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), indent=4)

    def __getattr__(self, name: str):
        return None

    def __dir__(self):
        return list(self.__dict__.keys())

    def __iter__(self):
        return self.__iter__

    def __deepcopy__(self, _memo):
        return self.__class__(
            **copy.deepcopy(
                {
                    k: v
                    for k, v in self.__dict__.items()
                    if k not in {"parent", "scope", "end_scope"}
                }
            )
        )

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d: dict[str, Any]):
        self.__dict__.update(d)


@dataclass
class Pattern(Node):
    pass


@dataclass
class Expression(Node):
    pass


@dataclass
class Identifier(Expression, Pattern):
    name: str


@dataclass
class Literal(Expression):
    value: Any
    raw: str
    regex: Optional[dict[str, Any]] = None
    bigint: Optional[str] = None


@dataclass
class Program(Node):
    sourceType: str
    body: list[Union[Statement, ImportOrExportDeclaration]]


@dataclass(kw_only=True)
class Function(Node):
    id: Optional[Identifier] = None
    params: list[Pattern] = field(default_factory=list)
    generator: bool
    isAsync: bool = False


# Statements
@dataclass
class Statement(Node):
    pass


@dataclass
class ExpressionStatement(Statement):
    expression: Expression
    directive: str


@dataclass
class BlockStatement(Statement):
    body: list[Statement] = field(default_factory=list)


@dataclass
class StaticBlock(BlockStatement):
    pass


@dataclass
class FunctionBody(BlockStatement):
    body: list[Union[ExpressionStatement, Statement]]


@dataclass
class EmptyStatement(Statement):
    pass


@dataclass
class DebuggerStatement(Statement):
    pass


@dataclass
class WithStatement(Statement):
    object: Expression
    body: Statement


# Control Flow
@dataclass
class ReturnStatement(Statement):
    argument: Optional[Expression]


@dataclass
class LabeledStatement(Statement):
    label: Identifier
    body: Statement


@dataclass
class BreakStatement(Statement):
    label: Optional[Identifier]


@dataclass
class ContinueStatement(Statement):
    label: Optional[Identifier]


# Choice
@dataclass
class IfStatement(Statement):
    test: Expression
    consequent: Statement
    alternate: Optional[Statement]


@dataclass
class SwitchStatement(Statement):
    discriminant: Expression
    cases: list[SwitchCase] = field(default_factory=list)


@dataclass
class SwitchCase(Node):
    test: Optional[Expression]
    consequent: list[Statement] = field(default_factory=list)


# Exceptions
@dataclass
class ThrowStatement(Statement):
    argument: Expression


@dataclass
class TryStatement(Statement):
    block: BlockStatement
    handler: Optional[CatchClause]
    finalizer: Optional[BlockStatement]
    param: Optional[Pattern]


@dataclass
class CatchClause(Node):
    param: Pattern
    body: BlockStatement


# Loops
@dataclass
class WhileStatement(Statement):
    test: Expression
    body: Statement


@dataclass
class DoWhileStatement(Statement):
    body: Statement
    test: Expression


@dataclass(kw_only=True)
class ForStatement(Statement):
    init: Optional[Union[VariableDeclaration, Expression]]
    test: Optional[Expression]
    update: Optional[Expression]
    body: Statement


@dataclass
class ForInStatement(Statement):
    left: Union[VariableDeclaration, Pattern]
    right: Expression
    body: Statement


@dataclass
class ForOfStatement(ForInStatement):
    awaitAllowed: bool


# Declarations
@dataclass
class Declaration(Statement):
    pass


@dataclass(kw_only=True)
class FunctionDeclaration(Function, Declaration):
    id: Identifier
    expression: bool = False
    body: Union[FunctionBody, Expression]


@dataclass(kw_only=True)
class VariableDeclaration(Declaration):
    declarations: list[VariableDeclarator]
    kind: str


@dataclass
class VariableDeclarator(Node):
    id: Pattern
    init: Optional[Expression]


# Expressions
@dataclass
class ThisExpression(Expression):
    pass


@dataclass
class ArrayExpression(Expression):
    elements: list[Optional[Union[Expression, SpreadElement]]] = field(
        default_factory=list
    )


@dataclass
class ObjectExpression(Expression):
    properties: list[Union[Property, SpreadElement]] = field(default_factory=list)


@dataclass
class Property(Node):
    key: Expression
    value: Expression
    kind: str
    method: bool
    shorthand: bool
    computed: bool


@dataclass
class FunctionExpression(Function, Expression):
    expression: bool
    body: Union[FunctionBody, Expression]


# Unary operations
@dataclass
class UnaryExpression(Expression):
    operator: str
    prefix: bool
    argument: Expression


@dataclass
class UpdateExpression(Expression):
    operator: str
    argument: Expression
    prefix: bool


# Binary operations
@dataclass
class BinaryExpression(Expression):
    operator: str
    left: Union[Expression, PrivateIdentifier]
    right: Expression


@dataclass
class AssignmentExpression(Expression):
    operator: str
    left: Pattern
    right: Expression


@dataclass
class LogicalExpression(Expression):
    operator: str
    left: Expression
    right: Expression


@dataclass
class ChainExpression(Expression):
    expression: Expression


@dataclass
class ChainElement(Node):
    optional: bool


@dataclass
class MemberExpression(ChainElement):
    object: Union[Expression, Super]
    property: Union[Expression, PrivateIdentifier]
    computed: bool


@dataclass
class ConditionalExpression(Expression):
    test: Expression
    consequent: Expression
    alternate: Expression


@dataclass
class CallExpression(ChainElement):
    callee: Union[Expression, Super]
    arguments: list[Union[Expression, SpreadElement]] = field(default_factory=list)


@dataclass
class NewExpression(Expression):
    callee: Expression
    arguments: list[Union[Expression, SpreadElement]] = field(default_factory=list)


@dataclass
class SequenceExpression(Expression):
    expressions: list[Expression] = field(default_factory=list)


@dataclass
class Super(Node):
    pass


@dataclass
class SpreadElement(Node):
    argument: Expression


@dataclass(kw_only=True)
class ArrowFunctionExpression(Function, Expression):
    body: Union[FunctionBody, Expression]
    expression: bool
    generator: bool = False


@dataclass(kw_only=True)
class YieldExpression(Expression):
    argument: Optional[Expression]
    delegate: bool


@dataclass
class AwaitExpression(Expression):
    argument: Expression


@dataclass
class ImportExpression(Expression):
    source: Expression


# Template Literals
@dataclass
class TemplateLiteral(Expression):
    quasis: list[TemplateElement] = field(default_factory=list)
    expressions: list[Expression] = field(default_factory=list)


@dataclass
class TaggedTemplateExpression(Expression):
    tag: Expression
    quasi: TemplateLiteral


@dataclass
class TemplateElement(Node):
    tail: bool
    value: dict[str, Any]


# Patterns
@dataclass(kw_only=True)
class AssignmentProperty(Property):
    kind: str = "init"
    method: bool = False


@dataclass
class ObjectPattern(Pattern):
    properties: list[Union[AssignmentProperty, RestElement]] = field(
        default_factory=list
    )


@dataclass
class ArrayPattern(Pattern):
    elements: list[Optional[Pattern]] = field(default_factory=list)


@dataclass
class RestElement(Pattern):
    argument: Pattern


@dataclass
class AssignmentPattern(Pattern):
    left: Pattern
    right: Expression


# Classes
@dataclass
class Class(Node):
    id: Optional[Identifier]
    superClass: Optional[Expression]
    body: ClassBody


@dataclass
class ClassBody(Node):
    body: list[Union[MethodDefinition, PropertyDefinition, StaticBlock]] = field(
        default_factory=list
    )


@dataclass
class PrivateIdentifier(Node):
    name: str


@dataclass
class PropertyDefinition(Node):
    key: Union[Expression, PrivateIdentifier]
    value: Optional[Expression]
    kind: str
    computed: bool
    static: bool


@dataclass
class MethodDefinition(Node):
    key: Union[Expression, PrivateIdentifier]
    value: FunctionExpression
    kind: str
    computed: bool
    static: bool


@dataclass
class ClassDeclaration(Class, Declaration):
    id: Identifier


@dataclass
class ClassExpression(Class, Expression):
    pass


@dataclass
class MetaProperty(Expression):
    meta: Identifier
    property: Identifier


# Modules
@dataclass
class ImportOrExportDeclaration(Node):
    pass


@dataclass
class ModuleSpecifier(Node):
    local: Identifier


@dataclass(kw_only=True)
class ImportDeclaration(ImportOrExportDeclaration):
    specifiers: list[
        Union[ImportSpecifier, ImportDefaultSpecifier, ImportNamespaceSpecifier]
    ] = field(default_factory=list)
    source: Literal


@dataclass
class ImportSpecifier(ModuleSpecifier):
    imported: Union[Identifier, Literal]


@dataclass
class ImportDefaultSpecifier(ModuleSpecifier):
    pass


@dataclass
class ImportNamespaceSpecifier(ModuleSpecifier):
    pass


@dataclass(kw_only=True)
class ExportNamedDeclaration(ImportOrExportDeclaration):
    declaration: Optional[Declaration]
    specifiers: list[ExportSpecifier] = field(default_factory=list)
    source: Optional[Literal]


@dataclass
class ExportSpecifier(ModuleSpecifier):
    local: Union[Identifier, Literal]
    exported: Union[Identifier, Literal]


@dataclass
class ExportDefaultDeclaration(ImportOrExportDeclaration):
    declaration: Union[FunctionDeclaration, ClassDeclaration, Expression]


@dataclass
class ExportAllDeclaration(ImportOrExportDeclaration):
    source: Literal
    exported: Optional[Union[Identifier, Literal]]
