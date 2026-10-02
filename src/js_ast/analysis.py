import copy
import random
from typing import Optional

from js_ast.natives import is_native_name
from js_ast.nodes import ArrayPattern
from js_ast.nodes import ArrowFunctionExpression
from js_ast.nodes import AssignmentPattern
from js_ast.nodes import BreakStatement
from js_ast.nodes import BlockStatement
from js_ast.nodes import CallExpression
from js_ast.nodes import CatchClause
from js_ast.nodes import ClassBody
from js_ast.nodes import ClassDeclaration
from js_ast.nodes import ContinueStatement
from js_ast.nodes import FunctionDeclaration
from js_ast.nodes import FunctionExpression
from js_ast.nodes import Identifier
from js_ast.nodes import LabeledStatement
from js_ast.nodes import Literal
from js_ast.nodes import MemberExpression
from js_ast.nodes import MetaProperty
from js_ast.nodes import MethodDefinition
from js_ast.nodes import NewExpression
from js_ast.nodes import Node
from js_ast.nodes import ObjectPattern
from js_ast.nodes import Program
from js_ast.nodes import Property
from js_ast.nodes import PropertyDefinition
from js_ast.nodes import RestElement
from js_ast.nodes import Statement
from js_ast.nodes import UnaryExpression
from js_ast.nodes import VariableDeclaration
from js_ast.nodes import VariableDeclarator
from js_ast.scope import Scope
from js_ast.scope import ScopeType

from utils.interesting_values import interesting_floats
from utils.interesting_values import interesting_integers


# Only names d8 defines (with the fuzzer's flags): a name listed here is never
# renamed, so one the engine lacks (`require`, `log`, `quit` with --omit-quit,
# SpiderMonkey's `InternalError`, ...) would always throw a ReferenceError.
BUILTIN_FUNCTIONS = set(
    [
        "gc",
        "print",
        "eval",
        "parseInt",
        "parseFloat",
        "isNaN",
        "isFinite",
        "encodeURI",
        "encodeURIComponent",
        "decodeURIComponent",
    ]
)

BUILTIN_CONSTRUCTORS = set(
    [
        "Object",
        "Function",
        "Boolean",
        "Symbol",
        "Error",
        "EvalError",
        "RangeError",
        "ReferenceError",
        "SyntaxError",
        "TypeError",
        "URIError",
        "Number",
        "BigInt",
        "Math",
        "Date",
        "String",
        "RegExp",
        "Array",
        "Int8Array",
        "Uint8Array",
        "Uint8ClampedArray",
        "Int16Array",
        "Uint16Array",
        "Int32Array",
        "Uint32Array",
        "Float32Array",
        "Float64Array",
        "BigInt64Array",
        "BigUint64Array",
        "Map",
        "Set",
        "WeakMap",
        "WeakSet",
        "ArrayBuffer",
        "SharedArrayBuffer",
        "Atomics",
        "DataView",
        "JSON",
        "Promise",
        "Proxy",
        "Reflect",
        "WeakRef",
        "FinalizationRegistry",
        "AggregateError",
    ]
)

# random.choice needs a sequence, so keep sorted copies of the sets above
BUILTIN_FUNCTIONS_LIST = sorted(BUILTIN_FUNCTIONS)
BUILTIN_CONSTRUCTORS_LIST = sorted(BUILTIN_CONSTRUCTORS)

# Every global name a program can legitimately reference without declaring it.
# fix_node_references must never rename these, otherwise e.g. `new Array(10)`
# becomes `new v0(10)` and the program throws before reaching interesting code.
GLOBAL_NAMES = (
    BUILTIN_FUNCTIONS
    | BUILTIN_CONSTRUCTORS
    | {
        "undefined",
        "NaN",
        "Infinity",
        "arguments",
        "globalThis",
        "this",
        "decodeURI",
        "encodeURI",
        "escape",
        "unescape",
        "Intl",
        "WebAssembly",
        "Atomics",
        "Math",
        "console",
        # d8 shell builtins
        "load",
        "read",
        "readbuffer",
        "readline",
        "write",
        "version",
        "Realm",
        "os",
        "performance",
        "setTimeout",
        "testRunner",
    }
)


# Fallbacks for an undeclared name when nothing in scope can replace it. Calling
# these with any arguments doesn't throw, and assigning to the values is a silent
# no-op (outside strict mode), so they also work as assignment targets.
FALLBACK_FUNCTIONS = ["parseInt", "parseFloat", "isNaN", "isFinite", "String", "Number", "Boolean"]
FALLBACK_CONSTRUCTORS = ["Object", "Array", "Error", "Map", "Set", "Date", "RegExp"]
FALLBACK_VALUES = ["undefined", "NaN", "Infinity"]


FUNCTION_NODES = (FunctionDeclaration, FunctionExpression, ArrowFunctionExpression)


def _snapshot(scope: Scope) -> Scope:
    return Scope(
        scope.available_variables(),
        scope.available_functions(),
        scope.available_classes(),
    )


def pattern_names(pattern: Optional[Node]) -> list[str]:
    """Names bound by a binding pattern, e.g. `a`, `a = 1`, `...a`, `[a, {b}]`."""
    if pattern is None:
        return []
    if isinstance(pattern, Identifier):
        return [pattern.name]
    if isinstance(pattern, AssignmentPattern):
        return pattern_names(pattern.left)
    if isinstance(pattern, RestElement):
        return pattern_names(pattern.argument)
    if isinstance(pattern, ArrayPattern):
        return [name for el in pattern.elements for name in pattern_names(el)]
    if isinstance(pattern, ObjectPattern):
        names: list[str] = []
        for prop in pattern.properties:
            if isinstance(prop, RestElement):
                names += pattern_names(prop)
            else:
                names += pattern_names(prop.value)
        return names
    return []


def _analyse_block(node: Node, block_scope: Scope):
    # Hoist function and class declarations so they are visible to the whole block
    for item in node.body:
        if isinstance(item, FunctionDeclaration) and item.id is not None:
            block_scope.functions[item.id.name] = len(item.params)
        elif isinstance(item, ClassDeclaration) and item.id is not None:
            block_scope.classes.add(item.id.name)

    for item in node.body:
        scope_analysis(item, block_scope)

    # Store the scope at the end of the block so that it can be used for add mutation
    node.end_scope = _snapshot(block_scope)


def _analyse_function(node: Node, scope: Scope):
    # Parameters (and a function expression's own name) are only visible inside
    # the function, so they live in a new function scope rather than `scope`
    fn_scope = Scope(parent=scope, scope_type=ScopeType.FUNCTION)

    for param in node.params:
        fn_scope.variables.update(pattern_names(param))

    if isinstance(node, FunctionExpression) and node.id is not None:
        fn_scope.functions[node.id.name] = len(node.params)

    # Visit the parameters too (patterns and default values are evaluated in the
    # function scope), so every node inside them has a scope
    for param in node.params:
        scope_analysis(param, fn_scope)

    body = node.body
    if isinstance(body, BlockStatement):
        body.scope = _snapshot(fn_scope)
        _analyse_block(body, fn_scope)
    elif isinstance(body, Node):
        # Arrow function with an expression body
        scope_analysis(body, fn_scope)


# Calculates variables, classes and functions available at each node and stores it in
# a node attribute
def scope_analysis(node: Node, scope: Optional[Scope] = None):
    node_type = node.type
    if node_type in ("Literal", "Identifier"):
        return

    if scope is None:
        scope = Scope(scope_type=ScopeType.GLOBAL)

    node.scope = _snapshot(scope)

    if isinstance(node, FUNCTION_NODES):
        _analyse_function(node, scope)

    elif isinstance(node, (Program, BlockStatement, ClassBody)):
        if node_type == "Program":
            new_scope = scope
        elif node_type == "ClassBody":
            new_scope = Scope(parent=scope, scope_type=ScopeType.CLASS)
        else:
            new_scope = Scope(parent=scope, scope_type=ScopeType.BLOCK)

        _analyse_block(node, new_scope)

    elif isinstance(node, CatchClause):
        catch_scope = Scope(parent=scope, scope_type=ScopeType.BLOCK)
        catch_scope.variables.update(pattern_names(node.param))
        if node.param is not None:
            scope_analysis(node.param, catch_scope)
        scope_analysis(node.body, catch_scope)

    elif isinstance(node, VariableDeclarator) and isinstance(node.id, Identifier):
        if node.parent and isinstance(node.parent, VariableDeclaration):
            if node.init:
                scope_analysis(node.init, scope)

            if node.parent.kind == "var":
                current_scope = scope
                while (
                    current_scope.scope_type == ScopeType.BLOCK and current_scope.parent
                ):
                    current_scope.parent_variables.add(node.id.name)
                    # Add variable functions to scope functions
                    if node.init and isinstance(
                        node.init, (FunctionExpression, ArrowFunctionExpression)
                    ):
                        current_scope.parent_functions[node.id.name] = len(
                            node.init.params
                        )
                    current_scope = current_scope.parent

                current_scope.variables.add(node.id.name)
                if node.init and isinstance(
                    node.init, (FunctionExpression, ArrowFunctionExpression)
                ):
                    current_scope.parent_functions[node.id.name] = len(node.init.params)
            else:
                scope.variables.add(node.id.name)
                # Add variable functions to scope functions
                if node.init and isinstance(
                    node.init, (FunctionExpression, ArrowFunctionExpression)
                ):
                    scope.parent_functions[node.id.name] = len(node.init.params)
    else:
        for child in node.children():
            scope_analysis(child, scope)


def _is_binding_or_label(node: Identifier) -> bool:
    """Whether an identifier is a name being declared, a property name or a label,
    rather than a reference to a variable."""
    parent = node.parent

    if isinstance(parent, MemberExpression):
        return parent.property is node and not parent.computed
    if isinstance(parent, (Property, MethodDefinition, PropertyDefinition)):
        return parent.key is node and not parent.computed
    if isinstance(parent, (LabeledStatement, BreakStatement, ContinueStatement)):
        return parent.label is node
    if isinstance(parent, (CatchClause, MetaProperty)):
        return True
    if isinstance(parent, FUNCTION_NODES):
        return any(node is param for param in parent.params)
    if isinstance(parent, (RestElement, ArrayPattern, ObjectPattern)):
        return True
    if isinstance(parent, AssignmentPattern):
        return parent.left is node

    return False


def nearest_scope(node: Node) -> Optional[Scope]:
    """The scope of the closest node (itself or an ancestor) that has one. Every
    non-leaf node should have a scope after scope_analysis, but fall back rather
    than failing a long fuzzing run on an unanalysed corner of the AST."""
    current: Optional[Node] = node if node.type != "Identifier" else node.parent
    while current is not None:
        if current.scope is not None:
            return current.scope
        current = current.parent
    return None


def _is_known_name(name: str, scope: Scope) -> bool:
    return (
        name in scope.available_variables()
        or name in scope.available_functions()
        or name in scope.available_classes()
        or name in GLOBAL_NAMES
        or is_native_name(name)
    )


def _replacement_name(node: Identifier, scope: Scope) -> str:
    """A name in scope to replace an undeclared reference with. Prefers names of the
    same kind (functions for calls, classes for `new`, variables otherwise), then
    any other name in scope (functions and classes are values too), then a builtin,
    so the reference never stays undeclared."""
    variables = sorted(scope.available_variables())
    functions = sorted(scope.available_functions())
    classes = sorted(scope.available_classes())
    parent = node.parent

    if isinstance(parent, CallExpression) and parent.callee is node:
        # Calling a variable or a class (without `new`) would throw a TypeError
        return random.choice(functions or FALLBACK_FUNCTIONS)
    if isinstance(parent, NewExpression) and parent.callee is node:
        return random.choice(classes + functions or FALLBACK_CONSTRUCTORS)
    return random.choice(variables or functions + classes or FALLBACK_VALUES)


# Fixes the node by replacing identifiers and function calls with available variables and
# functions
def fix_node_references(
    node: Node, subtrees: dict[str, list[Node]], target: Optional[Node] = None
):
    if isinstance(node, Identifier):
        if node.parent is None or _is_binding_or_label(node):
            return

        scope = nearest_scope(node)
        if scope is None:
            return

        if not _is_known_name(node.name, scope):
            node.name = _replacement_name(node, scope)
    else:
        for child in node.children():
            fix_node_references(child, subtrees, target)


# Gets random literal or identifier
def random_value(scope: Scope, parent: Node, subtrees: dict[str, list[Node]]):
    if subtrees.get("Literal") and random.random() < 0.25:
        new_node = copy.deepcopy(random.choice(subtrees["Literal"]))
        new_node.parent = parent
        return new_node

    if scope.available_variables() and random.random() < 0.5:
        return Identifier(
            name=random.choice(list(scope.available_variables())),
            parent=parent,
        )
    else:
        # TODO: add more types and interesting values
        if random.random() < 0.5:
            value = random.choice(interesting_integers)
        else:
            value = random.choice(interesting_floats)

        if value < 0 or str(value).startswith("-"):
            value = -value
            literal = Literal(value=value, raw=str(value))
            return UnaryExpression(
                operator="-", argument=literal, prefix=True, parent=parent, scope=scope
            )

        return Literal(value=value, raw=str(value), parent=parent)


def strip_scopes(roots) -> int:
    """Drops scope information from ASTs that are only ever copied (the corpus and
    the subtree pool). Copies don't keep scopes and are re-analysed when used, but
    the stored scopes (sets of names at every node) take a lot of memory."""
    count = 0
    for root in roots:
        for node in root.traverse():
            if node.scope is not None or node.end_scope is not None:
                node.scope = None
                node.end_scope = None
                count += 1
    return count


def count_statements(root: Node):
    count = 0
    for node in root.traverse():
        if issubclass(node.__class__, Statement):
            count += 1

    return count
