import random
from typing import Optional

from js_ast.nodes import ArrowFunctionExpression
from js_ast.nodes import AssignmentPattern
from js_ast.nodes import BlockStatement
from js_ast.nodes import CallExpression
from js_ast.nodes import CatchClause
from js_ast.nodes import ClassBody
from js_ast.nodes import ClassDeclaration
from js_ast.nodes import FunctionDeclaration
from js_ast.nodes import FunctionExpression
from js_ast.nodes import Identifier
from js_ast.nodes import Literal
from js_ast.nodes import MemberExpression
from js_ast.nodes import Node
from js_ast.nodes import Program
from js_ast.nodes import Statement
from js_ast.nodes import UnaryExpression
from js_ast.nodes import VariableDeclaration
from js_ast.nodes import VariableDeclarator
from js_ast.scope import Scope
from js_ast.scope import ScopeType

from utils.interesting_values import interesting_floats
from utils.interesting_values import interesting_integers


BUILTIN_FUNCTIONS = set(
    [
        "gc",
        "print",
        "log",
        "exit",
        "quit",
        "eval",
        "require",
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
        "InternalError",
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
    ]
)


# Calculates variables, classes and functions available at each node and stores it in
# a node attribute
def scope_analysis(node: Node, scope: Optional[Scope] = None):
    node_type = node.type
    if node_type in set(["Literal", "Identifier"]):
        return

    if scope is None:
        scope = Scope(scope_type=ScopeType.GLOBAL)

    node.scope = Scope(
        scope.available_variables(),
        scope.available_functions(),
        scope.available_classes(),
    )

    if isinstance(node, (Program, BlockStatement, ClassBody)):
        if node_type == "Program":
            new_scope = scope
        elif node_type == "ClassBody":
            new_scope = Scope(parent=scope, scope_type=ScopeType.CLASS)
        elif (
            node_type == "BlockStatement"
            and node.parent
            and (
                node.parent.type == "FunctionDeclaration"
                or node.parent.type == "FunctionExpression"
                or node.parent.type == "ArrowFunctionExpression"
            )
        ):
            new_scope = Scope(parent=scope, scope_type=ScopeType.FUNCTION)
        else:
            new_scope = Scope(parent=scope, scope_type=ScopeType.BLOCK)

        for item in node.body:
            if isinstance(
                item, (FunctionDeclaration, FunctionExpression, ArrowFunctionExpression)
            ):
                for param in item.params:
                    if isinstance(param, Identifier):
                        new_scope.variables.add(param.name)
                    elif isinstance(param, AssignmentPattern) and isinstance(
                        param.left, Identifier
                    ):
                        new_scope.variables.add(param.left.name)

                new_scope.functions[item.id.name] = len(item.params)
            elif isinstance(item, ClassDeclaration):
                new_scope.classes.add(item.id.name)
            # elif (
            #     isinstance(item, MethodDefinition)
            #     and item.kind != "constructor"
            #     and isinstance(item.key, Identifier)
            # ):
            #     new_scope.functions[item.key.name] = item.value.params

        for item in node.body:
            scope_analysis(item, new_scope)

        # Store the scope at the end of the block so that it can be used for add mutation
        node.end_scope = Scope(
            new_scope.available_variables(),
            new_scope.available_functions(),
            new_scope.available_classes(),
        )

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


# Fixes the node by replacing identifiers and function calls with available variables and
# functions
def fix_node_references(
    node: Node, subtrees: dict[str, list[Node]], target: Optional[Node] = None
):
    if isinstance(node, Identifier):
        if (
            isinstance(node.parent, MemberExpression) and node.parent.object != node
        ) or (isinstance(node.parent, CatchClause)):
            return

        if isinstance(
            node.parent,
            (FunctionDeclaration, FunctionExpression, ArrowFunctionExpression),
        ):
            if node in node.parent.params:
                return

        scope = node.parent.scope

        if not scope:
            print(node.parent)
            raise Exception("Scope not found")

        available_variables = scope.available_variables()

        if available_variables and node.name not in available_variables:
            node.name = random.choice(list(available_variables))

    elif isinstance(node, CallExpression) and isinstance(node.callee, Identifier):
        scope = node.scope

        if not scope:
            print(node.parent)
            raise Exception("Scope not found")

        available_functions = scope.available_functions()

        if (
            available_functions
            and node.callee.name not in available_functions
            and node.callee.name not in BUILTIN_FUNCTIONS
            and node.callee.name not in BUILTIN_CONSTRUCTORS
        ):
            function, num_params = random.choice(list(available_functions.items()))
            node.callee.name = function
            if target and target.parent == node:
                node.arguments = [target] + [
                    random_value(scope, node, subtrees) for _ in range(num_params - 1)
                ]
            else:
                node.arguments = [
                    random_value(scope, node, subtrees) for _ in range(num_params)
                ]
        else:
            for node in node.arguments:
                fix_node_references(node, subtrees, target)
    else:
        for child in node.children():
            fix_node_references(child, subtrees, target)


# Gets random literal or identifier
def random_value(scope: Scope, parent: Node, subtrees: dict[str, list[Node]]):
    if "Literal" in subtrees and random.random() < 0.25:
        new_node = random.choice(subtrees["Literal"])
        new_node.parent = parent

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


def count_statements(root: Node):
    count = 0
    for node in root.traverse():
        if issubclass(node.__class__, Statement):
            count += 1

    return count
