import random
from typing import Optional

from js_ast.nodes import AssignmentPattern
from js_ast.nodes import BlockStatement
from js_ast.nodes import CallExpression
from js_ast.nodes import ClassBody
from js_ast.nodes import ClassDeclaration
from js_ast.nodes import FunctionDeclaration
from js_ast.nodes import Identifier
from js_ast.nodes import Literal
from js_ast.nodes import Node
from js_ast.nodes import Program
from js_ast.nodes import UnaryExpression
from js_ast.nodes import VariableDeclaration
from js_ast.nodes import VariableDeclarator
from js_ast.scope import Scope
from js_ast.scope import ScopeType

from utils.interesting_values import interesting_floats
from utils.interesting_values import interesting_integers


# Calculates variables, classes and functions available at each node and stores it in
# a node attribute
def scope_analysis(node: Node, scope: Optional[Scope] = None):
    if scope is None:
        scope = Scope(scope_type=ScopeType.GLOBAL)

    node.scope = Scope(
        scope.available_variables(),
        scope.available_functions(),
        scope.available_classes(),
    )

    if (
        isinstance(node, Program)
        or isinstance(node, BlockStatement)
        or isinstance(node, ClassBody)
    ):
        if node.type == "Program":
            new_scope = scope
        elif node.type == "ClassBody":
            new_scope = Scope(parent=scope, scope_type=ScopeType.CLASS)
        elif (
            node.type == "BlockStatement"
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
            if isinstance(item, FunctionDeclaration):
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
                    current_scope = current_scope.parent

                current_scope.variables.add(node.id.name)
            else:
                scope.variables.add(node.id.name)
    else:
        for child in node.children():
            scope_analysis(child, scope)


# Fixes the node by replacing identifiers and function calls with available variables and
# functions
def fix_node_references(node: Node):
    if not node.scope:
        print(node)
        exit(1)

    scope = node.scope
    if isinstance(node, Identifier):
        if scope.available_variables() and node.name not in scope.available_variables():
            node.name = random.choice(list(scope.available_variables()))

    elif isinstance(node, CallExpression) and isinstance(node.callee, Identifier):
        if (
            scope.available_functions()
            and node.callee.name not in scope.available_functions()
        ):
            function, num_params = random.choice(
                list(scope.available_functions().items())
            )
            node.callee.name = function
            node.arguments = [random_value(scope, node) for _ in range(num_params)]
    else:
        for child in node.children():
            fix_node_references(child)


# Gets random literal or identifier
def random_value(scope: Scope, parent: Node):
    if scope.available_variables() and random.random() < 0.5:
        return Identifier(
            name=random.choice(list(scope.available_variables())),
            parent=parent,
            scope=scope,
        )
    else:
        # TODO: add more types and interesting values
        if random.random() < 0.5:
            value = random.choice(interesting_integers)
        else:
            value = random.choice(interesting_floats)

        if value < 0:
            value = -value
            literal = Literal(value=value, raw=str(value), scope=scope)
            return UnaryExpression(
                operator="-", argument=literal, prefix=True, parent=parent, scope=scope
            )

        return Literal(value=value, raw=str(value), scope=scope)
