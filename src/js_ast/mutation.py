import copy
import random
from typing import Tuple

from js_ast.analysis import BUILTIN_CONSTRUCTORS
from js_ast.analysis import fix_node_references
from js_ast.analysis import random_value
from js_ast.analysis import scope_analysis
from js_ast.nodes import AssignmentExpression
from js_ast.nodes import AssignmentProperty
from js_ast.nodes import BinaryExpression
from js_ast.nodes import Expression
from js_ast.nodes import ExpressionStatement
from js_ast.nodes import Identifier
from js_ast.nodes import ImportOrExportDeclaration
from js_ast.nodes import Literal
from js_ast.nodes import Node
from js_ast.nodes import Pattern
from js_ast.nodes import Property
from js_ast.nodes import RestElement
from js_ast.nodes import SpreadElement
from js_ast.nodes import Statement
from js_ast.nodes import SwitchCase
from js_ast.nodes import VariableDeclarator
import numpy as np


node_add_types: dict[str, Tuple[str, list[Node]]] = {
    "Program": ("body", [Statement, ImportOrExportDeclaration]),
    "Function": ("params", [Pattern]),
    "BlockStatement": ("body", [Statement]),
    "FunctionBody": ("body", [ExpressionStatement, Statement]),
    "SwitchStatement": ("cases", [SwitchCase]),
    "SwitchCase": ("consequent", [Statement]),
    "VariableDeclaration": ("declarations", [VariableDeclarator]),
    "ArrayExpression": ("elements", [Expression, SpreadElement]),
    "ObjectExpression": ("properties", [Property, SpreadElement]),
    "CallExpression": ("arguments", [Expression, SpreadElement]),
    "NewExpression": ("arguments", [Expression, SpreadElement]),
    "SequenceExpression": ("expressions", [Expression]),
    "TemplateLiteral": ("expressions", [Expression]),
    "ObjectPattern": ("properties", [AssignmentProperty, RestElement]),
    "ArrayPattern": ("elements", [Pattern]),
}  # type: ignore

non_empty_nodes = {"declarations"}
non_add_types = {
    "ReturnStatement",
    "ThrowStatement",
    "BreakStatement",
    "ContinueStatement",
    "YieldExpression",
    "Super",
}

REPLACE_FUNCTIONS = [
    "parseInt",
    "parseFloat",
    "isNaN",
    "isFinite",
    "encodeURI",
    "decodeURI",
    "eval",
    "require",
    "gc",
]


def replace(
    subtrees: dict[str, list[Node]], target: Node, root: Node
) -> tuple[Node, bool]:
    if target.parent is None:
        return target, False

    if target.type not in subtrees:
        return target, False

    if target.type == "Literal" or target.type == "Identifier":
        scope_analysis(root)
        scope = target.parent.scope

        if target.parent.type == "CallExpression" and target.parent.callee is target:
            available_functions = scope.available_functions()

            if available_functions:
                function = random.choice(list(available_functions.keys()))
            else:
                function = random.choice(REPLACE_FUNCTIONS)

            new_node = Identifier(name=function, parent=target.parent)

        elif target.parent.type == "NewExpression" and target.parent.left is target:
            available_classes = scope.available_classes()

            if available_classes:
                function = random.choice(list(available_classes))
            else:
                function = random.choice(BUILTIN_CONSTRUCTORS)

            new_node = Identifier(name=function, parent=target.parent)
        else:
            new_node = random_value(scope, target.parent, subtrees)
    else:
        new_node = copy.deepcopy(random.choice(subtrees[target.type]))
        new_node.parent = target.parent

    # print(target)
    # print(target.parent)
    # print(target.parent.parent)
    # print(new_node)

    # TODO: Tidy up this code by possibly adding field to parent property of node
    # which indicates which field in the parent the child belongs to
    found = False
    for field in target.parent.fields:
        val = getattr(target.parent, field)
        if isinstance(val, list):
            for i, item in enumerate(val):
                if item is target:
                    val[i] = new_node
                    found = True
                    break
        elif val is target:
            setattr(target.parent, field, new_node)
            found = True

        if found:
            break

    if not found:
        print(target)
        print(target.parent)
        raise ValueError("Could not find target in parent")

    # print(target.parent)
    scope_analysis(root)
    # Fix references in all nodes as we may have replaced function/variable declarations
    fix_node_references(root, subtrees, new_node)
    # print(target.parent)
    # print(target.parent.parent)

    return new_node, True


def remove(
    subtrees: dict[str, list[Node]], target: Node, root: Node
) -> tuple[Node, bool]:
    if target.parent is None:
        return target, False

    # print(target)
    # print(target.parent)

    for field in target.parent.fields:
        val = getattr(target.parent, field)

        if isinstance(val, list):
            for i, item in enumerate(val):
                if item is target:
                    if field in non_empty_nodes and len(val) == 1:
                        return target, False

                    val.pop(i)

                    # Re-analyze the scope of the parent as it may have changed
                    scope_analysis(root)
                    # print(target.parent)
                    # print(target)
                    # Fix references in all nodes as we may have removed function/variable declarations
                    fix_node_references(root, subtrees, target.parent)

                    return target.parent, True
        elif val is target:
            return target, False

    raise ValueError("Could not find target in parent")


def add(subtrees: dict[str, list[Node]], target: Node, root: Node) -> tuple[Node, bool]:
    if target.type not in node_add_types:
        return target, False

    field, types = node_add_types[target.type]

    sub_types = set(types)
    for t in types:
        sub_types.update(t.__subclasses__())

    candidate_types = [
        t.__name__
        for t in sub_types
        if t.__name__ in subtrees and t.__name__ not in non_add_types
    ]

    if len(candidate_types) == 0:
        return target, False

    add_type = np.random.choice(candidate_types)
    new_node: Node = copy.deepcopy(random.choice(subtrees[add_type]))

    list_nodes = getattr(target, field)

    new_node.parent = target
    list_nodes.append(new_node)

    scope_analysis(root)
    # Only fix references of new node since we are adding it to the tree
    fix_node_references(new_node, subtrees)

    return new_node, True


BINARY_OPERATORS = ["+", "-", "*", "/", "%", "**", "<<", ">>", ">>>", "&", "|", "^"]
COMPARISON_OPERATORS = ["==", "!=", "===", "!==", ">", ">=", "<", "<="]
LOGICAL_OPERATORS = ["&&", "||", "??"]
ASSIGNMENT_OPERATORS = ["=", "+=", "-=", "*=", "/=", "%=", "**="]


def modify(target: Node) -> bool:
    if isinstance(target, BinaryExpression):
        target.operator = np.random.choice(
            BINARY_OPERATORS + COMPARISON_OPERATORS + LOGICAL_OPERATORS
        )
        return True
    elif isinstance(target, AssignmentExpression):
        target.operator = np.random.choice(ASSIGNMENT_OPERATORS)
        return True

    return False
