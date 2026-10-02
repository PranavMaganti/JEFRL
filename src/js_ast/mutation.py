import copy
import random
from typing import Tuple

from js_ast.analysis import BUILTIN_CONSTRUCTORS_LIST
from js_ast.analysis import fix_node_references
from js_ast.analysis import nearest_scope
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
from js_ast.nodes import LogicalExpression
from js_ast.nodes import Node
from js_ast.nodes import Pattern
from js_ast.nodes import Property
from js_ast.nodes import RestElement
from js_ast.nodes import SpreadElement
from js_ast.nodes import Statement
from js_ast.nodes import SwitchCase
from js_ast.nodes import UnaryExpression
from js_ast.nodes import UpdateExpression
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
        # Nodes inside parts of the AST that scope analysis skips have no scope
        scope = nearest_scope(target)
        if scope is None:
            return target, False

        if target.parent.type == "CallExpression" and target.parent.callee is target:
            available_functions = scope.available_functions()

            if available_functions:
                function = random.choice(sorted(available_functions.keys()))
            else:
                function = random.choice(REPLACE_FUNCTIONS)

            new_node = Identifier(name=function, parent=target.parent)

        elif target.parent.type == "NewExpression" and target.parent.callee is target:
            available_classes = scope.available_classes()

            if available_classes:
                function = random.choice(sorted(available_classes))
            else:
                function = random.choice(BUILTIN_CONSTRUCTORS_LIST)

            new_node = Identifier(name=function, parent=target.parent)
        else:
            new_node = random_value(scope, target.parent, subtrees)
    else:
        new_node = copy.deepcopy(random.choice(subtrees[target.type]))
        new_node.parent = target.parent

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
        raise ValueError("Could not find target in parent")

    scope_analysis(root)
    # Fix references in all nodes as we may have replaced function/variable declarations
    fix_node_references(root, subtrees, new_node)

    return new_node, True


def remove(
    subtrees: dict[str, list[Node]], target: Node, root: Node
) -> tuple[Node, bool]:
    if target.parent is None:
        return target, False

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
                    # Fix references in all nodes as we may have removed function/variable declarations
                    fix_node_references(root, subtrees, target.parent)

                    return target.parent, True
        elif val is target:
            return target, False

    raise ValueError("Could not find target in parent")


def add(subtrees: dict[str, list[Node]], target: Node, root: Node) -> tuple[Node, bool]:
    candidate_types = add_candidate_types(subtrees, target)
    if not candidate_types:
        return target, False

    field, _ = node_add_types[target.type]
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
# `??` is left out: mixing it with && or || without brackets is a SyntaxError
LOGICAL_OPERATORS = ["&&", "||"]
ASSIGNMENT_OPERATORS = [
    "=",
    "+=",
    "-=",
    "*=",
    "/=",
    "%=",
    "**=",
    "<<=",
    ">>=",
    ">>>=",
    "&=",
    "|=",
    "^=",
]
UNARY_OPERATORS = ["-", "+", "!", "~", "typeof", "void"]
UPDATE_OPERATORS = ["++", "--"]

MODIFIABLE_TYPES = (
    BinaryExpression,
    LogicalExpression,
    AssignmentExpression,
    UnaryExpression,
    UpdateExpression,
)


def modify(target: Node) -> bool:
    if isinstance(target, BinaryExpression):
        # `in` and `instanceof` are also BinaryExpressions, rewriting them is fine
        target.operator = str(
            np.random.choice(BINARY_OPERATORS + COMPARISON_OPERATORS)
        )
        return True
    elif isinstance(target, LogicalExpression):
        target.operator = str(np.random.choice(LOGICAL_OPERATORS))
        return True
    elif isinstance(target, AssignmentExpression):
        # Destructuring assignments only support `=`
        if target.left is not None and target.left.type in (
            "ObjectPattern",
            "ArrayPattern",
        ):
            return False
        target.operator = str(np.random.choice(ASSIGNMENT_OPERATORS))
        return True
    elif isinstance(target, UnaryExpression):
        # `delete` changes semantics too much and has strict-mode restrictions
        if target.operator == "delete":
            return False
        target.operator = str(np.random.choice(UNARY_OPERATORS))
        return True
    elif isinstance(target, UpdateExpression):
        target.operator = str(np.random.choice(UPDATE_OPERATORS))
        return True

    return False


# The functions below mirror the preconditions of the mutations above without
# changing the AST, so the environment can mask out actions which would fail.


def add_candidate_types(subtrees: dict[str, list[Node]], target: Node) -> list[str]:
    if target.type not in node_add_types:
        return []

    _, types = node_add_types[target.type]

    sub_types = set(types)
    for t in types:
        sub_types.update(t.__subclasses__())

    return sorted(
        t.__name__
        for t in sub_types
        if subtrees.get(t.__name__) and t.__name__ not in non_add_types
    )


def can_replace(subtrees: dict[str, list[Node]], target: Node) -> bool:
    return target.parent is not None and bool(subtrees.get(target.type))


def can_add(subtrees: dict[str, list[Node]], target: Node) -> bool:
    return bool(add_candidate_types(subtrees, target))


def can_remove(target: Node) -> bool:
    if target.parent is None:
        return False

    for field in target.parent.fields:
        val = getattr(target.parent, field)
        if isinstance(val, list):
            if any(item is target for item in val):
                return not (field in non_empty_nodes and len(val) == 1)
        elif val is target:
            return False

    return False


def can_modify(target: Node) -> bool:
    if isinstance(target, AssignmentExpression):
        return target.left is None or target.left.type not in (
            "ObjectPattern",
            "ArrayPattern",
        )
    if isinstance(target, UnaryExpression):
        return target.operator != "delete"
    return isinstance(target, MODIFIABLE_TYPES)
