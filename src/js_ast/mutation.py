import copy
import random
from re import Pattern
from typing import Tuple

from js_ast.analysis import fix_node_references
from js_ast.analysis import scope_analysis
from js_ast.nodes import AssignmentProperty
from js_ast.nodes import Expression
from js_ast.nodes import ExpressionStatement
from js_ast.nodes import ImportOrExportDeclaration
from js_ast.nodes import Node
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
}


def replace(subtrees: dict[str, list[Node]], target: Node) -> Node:
    if target.parent is None:
        return target

    if target.type not in subtrees:
        return target

    root = target.root()
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
        print(target)
        print(target.parent)
        raise ValueError("Could not find target in parent")

    scope_analysis(root)
    # Fix references in all nodes as we may have replaced function/variable declarations
    fix_node_references(root)

    return new_node


def remove(target: Node) -> Node:
    if target.parent is None:
        return target

    for field in target.parent.fields:
        val = getattr(target.parent, field)

        if isinstance(val, list):
            for i, item in enumerate(val):
                if item is target:
                    if field in non_empty_nodes and len(val) == 1:
                        return target

                    val.pop(i)

                    # Re-analyze the scope of the parent as it may have changed
                    root = target.root()

                    scope_analysis(root)
                    # Fix references in all nodes as we may have removed function/variable declarations
                    fix_node_references(root)

                    return target.parent
        elif val is target:
            return target

    raise ValueError("Could not find target in parent")


def add(subtrees: dict[str, list[Node]], target: Node) -> Node:
    if target.type not in node_add_types:
        return target

    field, types = node_add_types[target.type]
    for t in types:
        types += t.__subclasses__()

    types_name = [
        t.__name__
        for t in types
        if t.__name__ in subtrees and t.__name__ not in non_add_types
    ]

    if len(types_name) == 0:
        return target

    add_type = np.random.choice(types_name)
    new_node: Node = copy.deepcopy(random.choice(subtrees[add_type]))

    list_nodes = getattr(target, field)

    new_node.parent = target
    list_nodes.append(new_node)

    root = target.root()
    scope_analysis(root)
    # Only fix references of new node since we are adding it to the tree
    fix_node_references(new_node)

    return new_node
