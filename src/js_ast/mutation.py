import copy
import random
from hmac import new
from re import Pattern

import numpy as np
from torch import le

from js_ast.analysis import fix_node_references, scope_analysis
from js_ast.nodes import (
    AssignmentProperty,
    Expression,
    ExpressionStatement,
    ImportOrExportDeclaration,
    Node,
    Property,
    RestElement,
    SpreadElement,
    Statement,
    SwitchCase,
    VariableDeclarator,
)

node_add_types = {
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
}

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

    new_node = copy.deepcopy(random.choice(subtrees[target.type]))
    new_node.parent = target.parent

    scope_analysis(new_node, target.scope)
    fix_node_references(new_node)

    for field in target.parent.fields:
        val = getattr(target.parent, field)

        if isinstance(val, list):
            for i, item in enumerate(val):
                if item == target:
                    val[i] = new_node
                    return new_node
        elif val is target:
            setattr(target.parent, field, new_node)
            return new_node

    raise ValueError("Could not find target in parent")


def remove(target: Node) -> Node:
    if target.parent is None:
        return target

    for field in target.parent.fields:
        val = getattr(target.parent, field)

        if isinstance(val, list):
            for i, item in enumerate(val):
                if item == target:
                    if field in non_empty_nodes and len(val) == 1:
                        return target

                    val.pop(i)

                    # Re-analyze the scope of the parent as it may have changed
                    scope_analysis(target.parent, target.parent.scope)
                    return target.parent
        elif val == target:
            return target

    print("Could not find target in parent", target, target.parent)

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
    add_type = np.random.choice(types_name)
    new_node: Node = copy.deepcopy(random.choice(subtrees[add_type]))

    list_nodes = getattr(target, field)

    scope = target.end_scope if target.end_scope else target.scope
    new_node.parent = target

    scope_analysis(new_node, scope)
    fix_node_references(new_node)

    list_nodes.append(new_node)

    return new_node
