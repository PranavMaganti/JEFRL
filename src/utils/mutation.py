from re import Pattern

import numpy as np
from attr import has

from nodes.main import (
    Directive,
    Expression,
    Node,
    Statement,
    SwitchCase,
    VariableDeclarator,
)

node_add_types = {
    "Program": ("body", [Directive, Statement]),
    "Function": ("params", [Pattern]),
    "BlockStatement": ("body", [Statement]),
    "FunctionBody": ("body", [Directive, Statement]),
    "SwitchStatement": ("cases", [SwitchCase]),
    "SwitchCase": ("consequent", [Statement]),
    "VariableDeclaration": ("declarations", [VariableDeclarator]),
    "ArrayExpression": ("elements", [Expression]),
    "ObjectExpression": ("properties", [Expression]),
    "CallExpression": ("arguments", [Expression]),
    "NewExpression": ("arguments", [Expression]),
    "SequenceExpression": ("expressions", [Expression]),
}


def replace(subtrees: dict[str, list[Node]], target: Node) -> Node:
    if target.parent is None:
        return target

    new_node = np.random.choice(subtrees[target.type])

    for field in target.parent.fields:
        val = getattr(target.parent, field)

        if isinstance(val, list):
            for i, item in enumerate(val):
                if item is target:
                    val[i] = new_node
                    return target
        elif val is target:
            setattr(target.parent, field, new_node)
            return target

    raise ValueError("Could not find target in parent")


def remove(target: Node) -> Node:
    if target.parent is None:
        return target

    for field in target.parent.fields:
        val = getattr(target.parent, field)

        if isinstance(val, list):
            for i, item in enumerate(val):
                if item is target:
                    val.pop(i)
                    return target.parent
        elif val is target:
            return target

    raise ValueError("Could not find target in parent")


def add(subtrees: dict[str, list[Node]], target: Node) -> Node:
    if target.parent is None or not target.type in node_add_types:
        return target

    field, types = node_add_types[target.type]
    types = types + [t.__subclasses__() for t in types]
    types_name = [t.__name__ for t in types if t.__name__ in subtrees]
    add_type = np.random.choice(types_name)
    new_node = np.random.choice(subtrees[add_type])

    for field in target.parent.fields:
        val = getattr(target.parent, field)

        if isinstance(val, list):
            for i, item in enumerate(val):
                if item is target:
                    val.insert(i, new_node)
                    return target
        elif val is target:
            setattr(target.parent, field, new_node)
            return target

    raise ValueError("Could not find target in parent")
