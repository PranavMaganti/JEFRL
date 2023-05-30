import hashlib
from typing import Any, Optional

from js_ast.nodes import Node

TERM_TYPE = [
    "DebuggerStatement",
    "ThisExpression",
    "Super",
    "EmptyStatement",
    "Import",
]


def node_to_frags(
    node: Node,
    frag_seq: list[dict[str, Any]],
    frag_info_seq: list[tuple[int, str]],
    node_types: set[str],
    stack: Optional[list[tuple[int, str]]] = None,
):
    """
    Converts a js_ast Node into a list of fragments of height 1
    """

    if stack is None:
        stack = [(-1, node.type)]

    # Append the node before visiting its children
    frag: dict[str, Any] = dict()
    frag_idx = len(frag_seq)
    frag_seq.append(frag)

    # Push node info into the stack
    if len(stack) > 0:
        frag_info = stack.pop()
        frag_info_seq.append(frag_info)

    push(stack, node, frag_idx)

    node_types.add(node.type)

    for key in node.fields:
        val = getattr(node, key)

        # If it has a single child
        if isinstance(val, Node):
            if val.type in TERM_TYPE:
                frag[key] = val.to_dict()
            else:
                frag[key] = {"type": val.type}
                node_to_frags(val, frag_seq, frag_info_seq, node_types, stack)

        # If it has multiple children
        elif isinstance(val, list):
            frag[key] = []
            for child in val:
                if isinstance(child, Node):
                    if child.type in TERM_TYPE:
                        frag[key].append(child.to_dict())
                    else:
                        frag[key].append({"type": child.type})
                        node_to_frags(child, frag_seq, frag_info_seq, node_types, stack)
                else:
                    frag[key].append(child)

        # If it is a terminal
        else:
            frag[key] = getattr(node, key)


def push(stack: list[tuple[int, str]], node: Node, parent_idx: int):
    for key in reversed(node.fields):
        val = getattr(node, key)

        # If it has a single child
        if isinstance(val, Node) and val.type not in TERM_TYPE:
            frag_info = (parent_idx, val.type)
            stack.append(frag_info)

        # If it has multiple children
        elif isinstance(val, list):
            for child in reversed(val):
                if isinstance(child, Node) and child.type not in TERM_TYPE:
                    frag_info = (parent_idx, child.type)
                    stack.append(frag_info)


def frag_to_str(frag: dict[str, Any]) -> str:
    out: str = ""
    for key, val in frag.items():
        # If it has a single child
        if isinstance(val, dict):
            out += "{" + frag_to_str(val) + "}"
        # If it has multiple children
        elif isinstance(val, list):
            child_str = [
                frag_to_str(child) if child is not None else "None" for child in val
            ]
            out += "[" + ",".join(child_str) + "]"
        # If it is a terminal
        else:
            out += str((key, val))

    return out


def hash_frag(frag: dict[str, Any]):
    frag_text = frag_to_str(frag)
    return hashlib.sha256(frag_text.encode("utf-8")).hexdigest()
