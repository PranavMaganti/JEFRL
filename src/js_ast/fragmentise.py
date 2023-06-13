import hashlib
from typing import Any, Optional

from js_ast.nodes import Node


TERM_TYPE = set(
    [
        "DebuggerStatement",
        "ThisExpression",
        "Super",
        "EmptyStatement",
        "Import",
    ]
)


def node_to_frags(
    node: Node,
    frag_seq: list[dict[str, Any]],
    node_types: set[str],
    max_len: Optional[int] = None,
):
    """
    Converts a js_ast Node into a list of fragments of height 1
    """
    if max_len is not None and len(frag_seq) >= max_len:
        return

    # Append the node before visiting its children
    frag: dict[str, Any] = dict()
    frag_seq.append(frag)

    node_types.add(node.type)

    for key in node.fields:
        val = getattr(node, key)

        # If it has a single child
        if isinstance(val, Node):
            if val.type in TERM_TYPE:
                frag[key] = val.to_dict()
            else:
                frag[key] = {"type": val.type}
                node_to_frags(val, frag_seq, node_types, max_len)

        # If it has multiple children
        elif isinstance(val, list):
            frag[key] = []
            for child in val:
                if isinstance(child, Node):
                    if child.type in TERM_TYPE:
                        frag[key].append(child.to_dict())
                    else:
                        frag[key].append({"type": child.type})
                        node_to_frags(child, frag_seq, node_types, max_len)
                else:
                    frag[key].append(child)

        # If it is a terminal
        else:
            frag[key] = getattr(node, key)


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
