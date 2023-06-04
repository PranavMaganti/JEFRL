from typing import Optional

from js_ast.nodes import Identifier
from js_ast.nodes import MemberExpression
from js_ast.nodes import Node


def is_declared_id(node: Node, prop: Optional[str]) -> bool:
    if not isinstance(node, Identifier) or node.parent is None:
        return False

    # var, const, let, func
    if node.parent.type in [
        "VariableDeclarator",
        "FunctionDeclaration",
        "ClassDeclaration",
    ]:
        return prop == "id"
    # Assignment Expression
    elif node.parent.type == "AssignmentExpression":
        return prop == "left"

    return False


def add_id(org_id: str, id_dict: dict[str, str], id_cnt: dict[str, int], id_type: str):
    if org_id not in id_dict:
        id_idx = id_cnt[id_type]
        id_cnt[id_type] += 1
        id_dict[org_id] = f"{id_type}{id_idx}"


def collect_id(
    node: Node,
    id_dict: dict[str, str],
    id_cnt: dict[str, int],
    prop: Optional[str] = None,
):
    for field in node.fields:
        val = getattr(node, field)

        if isinstance(val, Node):
            collect_id(val, id_dict, id_cnt, field)
        elif isinstance(val, list):
            for child in val:
                if isinstance(child, Node):
                    collect_id(child, id_dict, id_cnt, field)

    if node.parent is not None and is_declared_id(node, prop):
        assert isinstance(node, Identifier)

        match node.parent.type:
            case "FunctionDeclaration":
                id_type = "f"
            case "ClassDeclaration":
                id_type = "c"
            case _:
                id_type = "v"

        add_id(node.name, id_dict, id_cnt, id_type)


def normalize_id(node: Node, id_dict: dict[str, str], prop: Optional[str] = None):
    if node.type == "ObjectPattern":
        return

    for key in node.fields:
        val = getattr(node, key)

        # Traversal
        if isinstance(val, Node):
            normalize_id(val, id_dict, key)
        elif isinstance(val, list):
            for child in val:
                if isinstance(child, Node):
                    normalize_id(child, id_dict, key)

    # Exit if the node is not an ID
    if node.type != "Identifier" or not node.parent:
        return

    # Exit if the node is a property of an object
    if (
        isinstance(node.parent, MemberExpression)
        and prop != "object"
        and not node.parent.computed
    ):
        return

    # Do not normalize keys (ObjectExpression)
    if node.parent.type == "Property" and prop == "key":
        return

    # Replace the ID
    id_name = getattr(node, "name")
    if id_name in id_dict:
        setattr(node, "name", id_dict[id_name])
