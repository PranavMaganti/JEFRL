from collections import defaultdict
from itertools import count
from typing import Optional

import esprima
import graphviz
from js_ast.nodes import Node
from sympy import false


def draw_node(
    root: Node,
    graph: graphviz.Graph,
    count_dict: dict[str, int],
    target_node: Optional[Node] = None,
    context_node: Optional[Node] = None,
    node_attrs: dict[str, str] = {},
):
    graph.node(str(id(root)), root.type, **node_attrs)

    for field in root.fields[1:]:
        item = getattr(root, field)
        if isinstance(item, Node):
            graph.edge(str(id(root)), str(id(item)), label=field)
            visualise_ast(item, graph, count_dict, target_node, context_node)
        elif isinstance(item, list):
            for i in item:
                if isinstance(i, Node):
                    graph.edge(str(id(root)), str(id(i)), label=field)
                    visualise_ast(i, graph, count_dict, target_node, context_node)
        else:
            if isinstance(item, str):
                item = f'"{item}"'
            else:
                item = str(item)

            item_id = count_dict["count"]
            count_dict["count"] += 1

            graph.node(str(item_id), item)
            graph.edge(str(id(root)), str(item_id), label=field)


def visualise_ast(
    root: Node,
    graph: graphviz.Graph,
    count_dict: dict[str, int],
    target_node: Optional[Node] = None,
    context_node: Optional[Node] = None,
):
    node_attrs = {}
    if target_node is not None and root is target_node:
        with graph.subgraph(name="cluster_target") as c_target:
            c_target.attr(color="blue", style="rounded")
            c_target.node_attr["style"] = "filled"
            draw_node(root, c_target, count_dict, target_node, context_node, node_attrs)
            c_target.attr(label="Target Node", labeljust="r", fontsize="20")

    elif context_node is not None and root is context_node:
        with graph.subgraph(name="cluster_context") as c_context:
            c_context.attr(style="filled", color="lightgrey")
            c_context.node_attr.update(style="filled", color="white")
            draw_node(
                root, c_context, count_dict, target_node, context_node, node_attrs
            )
            c_context.attr(label="Context Node", labeljust="r", fontsize="20")

    else:
        draw_node(root, graph, count_dict, target_node, context_node, node_attrs)


if __name__ == "__main__":
    program = esprima.parseScript("var answer = 42")
    ast = Node.from_dict(program.toDict())

    graph = graphviz.Graph(format="png", graph_attr={"splines": "false"})
    count_dict = defaultdict(int)
    visualise_ast(ast, graph, count_dict, ast.body[0].declarations[0].init, ast.body[0])

    graph.render("ast")
