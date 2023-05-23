import copy

from js_ast.nodes import Node

from utils.js_engine import Coverage
from utils.js_engine import Engine


def basic_minimiser(root: Node, engine: Engine, coverage: Coverage):
    # Make a deep copy of the root node
    root_copy = copy.deepcopy(root)

    # Recursively traverse the tree and remove nodes that do not affect the output
    def remove_nodes(node: Node):
        # Recursively remove nodes from child nodes
        for field in node.fields:
            val = getattr(node, field)
            if isinstance(val, list):
                i = 0
                while i < len(val):
                    child = val[i]
                    del val[i]
                    remove_nodes(child)

                    # Check if removing the child node affects the output
                    new_output = engine.execute_text(root_copy.generate_code())
                    if new_output and new_output.coverage == coverage:
                        continue  # Node can be safely removed
                    else:
                        val.insert(i, child)  # Node cannot be removed
                        i += 1

            elif isinstance(val, Node):
                remove_nodes(val)

    remove_nodes(root_copy)
    return root_copy
