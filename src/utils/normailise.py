from nodes.main import Node


def normalise(root: Node, fun_num: int = 0, var_num: int = 0):
    fun_mapping = {}
    var_mapping = {}

    for node in root.traverse():
        if hasattr(node, "type"):
            if node.type == "FunctionDeclaration":
                fun_mapping[node.id.name] = f"f{fun_num}"
                fun_num += 1
            elif node.type == "VariableDeclarator":
                var_mapping[node.id.name] = f"v{var_num}"
                var_num += 1
