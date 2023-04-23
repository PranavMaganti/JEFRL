#  Test that the function correctly replaces a node with a subtree
from js_ast.nodes import (
    BinaryExpression,
    BlockStatement,
    Expression,
    ExpressionStatement,
    Identifier,
    Literal,
    Node,
    SpreadElement,
    Statement,
    VariableDeclarator,
)
from utils.mutation import replace


# When target is the root node, return the original node
def test_replace_root_node():
    node = Statement()
    assert replace({"Statement": []}, node) == node


# When target is a field of parent, replace the field with the subtree
def test_replace_field():
    node = BinaryExpression(
        left=Identifier(name="a"),
        right=Identifier(name="b"),
        operator="+",
    )
    subtrees: dict[str, list[Node]] = {"Literal": [Literal(value=1, raw="1")]}
    new_node = replace(subtrees, node)

    assert new_node.left == Identifier(name="a", parent=new_node)
    assert new_node.right == Literal(value=1, raw="1", parent=new_node)
    assert new_node.operator == "+"


# # code = """
# # function f(x) {
# #     return x + 1;
# # }

# # f(10);
# # """
# code = {
#     "type": "Program",
#     "sourceType": "script",
#     "body": [
#         {
#             "type": "FunctionDeclaration",
#             "id": {"type": "Identifier", "name": "f"},
#             "params": [{"type": "Identifier", "name": "x"}],
#             "generator": False,
#             "async": False,
#             "expression": False,
#             "body": {
#                 "type": "BlockStatement",
#                 "body": [
#                     {
#                         "type": "ReturnStatement",
#                         "argument": {
#                             "type": "BinaryExpression",
#                             "operator": "+",
#                             "left": {"type": "Identifier", "name": "x"},
#                             "right": {"type": "Literal", "value": 1, "raw": "1"},
#                         },
#                     }
#                 ],
#             },
#         },
#         {
#             "type": "FunctionDeclaration",
#             "id": {"type": "Identifier", "name": "g"},
#             "params": [{"type": "Identifier", "name": "x"}],
#             "generator": False,
#             "async": False,
#             "expression": False,
#             "body": {
#                 "type": "BlockStatement",
#                 "body": [
#                     {
#                         "type": "ReturnStatement",
#                         "argument": {
#                             "type": "BinaryExpression",
#                             "operator": "+",
#                             "left": {"type": "Identifier", "name": "x"},
#                             "right": {"type": "Literal", "value": 1, "raw": "1"},
#                         },
#                     }
#                 ],
#             },
#         },
#         {
#             "type": "ExpressionStatement",
#             "expression": {
#                 "type": "CallExpression",
#                 "callee": {"type": "Identifier", "name": "f"},
#                 "arguments": [{"type": "Literal", "value": 10, "raw": "10"}],
#             },
#         },
#     ],
# }

# tree = Node.from_dict(code)
# target = tree.body[2].expression
# live_variable_analysis(tree, Scope())

# subtrees = {
#     "CallExpression": [CallExpression(False, Identifier("g"), [Literal(1, "1")])]
# }
# # engine = V8Engine()
# # print(engine.execute(tree))
# print(tree)

# replace(subtrees, target)

# print(tree)
