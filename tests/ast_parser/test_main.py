from ast_parser.main import DecompVisitor
from esprima.nodes import ExpressionStatement, Literal
import esprima

def test_expression_statement(visitor):
    ast = esprima.parse("1;")
    visitor.visit(ast)

    assert "ExpressionStatement" in visitor.node_dict
    assert len(visitor.node_dict["ExpressionStatement"]) == 1
    assert visitor.node_dict["ExpressionStatement"][0].toDict() == ExpressionStatement(Literal(1, "1")).toDict()