from js_ast.analysis import scope_analysis
from js_ast.nodes import (
    BlockStatement,
    FunctionDeclaration,
    Identifier,
    Program,
    VariableDeclaration,
    VariableDeclarator,
)


def test_scope_analysis_vars():
    # Create a new program node
    program_node = Program(
        body=[
            VariableDeclaration(
                kind="let",
                declarations=[VariableDeclarator(id=Identifier(name="x"), init=None)],
            )
        ],
        sourceType="script",
    )

    # Analyze the program node for live variables
    scope_analysis(program_node)

    # Ensure that the variable x is in scope
    for node in program_node.traverse():
        assert hasattr(node, "scope")

    assert hasattr(program_node, "end_scope")
    assert "x" in program_node.end_scope.available_variables()


def test_scope_analysis_funcs():
    program_node = Program(
        body=[
            FunctionDeclaration(
                id=Identifier(name="f"),
                params=[Identifier(name="x")],
                body=BlockStatement(body=[]),
                generator=False,
            ),
            VariableDeclaration(
                kind="let",
                declarations=[VariableDeclarator(id=Identifier(name="x"), init=None)],
            ),
        ],
        sourceType="script",
    )  # type: ignore

    scope_analysis(program_node)

    for node in program_node.traverse():
        assert hasattr(node, "scope")

    assert hasattr(program_node, "end_scope")
    assert hasattr(program_node.body[0].body, "end_scope")
    assert "f" in program_node.body[1].scope.available_functions()
