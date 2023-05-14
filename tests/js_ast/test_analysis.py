from js_ast.analysis import scope_analysis
from js_ast.nodes import (
    BinaryExpression,
    BlockStatement,
    ExpressionStatement,
    FunctionDeclaration,
    Identifier,
    Literal,
    Program,
    VariableDeclaration,
    VariableDeclarator,
)


def test_scope_analysis_basic_vars():
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


def test_scope_analysis_basic_funcs():
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


def test_scope_analysis_block_vars():
    # Test that let variables are not available outside of the block
    program_node = Program(
        body=[
            BlockStatement(
                body=[
                    VariableDeclaration(
                        kind="let",
                        declarations=[
                            VariableDeclarator(id=Identifier(name="x"), init=None)
                        ],
                    ),
                ]
            ),
            VariableDeclaration(
                kind="var",
                declarations=[VariableDeclarator(id=Identifier(name="y"), init=None)],
            ),
        ],
        sourceType="script",
    )  # type: ignore
    scope_analysis(program_node)
    assert program_node.body[1].scope
    assert program_node.body[1].scope.available_variables() == set()

    # Test that var variables are available outside of the block
    program_node = Program(
        body=[
            BlockStatement(
                body=[
                    VariableDeclaration(
                        kind="var",
                        declarations=[
                            VariableDeclarator(id=Identifier(name="x"), init=None)
                        ],
                    ),
                ]
            ),
            VariableDeclaration(
                kind="var",
                declarations=[VariableDeclarator(id=Identifier(name="y"), init=None)],
            ),
        ],
        sourceType="script",
    )  # type: ignore
    scope_analysis(program_node)
    assert program_node.body[1].scope.available_variables() == {"x"}


def test_scope_analysis_nested_blocks():
    # Test that variables declared in nested blocks are not available outside of them
    program_node = Program(
        body=[
            BlockStatement(
                body=[
                    VariableDeclaration(
                        kind="let",
                        declarations=[
                            VariableDeclarator(id=Identifier(name="x"), init=None)
                        ],
                    ),
                    BlockStatement(
                        body=[
                            VariableDeclaration(
                                kind="let",
                                declarations=[
                                    VariableDeclarator(
                                        id=Identifier(name="y"), init=None
                                    )
                                ],
                            )
                        ]
                    ),
                ]
            ),
            VariableDeclaration(
                kind="var",
                declarations=[VariableDeclarator(id=Identifier(name="z"), init=None)],
            ),
        ],
        sourceType="script",
    )  # type: ignore
    scope_analysis(program_node)
    assert program_node.body[1].scope
    assert program_node.body[1].scope.available_variables() == set()
    assert program_node.body[0].body[1].scope.available_variables() == {"x"}


def test_scope_analysis_function_params():
    # Test that function parameters are in scope within the function body
    program_node = Program(
        body=[
            FunctionDeclaration(
                id=Identifier(name="f"),
                params=[Identifier(name="x")],
                body=BlockStatement(
                    body=[
                        ExpressionStatement(
                            expression=BinaryExpression(
                                operator="+",
                                left=Identifier(name="x"),
                                right=Literal(value=1, raw="1"),
                            ),
                            directive="",
                        )
                    ]
                ),
                generator=False,
            ),
        ],
        sourceType="script",
    )  # type: ignore
    scope_analysis(program_node)
    assert program_node.body[0].scope.available_variables() == {"x"}
