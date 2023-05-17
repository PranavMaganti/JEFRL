import random

from js_ast.analysis import fix_node_references, scope_analysis
from js_ast.nodes import (
    BinaryExpression,
    BlockStatement,
    CallExpression,
    ExpressionStatement,
    FunctionDeclaration,
    Identifier,
    Literal,
    Program,
    VariableDeclaration,
    VariableDeclarator,
)
from js_ast.scope import Scope


class TestScopeAnalysis:
    def test_basic_vars(self):
        # Create a new program node
        program_node = Program(
            body=[
                VariableDeclaration(
                    kind="let",
                    declarations=[
                        VariableDeclarator(
                            id=Identifier(name="x"), init=Literal(value=1, raw="1")
                        )
                    ],
                )
            ],
            sourceType="script",
        )

        # Analyze the program node for live variables
        scope_analysis(program_node)

        # Ensure that the variable x is in scope
        for node in program_node.traverse():
            if node.scope is None:
                print(node)
            assert node.scope is not None

        assert program_node.end_scope is not None
        assert "x" in program_node.end_scope.available_variables()

    def test_basic_funcs(self):
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
                    declarations=[
                        VariableDeclarator(id=Identifier(name="x"), init=None)
                    ],
                ),
            ],
            sourceType="script",
        )  # type: ignore

        scope_analysis(program_node)

        for node in program_node.traverse():
            assert node.scope is not None

        assert hasattr(program_node, "end_scope")
        assert hasattr(program_node.body[0].body, "end_scope")
        assert "f" in program_node.body[1].scope.available_functions()

    def test_block_vars(self):
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
                    declarations=[
                        VariableDeclarator(id=Identifier(name="y"), init=None)
                    ],
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
                    declarations=[
                        VariableDeclarator(id=Identifier(name="y"), init=None)
                    ],
                ),
            ],
            sourceType="script",
        )  # type: ignore
        scope_analysis(program_node)
        assert program_node.body[1].scope.available_variables() == {"x"}

    def test_nested_blocks(self):
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
                    declarations=[
                        VariableDeclarator(id=Identifier(name="z"), init=None)
                    ],
                ),
            ],
            sourceType="script",
        )  # type: ignore
        scope_analysis(program_node)
        assert program_node.body[1].scope
        assert program_node.body[1].scope.available_variables() == set()
        assert program_node.body[0].body[1].scope.available_variables() == {"x"}

    def test_function_params(self):
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


class TestFixNodeReferences:
    def test_identifier(self):
        node = Identifier(name="z")
        scope = Scope(variables={"x", "y"})
        node.scope = scope
        fix_node_references(node)
        assert node.name in scope.available_variables()
        for child in node.traverse():
            assert node.scope is not None

    def test_call_expression(self):
        node = CallExpression(
            callee=Identifier(name="foo"),
            arguments=[Literal(value=1, raw="1"), Literal(value=2, raw="2")],
            optional=False,
        )
        scope = Scope(functions={"bar": 1})
        node.scope = scope
        fix_node_references(node)
        assert node.callee.name in scope.available_functions()
        assert len(node.arguments) == 1
        for child in node.traverse():
            assert node.scope is not None

    def test_nested_nodes(self):
        random.seed(0)
        node = BinaryExpression(
            left=Identifier(name="x"),
            operator="+",
            right=CallExpression(
                callee=Identifier(name="bar"),
                arguments=[Identifier(name="y")],
                optional=False,
            ),
        )
        scope = Scope(variables={"x", "z"}, functions={"foo": 1})
        node.scope = scope
        node.right.scope = scope
        node.left.scope = scope
        node.right.arguments[0].scope = scope
        node.right.callee.scope = scope

        fix_node_references(node)
        assert node.left.name in scope.available_variables()
        assert node.right.callee.name in scope.available_functions()
        assert (
            node.right.arguments[0].type == "Identifier"
            or node.right.arguments[0].type == "Literal"
        )

        for child in node.traverse():
            assert node.scope is not None
