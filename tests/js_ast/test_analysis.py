import random

from js_ast.analysis import fix_node_references
from js_ast.analysis import scope_analysis
from js_ast.nodes import ArrowFunctionExpression
from js_ast.nodes import BinaryExpression
from js_ast.nodes import BlockStatement
from js_ast.nodes import CallExpression
from js_ast.nodes import ExpressionStatement
from js_ast.nodes import FunctionDeclaration
from js_ast.nodes import FunctionExpression
from js_ast.nodes import Identifier
from js_ast.nodes import Literal
from js_ast.nodes import Program
from js_ast.nodes import VariableDeclaration
from js_ast.nodes import VariableDeclarator
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
            if node.type == "Identifier" or node.type == "Literal":
                continue
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
            if node.type == "Identifier" or node.type == "Literal":
                continue

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

    def test_variable_function_declaration(self):
        # Test that function parameters are in scope within the function body
        program_node = Program(
            body=[
                VariableDeclaration(
                    kind="var",
                    declarations=[
                        VariableDeclarator(
                            id=Identifier(name="v0"),
                            init=FunctionExpression(
                                id=Identifier(name="f"),
                                params=[],
                                body=BlockStatement(
                                    body=[
                                        VariableDeclaration(
                                            kind="var",
                                            declarations=[
                                                VariableDeclarator(
                                                    id=Identifier(name="v1"),
                                                    init=Literal(value=1, raw="1"),
                                                )
                                            ],
                                        ),
                                    ]
                                ),
                                generator=False,
                                expression=False,
                            ),
                        )
                    ],
                ),
                BlockStatement(body=[]),
            ],
            sourceType="script",
        )  # type: ignore
        scope_analysis(program_node)
        assert program_node.body[1].scope.available_variables() == {"v0"}
        assert program_node.body[1].scope.available_functions() == {"v0": 0}

    def test_variable_arrow_function(self):
        # Test that function parameters are in scope within the function body
        program_node = Program(
            body=[
                VariableDeclaration(
                    kind="var",
                    declarations=[
                        VariableDeclarator(
                            id=Identifier(name="v0"),
                            init=ArrowFunctionExpression(
                                params=[],
                                body=BlockStatement(
                                    body=[
                                        VariableDeclaration(
                                            kind="var",
                                            declarations=[
                                                VariableDeclarator(
                                                    id=Identifier(name="v1"),
                                                    init=Literal(value=1, raw="1"),
                                                )
                                            ],
                                        ),
                                    ]
                                ),
                                generator=False,
                                expression=False,
                            ),
                        )
                    ],
                ),
                BlockStatement(body=[]),
            ],
            sourceType="script",
        )  # type: ignore
        scope_analysis(program_node)
        assert program_node.body[1].scope.available_variables() == {"v0"}
        assert program_node.body[1].scope.available_functions() == {"v0": 0}


class TestFixNodeReferences:
    def test_call_expression(self):
        node = CallExpression(
            callee=Identifier(name="foo"),
            arguments=[Literal(value=1, raw="1"), Literal(value=2, raw="2")],
            optional=False,
        )
        scope = Scope(functions={"bar": 1})
        node.scope = scope
        fix_node_references(node, {})
        assert node.callee.name in scope.available_functions()
        assert len(node.arguments) == 1
        for child in node.traverse():
            if child.type == "Identifier" or child.type == "Literal":
                continue
            assert child.scope is not None

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

        fix_node_references(node, {})
        assert node.left.name in scope.available_variables()
        assert node.right.callee.name in scope.available_functions()
        assert (
            node.right.arguments[0].type == "Identifier"
            or node.right.arguments[0].type == "Literal"
        )

        for child in node.traverse():
            if child.type == "Identifier" or child.type == "Literal":
                continue
            assert child.scope is not None
