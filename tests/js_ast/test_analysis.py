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
        function = program_node.body[0]
        # x is visible inside the function body...
        assert function.body.scope.available_variables() == {"x"}
        assert function.body.body[0].scope.available_variables() == {"x"}
        # ...but must not leak into the enclosing (global) scope
        assert "x" not in function.scope.available_variables()
        assert "x" not in program_node.end_scope.available_variables()

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
        arguments = list(node.arguments)
        fix_node_references(node, {})
        assert node.callee.name in scope.available_functions()
        # The arguments are kept (not regenerated), so nodes inside them stay attached
        assert node.arguments == arguments
        assert all(a is b for a, b in zip(node.arguments, arguments))
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


def _fix_code(code: str) -> str:
    """Parses code, runs one reference-fixing pass over it and generates it again."""
    from utils.loader import parse_code

    random.seed(0)
    ast = parse_code(code, "test.js")
    assert ast is not None
    scope_analysis(ast)
    fix_node_references(ast, {})
    out = ast.generate_code()
    assert out is not None
    return out


class TestFixNodeReferencesOnPrograms:
    """Valid programs must come out of reference fixing unchanged."""

    def test_builtins_are_kept(self):
        code = """
var v0 = 1;
var v1 = new Array(10);
var v2 = new ArrayBuffer(8);
var v3 = new Proxy({}, {});
var v4 = Math.floor(v0) + Object.keys(v1).length + Symbol.iterator.length;
var v5 = [undefined, NaN, Infinity, JSON.stringify(v1), Reflect.ownKeys(v3)];
"""
        out = _fix_code(code)
        for name in ["Array", "ArrayBuffer", "Proxy", "Math", "Object", "Symbol", "JSON", "Reflect", "undefined", "NaN"]:
            assert name in out

    def test_function_params_are_scoped_to_the_function(self):
        out = _fix_code("var v0 = 1;\nfunction f0(x) {\n    return x;\n}\nvar v1 = v0;")
        assert "return x;" in out
        # x must not be used outside f0
        assert "var v1 = v0;" in out

    def test_arrow_and_function_expression_params(self):
        code = "var v0 = [1, 2].map((a, b) => a + b);\nvar v1 = function (c) {\n    return c;\n};"
        out = _fix_code(code)
        assert "a + b" in out
        assert "return c;" in out

    def test_catch_param_and_property_keys(self):
        code = "var v0 = { key: 1 };\ntry {\n    v0.key = 2;\n} catch (e) {\n    v0 = e;\n}"
        out = _fix_code(code)
        assert "key: 1" in out
        assert "v0 = e;" in out

    def test_functions_and_classes_used_as_values(self):
        code = "function f0() {\n}\nclass c0 {\n}\nvar v0 = [f0, c0];\nvar v1 = new c0();"
        out = _fix_code(code)
        assert "[\n    f0,\n    c0\n]" in out or "[f0, c0]" in out
        assert "new c0()" in out

    def test_undeclared_variable_is_still_fixed(self):
        out = _fix_code("var v0 = 1;\nvar v1 = notDeclared + 1;")
        assert "notDeclared" not in out

    def test_default_rest_and_destructured_params(self):
        code = (
            "var v0 = 1;\nfunction f0(a = v0, { b, c = a } = {}, ...d) {\n    return a + b + c + d.length;\n}\n"
            "try {\n    f0();\n} catch ({ message }) {\n    v0 = message;\n}"
        )
        from utils.loader import parse_code

        ast = parse_code(code, "test.js")
        scope_analysis(ast)
        # Every non-leaf node must have a scope, otherwise mutations can't fix references
        for node in ast.traverse():
            if node.type not in ("Identifier", "Literal"):
                assert node.scope is not None, node.type
        assert _fix_code(code) == ast.generate_code()


def test_renaming_a_call_keeps_nodes_attached():
    """A node inside a renamed call's arguments must stay in the tree, otherwise the
    agent's next action on it fails ("Could not find target in parent")."""
    from utils.loader import parse_code

    random.seed(0)
    ast = parse_code("function f0(a) {\n    return a;\n}\nunknownFn(1 + 2, [3]);", "test.js")
    scope_analysis(ast)
    inner = next(n for n in ast.traverse() if n.type == "Literal" and n.value == 3)
    fix_node_references(ast, {})

    assert "f0(" in ast.generate_code()
    node = inner
    while node.parent is not None:
        assert any(child is node for child in node.parent.children())
        node = node.parent
    assert node is ast


class TestUndeclaredReferenceFallbacks:
    """An undeclared reference must always be renamed, even when nothing of the
    same kind is in scope, otherwise the program throws a ReferenceError."""

    def test_value_falls_back_to_a_function(self):
        out = _fix_code("function f0() {\n}\nnotDeclared + 1;")
        assert "f0 + 1;" in out

    def test_value_falls_back_to_a_builtin(self):
        from js_ast.analysis import FALLBACK_VALUES

        out = _fix_code("notDeclared + 1;")
        assert "notDeclared" not in out
        assert any(f"{value} + 1;" in out for value in FALLBACK_VALUES)

    def test_call_falls_back_to_a_builtin_function(self):
        from js_ast.analysis import FALLBACK_FUNCTIONS

        # Only a variable in scope: calling it would be a TypeError
        out = _fix_code("var v0 = 1;\nnotDeclared(v0);")
        assert any(f"{name}(v0);" in out for name in FALLBACK_FUNCTIONS)

    def test_new_prefers_classes(self):
        out = _fix_code("var v0 = 1;\nclass c0 {\n}\nnew notDeclared();")
        assert "new c0();" in out

    def test_names_missing_from_d8_are_renamed(self):
        out = _fix_code("function f0() {\n}\nrequire('fs');\nquit();")
        assert "require" not in out and "quit" not in out
