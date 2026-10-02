from js_ast.analysis import fix_node_references
from js_ast.analysis import scope_analysis
from js_ast.natives import decode_natives
from js_ast.natives import encode_natives
from preprocessing.normalise import normalize_ast
from rl.program_state import ProgramState

from utils.loader import parse_code


def test_encode_natives():
    code = "%PrepareFunctionForOptimization(f);\nx = %GetOptimizationStatus(f);"
    encoded = encode_natives(code)
    assert "%" not in encoded
    assert decode_natives(encoded) == code


def test_natives_in_statement_positions():
    cases = [
        "// Flags: --allow-natives-syntax\n%DebugToggleBlockCoverage(true);",
        "if (i%10 == 0) %OptimizeFunctionOnNextCall(sum1);",
        "while (x) %DeoptimizeFunction(f);",
        "if (a) b(); else %DeoptimizeFunction(f);",
        "function g() { return %GetOptimizationStatus(f); }",
    ]
    for code in cases:
        encoded = encode_natives(code)
        assert "%OptimizeFunctionOnNextCall" not in encoded
        assert encoded.count("__v8_native_") == 1, code
        assert decode_natives(encoded) == code


def test_modulo_is_left_alone():
    for code in ["a %Foo(1);", "x = 5 % Foo(2);", "g() %Bar(3);", "a[0] %Baz(4);"]:
        assert encode_natives(code) == code


def test_natives_survive_parse_normalise_mutate_and_codegen():
    code = """
    function foo(a) { return a + 1; }
    %PrepareFunctionForOptimization(foo);
    foo(1);
    %OptimizeFunctionOnNextCall(foo);
    foo(2);
    """
    ast = parse_code(code, "test.js")
    assert ast is not None

    normalize_ast(ast)
    scope_analysis(ast)
    # The fixer must not treat the natives as unknown functions
    fix_node_references(ast, {})

    out = ProgramState(ast, None).generate_program_code()  # type: ignore
    assert out is not None
    assert "%PrepareFunctionForOptimization(f0)" in out
    assert "%OptimizeFunctionOnNextCall(f0)" in out


def test_strings_with_quotes_round_trip():
    import esprima

    code = """var a = "it's";\nvar b = 'say "hi"';\nvar c = "both ' and \\"";\nvar d = "";\nvar e = "\\\\'";"""
    ast = parse_code(code, "test.js")
    assert ast is not None
    out = ProgramState(ast, None).generate_program_code()  # type: ignore
    assert out is not None

    reparsed = esprima.parseScript(out).toDict()
    values = [d["declarations"][0]["init"]["value"] for d in reparsed["body"]]
    assert values == ["it's", 'say "hi"', "both ' and \"", "", "\\'"]
