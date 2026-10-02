from js_ast.nodes import BinaryExpression
from js_ast.nodes import BlockStatement
from js_ast.nodes import CallExpression
from js_ast.nodes import ExpressionStatement
from js_ast.nodes import Identifier
from js_ast.nodes import Literal
import pytest

from utils.loader import sanitise_ast


class TestSanitiseAst:
    def test_remove_assert_calls(self):
        node = ExpressionStatement(
            expression=CallExpression(
                callee=Identifier(name="assert"),
                arguments=[Literal(value=True, raw="true")],
                optional=False,
            ),
            directive="",
        )
        root = BlockStatement(body=[node])

        sanitise_ast(root)

        assert root.body == []

    def test_keep_other_calls(self):
        node = ExpressionStatement(
            expression=CallExpression(
                callee=Identifier(name="foo"),
                arguments=[Literal(value=2, raw="2")],
                optional=False,
            ),
            directive="",
        )
        root = BlockStatement(body=[node])
        sanitise_ast(root)

        assert node in root.body

    def test_keep_other_nodes(self):
        node = ExpressionStatement(
            expression=BinaryExpression(
                left=Identifier(name="y"),
                operator="*",
                right=Literal(value=2, raw="2"),
            ),
            directive="",
        )

        root = BlockStatement(body=[node])
        sanitise_ast(root)

        assert node in root.body


if __name__ == "__main__":
    pytest.main()


class FakeEngine:
    """Runs a program cleanly only when accepts(code, lib_path) is true."""

    def __init__(self, accepts):
        self.accepts = accepts
        self.calls = []

    def execute_text(self, code, lib_path=None):
        from utils.js_engine import Coverage, ExecutionData, JSError

        self.calls.append(lib_path)
        error = JSError.NoError if self.accepts(code, lib_path) else JSError.Other
        return ExecutionData(Coverage(), error, "")


class TestLoadSeed:
    def _seed(self, tmp_path, code):
        path = tmp_path / "seed.js"
        path.write_text(code)
        return path

    def test_picks_first_working_harness(self, tmp_path):
        from pathlib import Path

        from utils.loader import load_seed

        lib = Path("corpus/libs/spidermonkey.js")
        engine = FakeEngine(lambda code, lib_path: lib_path == lib)
        result = load_seed(engine, self._seed(tmp_path, "var x = 1;"), None)  # type: ignore

        assert result.state is not None
        assert result.state.lib_path == lib
        # No harness and the cheaper harnesses were tried first
        assert engine.calls[0] is None
        assert engine.calls[-1] == lib

    def test_no_harness_needed(self, tmp_path):
        from utils.loader import load_seed

        engine = FakeEngine(lambda code, lib_path: True)
        result = load_seed(engine, self._seed(tmp_path, "var x = 1;"), None)  # type: ignore

        assert result.state is not None and result.state.lib_path is None
        assert engine.calls == [None]

    def test_falls_back_to_original_identifiers(self, tmp_path):
        from utils.loader import load_seed

        # Normalising renames `name` to `v0`; pretend that breaks the program
        engine = FakeEngine(lambda code, lib_path: "v0" not in code)
        result = load_seed(engine, self._seed(tmp_path, "var name = 1;"), None)  # type: ignore

        assert result.state is not None
        assert "var name = 1;" in result.state.generate_program_code()

    def test_gives_up_when_nothing_works(self, tmp_path):
        from utils.loader import load_seed

        engine = FakeEngine(lambda code, lib_path: False)
        result = load_seed(engine, self._seed(tmp_path, "var x = 1;"), None)  # type: ignore

        assert result.state is None and result.status == "exec"
