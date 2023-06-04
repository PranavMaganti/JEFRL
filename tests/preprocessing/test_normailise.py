import esprima
from js_ast import escodegen
from js_ast.nodes import Node
from preprocessing.normalise import collect_id
from preprocessing.normalise import is_declared_id
from preprocessing.normalise import normalize_id


class TestIsDeclaredId:
    def test_variable_declaration(self):
        ast = esprima.parseScript("const x = 42;")
        ast = Node.from_dict(ast.toDict())

        node = ast.body[0].declarations[0].id
        assert is_declared_id(node, "id") == True
        assert is_declared_id(node, "left") == False

    # Test with function declaration
    def test_function_declaration(self):
        ast = esprima.parseScript("function foo() {}")
        ast = Node.from_dict(ast.toDict())

        node = ast.body[0].id
        assert is_declared_id(node, "id") == True
        assert is_declared_id(node, "left") == False

    def test_class_declaration(self):
        ast = esprima.parseScript("class Foo {}")
        ast = Node.from_dict(ast.toDict())

        node = ast.body[0].id
        assert is_declared_id(node, "id") == True
        assert is_declared_id(node, "left") == False

    def test_assignment_expression(self):
        ast = esprima.parseScript("x = 42;")
        ast = Node.from_dict(ast.toDict())

        node = ast.body[0].expression.left
        assert is_declared_id(node, "id") == False
        assert is_declared_id(node, "left") == True

    def identifier_not_declared(self):
        ast = esprima.parseScript("foo();")
        ast = Node.from_dict(ast.toDict())

        node = ast.body[0].expression
        assert is_declared_id(node, "id") == False
        assert is_declared_id(node, "left") == False


class TestCollectId:
    def test_variable_declaration(self):
        ast = esprima.parseScript("const x = 42;")
        ast = Node.from_dict(ast.toDict())

        id_dict = {}
        id_cnt = {"v": 0, "f": 0, "c": 0}
        collect_id(ast, id_dict, id_cnt)
        assert id_dict == {"x": "v0"}
        assert id_cnt == {"v": 1, "f": 0, "c": 0}

    def test_function_declaration(self):
        ast = esprima.parseScript("function foo() {}")
        ast = Node.from_dict(ast.toDict())

        id_dict = {}
        id_cnt = {"v": 0, "f": 0, "c": 0}
        collect_id(ast, id_dict, id_cnt)
        assert id_dict == {"foo": "f0"}
        assert id_cnt == {"f": 1, "v": 0, "c": 0}

    def test_class_declaration(self):
        ast = esprima.parseScript("class Foo {}")
        ast = Node.from_dict(ast.toDict())

        id_dict = {}
        id_cnt = {"v": 0, "f": 0, "c": 0}
        collect_id(ast, id_dict, id_cnt)
        assert id_dict == {"Foo": "c0"}
        assert id_cnt == {"c": 1, "v": 0, "f": 0}

    def test_assignment_expression(self):
        ast = esprima.parseScript("x = 42;")
        ast = Node.from_dict(ast.toDict())

        id_dict = {}
        id_cnt = {"v": 0, "f": 0, "c": 0}
        collect_id(ast, id_dict, id_cnt)
        assert id_dict == {"x": "v0"}
        assert id_cnt == {"v": 1, "f": 0, "c": 0}

    def test_identifier_not_declaration(self):
        ast = esprima.parseScript("foo();")
        ast = Node.from_dict(ast.toDict())

        id_dict = {}
        id_cnt = {"v": 0, "f": 0, "c": 0}
        collect_id(ast, id_dict, id_cnt)
        assert id_dict == {}
        assert id_cnt == {"v": 0, "f": 0, "c": 0}


class TestNormalizeId:
    def test_variable_declaration(self):
        code = """
        const bar = 42;
        let baz = bar + 1;
        """
        ast = esprima.parseScript(code)
        ast = Node.from_dict(ast.toDict())

        id_dict = {"bar": "v0", "baz": "v1"}
        normalize_id(ast, id_dict)
        assert escodegen.generate(ast) == "const v0 = 42;\nlet v1 = v0 + 1;"

    def test_function_declaration(self):
        code = """
        function foo() { return 2 + 3; }
        function bar() { return 1 + 2; }
        """

        ast = esprima.parseScript(code)
        ast = Node.from_dict(ast.toDict())

        id_dict = {"foo": "f0", "bar": "f1"}
        normalize_id(ast, id_dict)
        assert (
            escodegen.generate(ast)
            == "function f0() {\n    return 2 + 3;\n}\nfunction f1() {\n    return 1 + 2;\n}"
        )

    def test_object_not_changed_property(self):
        code = """
        const obj = {
            foo: 42,
            bar: 24
        };
        """

        ast = esprima.parseScript(code)
        ast = Node.from_dict(ast.toDict())

        id_dict = {"foo": "v0", "bar": "v1", "obj": "v0"}
        normalize_id(ast, id_dict)
        assert escodegen.generate(ast) == "const v0 = {\n    foo: 42,\n    bar: 24\n};"

    def test_array_assignment(self):
        code = """
        const arr = [1, 2, 3];
        arr[1] = 42;
        """
        ast = esprima.parseScript(code)
        ast = Node.from_dict(ast.toDict())

        id_dict = {"arr": "v0"}
        normalize_id(ast, id_dict)
        assert (
            escodegen.generate(ast)
            == "const v0 = [\n    1,\n    2,\n    3\n];\nv0[1] = 42;"
        )
