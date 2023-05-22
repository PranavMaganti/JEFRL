from js_ast.scope import Scope
from js_ast.scope import ScopeType


def test_base_scope():
    # Test that the base scope is correct
    scope = Scope()

    scope.variables.add("x")
    scope.functions["f"] = 1
    scope.classes.add("C")

    assert scope.available_variables() == {"x"}
    assert scope.available_functions() == {"f": 1}
    assert scope.available_classes() == {"C"}


def test_nested_scope_with_child_variable():
    root_scope = Scope(scope_type=ScopeType.GLOBAL)
    child_scope = Scope(parent=root_scope, scope_type=ScopeType.BLOCK)

    child_scope.variables.add("x")

    assert child_scope.available_variables() == {"x"}
    assert child_scope.available_functions() == {}
    assert child_scope.available_classes() == set()

    assert root_scope.available_variables() == set()
    assert root_scope.available_functions() == {}
    assert root_scope.available_classes() == set()


def test_nested_scope_with_root_variable():
    root_scope = Scope(scope_type=ScopeType.GLOBAL)
    root_scope.variables.add("x")

    child_scope = Scope(parent=root_scope, scope_type=ScopeType.BLOCK)

    assert child_scope.available_variables() == {"x"}
    assert child_scope.available_functions() == {}
    assert child_scope.available_classes() == set()

    assert root_scope.available_variables() == {"x"}
    assert root_scope.available_functions() == {}
    assert root_scope.available_classes() == set()
