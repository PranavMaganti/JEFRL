import random
from js_ast.nodes import Identifier, Literal, Node
from js_ast.scope import BlockScope, Scope


def live_variable_analysis(node: Node, scope: Scope):
    node.set_scope(
        Scope(
            scope.available_variables(),
            scope.available_functions(),
            scope.available_classes(),
        )
    )

    if (
        node.type == "Program"
        or node.type == "BlockStatement"
        or node.type == "ClassBody"
    ):
        if node.type == "BlockStatement" and node.parent.type != "FunctionDeclaration":
            new_scope = BlockScope(parent=scope)
        else:
            new_scope = Scope(parent=scope)

        functions = filter(lambda x: x.type == "FunctionDeclaration", node.body)
        classes = filter(lambda x: x.type == "ClassDeclaration", node.body)
        methods = filter(
            lambda x: x.type == "MethodDefinition" and x.kind != "constructor",
            node.body,
        )

        for function in functions:
            new_scope.functions[function.id.name] = function.params
        for class_ in classes:
            new_scope.classes.add(class_.id.name)
        for method in methods:
            new_scope.functions[method.key.name] = method.value.params

        for item in node.body:
            live_variable_analysis(item, new_scope)

        node.end_scope = Scope(
            new_scope.available_variables(),
            new_scope.available_functions(),
            new_scope.available_classes(),
        )

    elif node.type == "VariableDeclarator":
        if node.init:
            live_variable_analysis(node.init, scope)
        if node.kind == "var":
            current_scope = scope
            while current_scope.parent and isinstance(current_scope, BlockScope):
                current_scope = current_scope.parent

            current_scope.variables.add(node.id.name)
        else:
            scope.variables.add(node.id.name)
    else:
        for child in node.children():
            live_variable_analysis(child, scope)


def fix_node(node: Node):
    scope: Scope = node.scope
    if node.type == "Identifier":
        if node.name not in scope.available_variables():
            node.name = random.choice(list(scope.available_variables()))
    elif node.type == "CallExpression":
        if node.callee.type == "Identifier":
            if node.callee.name not in scope.available_functions():
                function, params = random.choice(
                    list(scope.available_functions().items())
                )
                node.callee.name = function
                node.arguments = [random_value(scope) for _ in range(len(params))]
    else:
        for child in node.children():
            fix_node(child)


# Gets random literal or identifier
def random_value(scope: Scope):
    if scope.available_variables() and random.random() < 0.5:
        return Identifier(name=random.choice(list(scope.available_variables())))

    else:
        # TODO: add more types and interesting values
        return Literal(value=random.randint(0, 100), raw=str(random.randint(0, 100)))
