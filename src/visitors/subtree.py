from asyncio.log import logger
import esprima 
from collections import defaultdict


class SubtreeVisitor(esprima.NodeVisitor):
    """
    A visitor class for extracting relevant subtrees from an AST
    """
    def __init__(self):
        super().__init__()
        # Cumulative list of all relevant subtrees in visited AST's
        self.nodes = defaultdict(list)

    def visit(self, node):
        if hasattr(node, 'type') and node.type != "Program":
            self.nodes[node.type].append(node)
            logger.debug(f"Found node of type {node.type}")

        return super().visit(node)
    
    def visit_Script(self, node):
        for child in node.body:
            self.visit(child)

    def visit_ForOfStatement(self, node):
        self.visit(node.body)

    def visit_BlockStatement(self, node):
        for child in node.body:
            self.visit(child)
    
    def visit_ExpressionStatement(self, node):
        self.visit(node.expression)
    
    def visit_VariableDeclaration(self, node):
        for child in node.declarations:
            self.visit(child)
    
    def visit_VariableDeclarator(self, node):
        self.visit(node.id)
        self.visit(node.init)
    
    def visit_ArrayExpression(self, node):
        for child in node.elements:
            self.visit(child)
    
    def visit_CallExpression(self, node):
        self.visit(node.callee)
        for child in node.arguments:
            self.visit(child)
    
    def visit_FunctionDeclaration(self, node):
        self.visit(node.id)
        for child in node.params:
            self.visit(child)
        self.visit(node.body)

    def visit_ObjectExpression(self, node):
        for child in node.properties:
            self.visit(child)

    def visit_Property(self, node):
        self.visit(node.key)
        self.visit(node.value)
    
    def visit_ThrowStatement(self, node):
        self.visit(node.argument)

    def visit_NewExpression(self, node):
        self.visit(node.callee)
        for child in node.arguments:
            self.visit(child)

    def visit_AssignmentExpression(self, node):
        self.visit(node.left)
        self.visit(node.right)
    
    def visit_IfStatement(self, node):
        self.visit(node.test)
        self.visit(node.consequent)
        self.visit(node.alternate)
    
    def visit_BinaryExpression(self, node):
        self.visit(node.left)
        self.visit(node.right)
    
    def visit_ReturnStatement(self, node):
        self.visit(node.argument)
    
    def visit_ArrowFunctionExpression(self, node):
        for child in node.params:
            self.visit(child)
        self.visit(node.body)
    
    def visit_Identifier(self, node):
        pass

    def visit_Literal(self, node):
        pass
    
    def visit_WhileStatement(self, node):
        self.visit(node.test)
        self.visit(node.body)
    
    def visit_ForStatement(self, node):
        self.visit(node.init)
        self.visit(node.test)
        self.visit(node.update)
        self.visit(node.body)
    
    def visit_DoWhileStatement(self, node):
        self.visit(node.body)
        self.visit(node.test)

    def visit_ClassExpression(self, node):
        self.visit(node.body)
    
    def visit_ClassBody(self, node):
        for child in node.body:
            self.visit(child)
    
    def visit_MethodDefinition(self, node):
        self.visit(node.key)
        self.visit(node.value)
    
    def visit_TryStatement(self, node):
        self.visit(node.block)
        self.visit(node.handler)
        self.visit(node.finalizer)

    def visit_SwitchStatement(self, node):
        self.visit(node.discriminant)
        for child in node.cases:
            self.visit(child)
        

if __name__ == "__main__":
    visitor = SubtreeVisitor()
    code = """
    const Rectangle = class {
        constructor(height, width) {
            this.height = height;
            this.width = width;
        }
        area() {
            return this.height * this.width;
        }
    };
    """
    ast = esprima.parseScript(code, tolerant=True, jsx=True)
    visitor.visit(ast)
    print(ast)