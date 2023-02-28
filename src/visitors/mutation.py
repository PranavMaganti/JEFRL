import esprima
from esprima.objects import Object

from mutation.mutation_type import MutationType


class MutationVisitor(esprima.NodeVisitor):
    """
    A visitor class for mutating an AST
    """
    def transform(self, obj, metadata):
        """Transform an Object."""
        if isinstance(obj, Object):
            method = 'transform_' + obj.__class__.__name__
            transformer = getattr(self, method, self.generic_transform)
            new_obj = transformer(obj, metadata)
            # if new_obj is not None and obj is not new_obj:
            obj = new_obj
        return obj

    def generic_transform(self, node, metadata):
        if node is metadata["target"]:
            match metadata["type"]:
                case MutationType.REPLACE:
                    return metadata["new"]
                case MutationType.REMOVE:
                    return None
                case MutationType.ADD:
                    if hasattr(node, "body") and isinstance(node.body, list):
                        node.body.append(metadata["new"])
                    elif hasattr(node, "declarations") and isinstance(node.declarations, list):
                        node.declarations.append(metadata["new"])
                    elif hasattr(node, "elements") and isinstance(node.elements, list):
                        node.elements.append(metadata["new"])
        
        return node

    def transform_Children(self, children, metadata):
        new_children = []
        for child in children:
            new_child = self.transform(child, metadata)
            if new_child is not None:
                new_children.append(new_child)

        return new_children

    def transform_Script(self, node, metadata):
        if node is metadata["target"]:
            self.generic_transform(node, metadata)

        node.body = self.transform_Children(node.body, metadata)

        return node

    def transform_ForOfStatement(self, node, metadata):
        if node is metadata["target"]:
            return self.generic_transform(node, metadata)

        node.body = self.transform(node.body, metadata)
        node.left = self.transform(node.left, metadata)
        node.right = self.transform(node.right, metadata)

        return node

    def transform_VariableDeclaration(self, node, metadata):
        if node is metadata["target"]:
            return self.generic_transform(node, metadata)

        node.declarations = self.transform_Children(node.declarations, metadata)
        return node

    def transform_ArrayExpression(self, node, metadata):
        if node is metadata["target"]:
            return self.generic_transform(node, metadata)

        node.elements = self.transform_Children(node.elements, metadata)
        return node

    def transform_BlockStatement(self, node, metadata):
        if node is metadata["target"]:
            return self.generic_transform(node, metadata)

        node.body = self.transform_Children(node.body, metadata)
        return node

    def transform_ExpressionStatement(self, node, metadata):
        if node is metadata["target"]:
            return self.generic_transform(node, metadata)

        node.expression = self.transform(node.expression, metadata)
        return node

    def transform_CallExpression(self, node, metadata):
        if node is metadata["target"]:
            return self.generic_transform(node, metadata)

        node.callee = self.transform(node.callee, metadata)
        node.arguments = self.transform_Children(node.arguments, metadata)
        return node
    
    def transform_Literal(self, node, metadata):
        if node is metadata["target"]:
            return self.generic_transform(node, metadata)

        return node

    def transform_ForStatement(self, node, metadata):
        if node is metadata["target"]:
            return self.generic_transform(node, metadata)

        node.init = self.transform(node.init, metadata)
        node.test = self.transform(node.test, metadata)
        node.update = self.transform(node.update, metadata)
        node.body = self.transform(node.body, metadata)
        return node

    def transform_IfStatement(self, node, metadata):
        if node is metadata["target"]:
            return self.generic_transform(node, metadata)

        node.test = self.transform(node.test, metadata)
        node.consequent = self.transform(node.consequent, metadata)
        node.alternate = self.transform(node.alternate, metadata)
        return node
    
    def transform_WhileStatement(self, node, metadata):
        if node is metadata["target"]:
            return self.generic_transform(node, metadata)

        node.test = self.transform(node.test, metadata)
        node.body = self.transform(node.body, metadata)
        return node

    def transform_DoWhileStatement(self, node, metadata):
        if node is metadata["target"]:
            return self.generic_transform(node, metadata)

        node.test = self.transform(node.test, metadata)
        node.body = self.transform(node.body, metadata)
        return node
    
    def transform_WithStatement(self, node, metadata):
        if node is metadata["target"]:
            return self.generic_transform(node, metadata)

        node.object = self.transform(node.object, metadata)
        node.body = self.transform(node.body, metadata)
        return node
    
    def transform_SwitchStatement(self, node, metadata):
        if node is metadata["target"]:
            return self.generic_transform(node, metadata)

        node.discriminant = self.transform(node.discriminant, metadata)
        node.cases = self.transform_Children(node.cases, metadata)
        return node
    
    def transform_ReturnStatement(self, node, metadata):
        if node is metadata["target"]:
            return self.generic_transform(node, metadata)

        node.argument = self.transform(node.argument, metadata)
        return node

    

