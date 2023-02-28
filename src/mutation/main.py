
from platform import node
import esprima
from esprima.nodes import Node
from mutation.mutation_type import MutationType

from visitors.mutation import MutationVisitor
import numpy as np


class MutationEngine:
    def __init__(self, subtrees: dict[str, list[Node]]):
        self.subtrees = subtrees
        self.mutation_visitor = MutationVisitor()

    def replace(self, root: Node, target: Node) -> Node:
        if not (hasattr(target, "type") and target.type in self.subtrees):
            return root
        
        new_idx = np.random.randint(len(self.subtrees[target.type]))
        new = self.subtrees[target.type][new_idx]

        return self.mutation_visitor.transform(
            root,
            {
                "type": MutationType.REPLACE,
                "target": target,
                "new": new,
            },
        )  # type: ignore
    
    def remove(self, root: Node, target: Node) -> Node:
        return self.mutation_visitor.transform(
            root,
            {
                "type": MutationType.REMOVE,
                "target": target,
            },
        )   # type: ignore
    
    def add(self, root: Node, target: Node) -> Node:
        new_node_type = np.random.choice(list(self.subtrees.keys()))
        new_idx = np.random.randint(len(self.subtrees[new_node_type]))
        new = self.subtrees[new_node_type][new_idx]

        return self.mutation_visitor.transform(
            root,
            {
                "type": MutationType.ADD,
                "target": target,
                "new": new,
            },
        ) # type: ignore



