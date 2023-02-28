from loader.main import load_corpus
from visitors.subtree import SubtreeVisitor
from mutation.main import MutationEngine
import esprima


CORPUS_PATH = "corpus"

corpus = load_corpus(CORPUS_PATH)
subtree_visitor = SubtreeVisitor()

# Extract subtrees from corpus files
for file in corpus:
    subtree_visitor.visit(file)

# for node_type in subtree_visitor.nodes:
#     print(node_type, len(subtree_visitor.nodes[node_type]))

test_code = """
for (let i of [1, 2, 3]) {
    console.log(i);
}
"""

ast = esprima.parseScript(test_code, tolerant=True, jsx=True)
print(ast)

# Replace the first element of the array with a new literal
target_node = ast.body[0].right.elements[0]
mutation_engine = MutationEngine(subtree_visitor.nodes)
print(mutation_engine.replace(ast, target_node))

