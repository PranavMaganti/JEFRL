import pytest
from ast_parser.main import DecompVisitor

@pytest.fixture
def visitor():
    return DecompVisitor()
