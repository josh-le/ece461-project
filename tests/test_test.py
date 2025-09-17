# this is a sample file just demonstrating how our tests work
# any python file whose name matches either test_*.py or *_test.py will be run with pytest
# the test functions must match test*

def add_one(x: int) -> int:
    return x + 1

def test_add_one() -> None:
    assert 1 + 1 == add_one(1)
