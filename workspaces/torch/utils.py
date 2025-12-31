"""
Example Python module
This can be imported by other Python files in the workspace
"""

def greet(name: str) -> str:
    """Return a greeting message"""
    return f"Hello, {name}!"

def factorial(n: int) -> int:
    """Calculate factorial recursively"""
    if n <= 1:
        return 1
    return n * factorial(n - 1)

def fibonacci(n: int) -> list[int]:
    """Generate first n Fibonacci numbers"""
    if n <= 0:
        return []
    if n == 1:
        return [0]

    fibs = [0, 1]
    for _ in range(2, n):
        fibs.append(fibs[-1] + fibs[-2])
    return fibs

class Counter:
    """Simple counter class"""
    def __init__(self, start: int = 0):
        self.value = start

    def increment(self, by: int = 1) -> int:
        self.value += by
        return self.value

    def decrement(self, by: int = 1) -> int:
        self.value -= by
        return self.value
