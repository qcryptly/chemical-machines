# Single-file Python example
# This file is executed as a single unit (no cells)
# It imports utils.py from the same directory

from utils import greet, factorial, fibonacci, Counter

# Use the greet function
print(greet("World"))

# Calculate some factorials
for i in range(1, 6):
    print(f"{i}! = {factorial(i)}")

# Generate Fibonacci sequence
fibs = fibonacci(10)
print(f"First 10 Fibonacci numbers: {fibs}")

# Use the Counter class
counter = Counter(10)
print(f"Counter starts at: {counter.value}")
counter.increment(5)
print(f"After +5: {counter.value}")
counter.decrement(3)
print(f"After -3: {counter.value}")
