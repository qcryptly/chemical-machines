# %% Cell 1 - Basic Import
# Cell-based Python example
# Each cell can be run independently

from utils import greet

print(greet("Cell 1"))
print("This is the first cell")

# %% Cell 2 - Math Functions
from utils import factorial, fibonacci

# Calculate factorials
for n in [5, 7, 10]:
    print(f"{n}! = {factorial(n)}")

# Show Fibonacci sequence
print(f"Fibonacci(15): {fibonacci(15)}")

# %% Cell 3 - Using Classes
from utils import Counter

# Create and use a counter
c = Counter(100)
print(f"Start: {c.value}")

for i in range(5):
    c.increment(i * 2)
    print(f"After +{i*2}: {c.value}")
