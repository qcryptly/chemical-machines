// %% Cell 1 - Include and Setup
// Cell-based C++ example
// Each cell is compiled and executed independently

#include <iostream>
#include "file.hpp"

int main() {
    std::cout << example::greet("Cell 1") << std::endl;
    return 0;
}

// %% Cell 2 - Using Range
#include <iostream>
#include "file.hpp"

int main() {
    auto nums = example::range(10);
    int sum = 0;
    for (int n : nums) sum += n;
    std::cout << "Sum of 0-9: " << sum << std::endl;
    return 0;
}
