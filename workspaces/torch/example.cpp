// Single-file C++ example
// This file is executed as a single unit (no cells)
// It includes file.hpp from the same directory

#include <iostream>
#include "file.hpp"

int main() {
    // Use the greet function from our header
    std::cout << example::greet("World") << std::endl;

    // Use the range function
    auto numbers = example::range(5);
    std::cout << "Numbers: ";
    for (int n : numbers) {
        std::cout << n << " ";
    }
    std::cout << std::endl;

    return 0;
}
