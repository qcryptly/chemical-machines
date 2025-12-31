#pragma once

// Example header file
// This can be included by both example.cpp and example.cell.cpp

#include <string>
#include <vector>

namespace example {

inline std::string greet(const std::string& name) {
    return "Hello, " + name + "!";
}

inline std::vector<int> range(int n) {
    std::vector<int> result;
    for (int i = 0; i < n; ++i) {
        result.push_back(i);
    }
    return result;
}

} // namespace example
