// %% Cell 1 - HTML Output Demo
// Demonstrates the cm::views module for C++

#include <iostream>
#include <cm/views.hpp>

int main() {
    // Output formatted HTML
    cm::views::html("<h2>C++ HTML Output Demo</h2>");
    cm::views::html("<p>This demonstrates <strong>rich HTML output</strong> from C++.</p>");

    // Log messages
    cm::views::log("Processing started...");
    cm::views::log_success("Cell 1 complete!");

    return 0;
}

// %% Cell 2 - Tables
#include <iostream>
#include <vector>
#include <string>
#include <cm/views.hpp>

int main() {
    // Display a table of data
    std::vector<std::vector<std::string>> data = {
        {"Hydrogen", "H", "1.008"},
        {"Helium", "He", "4.003"},
        {"Lithium", "Li", "6.941"},
        {"Beryllium", "Be", "9.012"}
    };

    std::vector<std::string> headers = {"Element", "Symbol", "Atomic Mass"};

    cm::views::table(data, headers);
    cm::views::log("Table with 4 elements displayed");

    return 0;
}

// %% Cell 3 - Logging Levels
#include <iostream>
#include <cm/views.hpp>

int main() {
    cm::views::html("<h3>Log Level Examples</h3>");

    cm::views::log("This is a regular log message");
    cm::views::log_success("Operation completed successfully!");
    cm::views::log_warning("This is a warning message");
    cm::views::log_error("This is an error message");

    // Multiple arguments
    int value = 42;
    cm::views::log("The answer is:", value);

    return 0;
}

// %% Cell 4 - LaTeX Math with Line Height
#include <iostream>
#include <cm/symbols.hpp>

int main() {
    cm::views::html("<h3>LaTeX Math Demo</h3>");

    // Set custom line height (try: "1", "1.5", "2", "normal")
    cm::symbols::set_line_height("1.8");

    // Render equations
    cm::symbols::latex("E = mc^2");
    cm::symbols::latex("\\int_0^\\infty e^{-x} dx = 1");
    cm::symbols::equation("F = ma", "1");

    // Reset to normal
    cm::symbols::set_line_height("normal");
    cm::symbols::latex("\\nabla \\times \\mathbf{B} = \\mu_0 \\mathbf{J}");

    cm::views::log_success("LaTeX rendering complete!");

    return 0;
}
