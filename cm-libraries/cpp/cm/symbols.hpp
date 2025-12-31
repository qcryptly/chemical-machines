/**
 * Chemical Machines Symbols Module for C++
 *
 * A header-only library for rendering LaTeX math expressions with notation styles.
 *
 * Notation Styles:
 *     - standard: Default LaTeX math notation
 *     - physicist: Physics notation (hbar, vectors with arrows, etc.)
 *     - chemist: Chemistry notation (reaction arrows, chemical formulas)
 *     - braket: Dirac bra-ket notation for quantum mechanics
 *     - engineering: Engineering notation (j for imaginary, etc.)
 *
 * Usage:
 *     #include <cm/symbols.hpp>
 *
 *     int main() {
 *         // Simple LaTeX rendering
 *         cm::symbols::latex("E = mc^2");
 *
 *         // Using the Math builder
 *         cm::symbols::Math m;
 *         m.frac("a", "b").plus().sqrt("c").render();
 *
 *         // Change notation style
 *         cm::symbols::set_notation("braket");
 *
 *         return 0;
 *     }
 */

#ifndef CM_SYMBOLS_HPP
#define CM_SYMBOLS_HPP

#include <string>
#include <sstream>
#include <vector>
#include <map>
#include "views.hpp"

namespace cm {
namespace symbols {

// Current notation style
inline std::string& notation_style() {
    static std::string style = "standard";
    return style;
}

// Current line height (default: normal)
inline std::string& line_height_value() {
    static std::string height = "normal";
    return height;
}

/**
 * Set the line height for math rendering.
 *
 * @param height CSS line-height value (e.g., "1", "1.5", "2", "normal", "1.2em")
 */
inline void set_line_height(const std::string& height) {
    line_height_value() = height;
}

/**
 * Get the current line height setting.
 */
inline std::string get_line_height() {
    return line_height_value();
}

// Helper to get inline style for line height
inline std::string get_line_height_style() {
    if (line_height_value() != "normal") {
        return " style=\"line-height: " + line_height_value() + ";\"";
    }
    return "";
}

/**
 * Set the notation style for math rendering.
 *
 * @param style One of "standard", "physicist", "chemist", "braket", "engineering"
 */
inline void set_notation(const std::string& style) {
    static const std::vector<std::string> valid = {
        "standard", "physicist", "chemist", "braket", "engineering"
    };
    for (const auto& v : valid) {
        if (v == style) {
            notation_style() = style;
            return;
        }
    }
    views::log_error("Unknown notation style:", style);
}

/**
 * Get the current notation style.
 */
inline std::string get_notation() {
    return notation_style();
}

/**
 * Render a LaTeX math expression.
 *
 * @param expression LaTeX math expression (without delimiters)
 * @param display If true, render as display math (centered). If false, inline.
 * @param label Optional label for the expression
 * @param justify Alignment: "left", "center", or "right"
 */
inline void latex(const std::string& expression, bool display = true,
                  const std::string& label = "", const std::string& justify = "center") {
    std::string delim_start = display ? "\\[" : "\\(";
    std::string delim_end = display ? "\\]" : "\\)";

    std::string justify_class = "cm-math-center";
    if (justify == "left") justify_class = "cm-math-left";
    else if (justify == "right") justify_class = "cm-math-right";

    std::string lh_style = get_line_height_style();

    std::ostringstream html;
    html << "<div class=\"cm-math " << justify_class << "\"" << lh_style << ">"
         << delim_start << expression << delim_end << "</div>";

    if (!label.empty()) {
        std::ostringstream labeled;
        labeled << "<div class=\"cm-math-labeled\"" << lh_style << "><span class=\"cm-math-label\">"
                << views::detail::html_escape(label) << "</span>" << html.str() << "</div>";
        views::html(labeled.str());
    } else {
        views::html(html.str());
    }
}

/**
 * Render a numbered equation.
 *
 * @param expression LaTeX math expression
 * @param number Equation number (as string)
 */
inline void equation(const std::string& expression, const std::string& number = "") {
    std::string lh_style = get_line_height_style();
    std::ostringstream html;
    if (!number.empty()) {
        html << "<div class=\"cm-equation\"" << lh_style << ">"
             << "<span class=\"cm-equation-content\">\\[" << expression << "\\]</span>"
             << "<span class=\"cm-equation-number\">(" << number << ")</span>"
             << "</div>";
    } else {
        html << "<div class=\"cm-equation\"" << lh_style << ">\\[" << expression << "\\]</div>";
    }
    views::html(html.str());
}

/**
 * Render aligned equations.
 *
 * @param equations Vector of LaTeX expressions with & for alignment
 */
inline void align(const std::vector<std::string>& equations) {
    std::ostringstream content;
    for (size_t i = 0; i < equations.size(); ++i) {
        if (i > 0) content << " \\\\ ";
        content << equations[i];
    }

    std::string lh_style = get_line_height_style();
    std::ostringstream html;
    html << "<div class=\"cm-math\"" << lh_style << ">\\[\\begin{aligned}"
         << content.str()
         << "\\end{aligned}\\]</div>";
    views::html(html.str());
}

/**
 * Render a matrix.
 *
 * @param data 2D vector of matrix elements (as strings)
 * @param style Matrix style: "pmatrix", "bmatrix", "vmatrix", "Vmatrix", "matrix"
 */
inline void matrix(const std::vector<std::vector<std::string>>& data,
                   const std::string& style = "pmatrix") {
    std::ostringstream content;
    for (size_t i = 0; i < data.size(); ++i) {
        if (i > 0) content << " \\\\ ";
        for (size_t j = 0; j < data[i].size(); ++j) {
            if (j > 0) content << " & ";
            content << data[i][j];
        }
    }

    std::string lh_style = get_line_height_style();
    std::ostringstream html;
    html << "<div class=\"cm-math\"" << lh_style << ">\\[\\begin{" << style << "}"
         << content.str()
         << "\\end{" << style << "}\\]</div>";
    views::html(html.str());
}

/**
 * Render a bulleted list of LaTeX expressions.
 *
 * @param expressions Vector of LaTeX expressions
 * @param display If true, use display math. If false, inline math.
 */
inline void bullets(const std::vector<std::string>& expressions, bool display = true) {
    std::string delim_start = display ? "\\[" : "\\(";
    std::string delim_end = display ? "\\]" : "\\)";

    std::string lh_style = get_line_height_style();
    std::ostringstream html;
    html << "<ul class=\"cm-math-list bulleted\"" << lh_style << ">";
    for (const auto& expr : expressions) {
        html << "<li>" << delim_start << expr << delim_end << "</li>";
    }
    html << "</ul>";
    views::html(html.str());
}

/**
 * Render a numbered list of LaTeX expressions.
 *
 * @param expressions Vector of LaTeX expressions
 * @param start Starting number (default: 1)
 * @param display If true, use display math. If false, inline math.
 */
inline void numbered(const std::vector<std::string>& expressions, int start = 1, bool display = true) {
    std::string delim_start = display ? "\\[" : "\\(";
    std::string delim_end = display ? "\\]" : "\\)";

    std::string lh_style = get_line_height_style();
    std::ostringstream html;
    html << "<ol class=\"cm-math-list numbered\" start=\"" << start << "\"" << lh_style << ">";
    for (const auto& expr : expressions) {
        html << "<li>" << delim_start << expr << delim_end << "</li>";
    }
    html << "</ol>";
    views::html(html.str());
}

/**
 * Render a plain list of LaTeX expressions (no bullets or numbers).
 *
 * @param expressions Vector of LaTeX expressions
 * @param display If true, use display math. If false, inline math.
 */
inline void items(const std::vector<std::string>& expressions, bool display = true) {
    std::string delim_start = display ? "\\[" : "\\(";
    std::string delim_end = display ? "\\]" : "\\)";

    std::string lh_style = get_line_height_style();
    std::ostringstream html;
    html << "<ul class=\"cm-math-list none\"" << lh_style << ">";
    for (const auto& expr : expressions) {
        html << "<li>" << delim_start << expr << delim_end << "</li>";
    }
    html << "</ul>";
    views::html(html.str());
}

/**
 * Render a chemical formula.
 *
 * @param formula Chemical formula (e.g., "H2O", "2H2 + O2 -> 2H2O")
 */
inline void chemical(const std::string& formula) {
    // Simple conversion: this is a basic implementation
    // For proper chemistry notation, consider using mhchem
    std::string result = formula;

    // Replace -> with rightarrow
    size_t pos;
    while ((pos = result.find("->")) != std::string::npos) {
        result.replace(pos, 2, " \\rightarrow ");
    }
    while ((pos = result.find("<->")) != std::string::npos) {
        result.replace(pos, 3, " \\rightleftharpoons ");
    }
    while ((pos = result.find("<=>")) != std::string::npos) {
        result.replace(pos, 3, " \\rightleftharpoons ");
    }

    latex("\\mathrm{" + result + "}");
}

/**
 * A builder class for constructing LaTeX expressions programmatically.
 */
class Math {
private:
    std::vector<std::string> parts_;

    Math& append(const std::string& content) {
        parts_.push_back(content);
        return *this;
    }

public:
    Math() = default;

    // Raw LaTeX
    Math& raw(const std::string& latex) { return append(latex); }

    // Text
    Math& text(const std::string& content) {
        return append("\\text{" + content + "}");
    }

    // Variable
    Math& var(const std::string& name) { return append(name); }

    // Basic operations
    Math& plus() { return append(" + "); }
    Math& minus() { return append(" - "); }
    Math& times() { return append(" \\times "); }
    Math& cdot() { return append(" \\cdot "); }
    Math& div() { return append(" \\div "); }
    Math& equals() { return append(" = "); }
    Math& approx() { return append(" \\approx "); }
    Math& neq() { return append(" \\neq "); }
    Math& lt() { return append(" < "); }
    Math& gt() { return append(" > "); }
    Math& leq() { return append(" \\leq "); }
    Math& geq() { return append(" \\geq "); }

    // Fractions and roots
    Math& frac(const std::string& num, const std::string& denom) {
        return append("\\frac{" + num + "}{" + denom + "}");
    }

    Math& sqrt(const std::string& content, const std::string& n = "") {
        if (!n.empty()) {
            return append("\\sqrt[" + n + "]{" + content + "}");
        }
        return append("\\sqrt{" + content + "}");
    }

    // Subscripts and superscripts
    Math& sub(const std::string& content) { return append("_{" + content + "}"); }
    Math& sup(const std::string& content) { return append("^{" + content + "}"); }
    Math& subsup(const std::string& s, const std::string& p) {
        return append("_{" + s + "}^{" + p + "}");
    }

    // Greek letters
    Math& alpha() { return append("\\alpha"); }
    Math& beta() { return append("\\beta"); }
    Math& gamma() { return append("\\gamma"); }
    Math& delta() { return append("\\delta"); }
    Math& epsilon() { return append("\\epsilon"); }
    Math& zeta() { return append("\\zeta"); }
    Math& eta() { return append("\\eta"); }
    Math& theta() { return append("\\theta"); }
    Math& iota() { return append("\\iota"); }
    Math& kappa() { return append("\\kappa"); }
    Math& lambda() { return append("\\lambda"); }
    Math& mu() { return append("\\mu"); }
    Math& nu() { return append("\\nu"); }
    Math& xi() { return append("\\xi"); }
    Math& pi() { return append("\\pi"); }
    Math& rho() { return append("\\rho"); }
    Math& sigma() { return append("\\sigma"); }
    Math& tau() { return append("\\tau"); }
    Math& upsilon() { return append("\\upsilon"); }
    Math& phi() { return append("\\phi"); }
    Math& chi() { return append("\\chi"); }
    Math& psi() { return append("\\psi"); }
    Math& omega() { return append("\\omega"); }

    // Capital Greek
    Math& Gamma() { return append("\\Gamma"); }
    Math& Delta() { return append("\\Delta"); }
    Math& Theta() { return append("\\Theta"); }
    Math& Lambda() { return append("\\Lambda"); }
    Math& Xi() { return append("\\Xi"); }
    Math& Pi() { return append("\\Pi"); }
    Math& Sigma() { return append("\\Sigma"); }
    Math& Phi() { return append("\\Phi"); }
    Math& Psi() { return append("\\Psi"); }
    Math& Omega() { return append("\\Omega"); }

    // Calculus
    Math& integral(const std::string& lower = "", const std::string& upper = "") {
        if (!lower.empty() && !upper.empty()) {
            return append("\\int_{" + lower + "}^{" + upper + "}");
        } else if (!lower.empty()) {
            return append("\\int_{" + lower + "}");
        }
        return append("\\int");
    }

    Math& sum(const std::string& lower = "", const std::string& upper = "") {
        if (!lower.empty() && !upper.empty()) {
            return append("\\sum_{" + lower + "}^{" + upper + "}");
        } else if (!lower.empty()) {
            return append("\\sum_{" + lower + "}");
        }
        return append("\\sum");
    }

    Math& prod(const std::string& lower = "", const std::string& upper = "") {
        if (!lower.empty() && !upper.empty()) {
            return append("\\prod_{" + lower + "}^{" + upper + "}");
        } else if (!lower.empty()) {
            return append("\\prod_{" + lower + "}");
        }
        return append("\\prod");
    }

    Math& lim(const std::string& var, const std::string& to) {
        return append("\\lim_{" + var + " \\to " + to + "}");
    }

    Math& deriv(const std::string& func = "", const std::string& var = "x") {
        if (!func.empty()) {
            return append("\\frac{d" + func + "}{d" + var + "}");
        }
        return append("\\frac{d}{d" + var + "}");
    }

    Math& partial(const std::string& func = "", const std::string& var = "x") {
        if (!func.empty()) {
            return append("\\frac{\\partial " + func + "}{\\partial " + var + "}");
        }
        return append("\\frac{\\partial}{\\partial " + var + "}");
    }

    Math& nabla() { return append("\\nabla"); }

    // Brackets
    Math& paren(const std::string& content) {
        return append("\\left(" + content + "\\right)");
    }

    Math& bracket(const std::string& content) {
        return append("\\left[" + content + "\\right]");
    }

    Math& brace(const std::string& content) {
        return append("\\left\\{" + content + "\\right\\}");
    }

    Math& abs(const std::string& content) {
        return append("\\left|" + content + "\\right|");
    }

    Math& norm(const std::string& content) {
        return append("\\left\\|" + content + "\\right\\|");
    }

    // Quantum mechanics / Bra-ket
    Math& bra(const std::string& content) {
        return append("\\langle " + content + " |");
    }

    Math& ket(const std::string& content) {
        return append("| " + content + " \\rangle");
    }

    Math& braket(const std::string& b, const std::string& k) {
        return append("\\langle " + b + " | " + k + " \\rangle");
    }

    Math& expval(const std::string& op) {
        return append("\\langle " + op + " \\rangle");
    }

    Math& matelem(const std::string& b, const std::string& op, const std::string& k) {
        return append("\\langle " + b + " | " + op + " | " + k + " \\rangle");
    }

    Math& op(const std::string& name) {
        return append("\\hat{" + name + "}");
    }

    Math& dagger() { return append("^\\dagger"); }

    Math& comm(const std::string& a, const std::string& b) {
        return append("[" + a + ", " + b + "]");
    }

    // Physics
    Math& vec(const std::string& content) {
        return append("\\vec{" + content + "}");
    }

    Math& hbar() { return append("\\hbar"); }
    Math& infty() { return append("\\infty"); }

    // Chemistry
    Math& ce(const std::string& formula) {
        return append("\\mathrm{" + formula + "}");
    }

    Math& yields() { return append(" \\rightarrow "); }
    Math& equilibrium() { return append(" \\rightleftharpoons "); }

    // Special functions
    Math& sin(const std::string& arg = "") {
        return append(arg.empty() ? "\\sin" : "\\sin{" + arg + "}");
    }

    Math& cos(const std::string& arg = "") {
        return append(arg.empty() ? "\\cos" : "\\cos{" + arg + "}");
    }

    Math& tan(const std::string& arg = "") {
        return append(arg.empty() ? "\\tan" : "\\tan{" + arg + "}");
    }

    Math& ln(const std::string& arg = "") {
        return append(arg.empty() ? "\\ln" : "\\ln{" + arg + "}");
    }

    Math& log(const std::string& arg = "", const std::string& base = "") {
        if (!base.empty()) {
            return append(arg.empty() ? "\\log_{" + base + "}" : "\\log_{" + base + "}{" + arg + "}");
        }
        return append(arg.empty() ? "\\log" : "\\log{" + arg + "}");
    }

    Math& exp(const std::string& arg = "") {
        return append(arg.empty() ? "\\exp" : "\\exp{" + arg + "}");
    }

    // Spacing
    Math& space() { return append("\\ "); }
    Math& quad() { return append("\\quad"); }
    Math& qquad() { return append("\\qquad"); }

    // Build and render
    std::string build() const {
        std::ostringstream ss;
        for (const auto& p : parts_) {
            ss << p;
        }
        return ss.str();
    }

    void render(bool display = true, const std::string& label = "") {
        latex(build(), display, label);
    }

    Math& clear() {
        parts_.clear();
        return *this;
    }
};

} // namespace symbols
} // namespace cm

#endif // CM_SYMBOLS_HPP
