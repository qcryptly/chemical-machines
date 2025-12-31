/**
 * Chemical Machines Views Module for C++
 *
 * A header-only library for rendering HTML outputs from C++ cells.
 * Outputs are written to .out/ directory and displayed in the workspace UI.
 *
 * Environment Variables:
 *     CM_OUTPUT_FILE: Path to the output HTML file
 *     CM_CELL_INDEX: Current cell index (0-based), or -1 for non-cell files
 *     CM_IS_CELL_FILE: "true" if this is a cell-based file
 *     CM_WORKSPACE_DIR: Path to the workspace directory
 *
 * Usage:
 *     #include <cm/views.hpp>
 *
 *     int main() {
 *         cm::views::html("<h1>Hello World</h1>");
 *         cm::views::text("Some plain text");
 *         cm::views::log("Status:", 42);
 *         cm::views::log_error("Something went wrong!");
 *         cm::views::image("plot.png");
 *         cm::views::clear();
 *         return 0;
 *     }
 */

#ifndef CM_VIEWS_HPP
#define CM_VIEWS_HPP

#include <string>
#include <sstream>
#include <fstream>
#include <map>
#include <vector>
#include <cstdlib>
#include <filesystem>
#include <algorithm>
#include <iomanip>

namespace cm {
namespace views {

namespace detail {

// Global state
inline std::string& output_file() {
    static std::string file = []() {
        const char* env = std::getenv("CM_OUTPUT_FILE");
        return env ? std::string(env) : "";
    }();
    return file;
}

inline int& cell_index() {
    static int idx = []() {
        const char* env = std::getenv("CM_CELL_INDEX");
        return env ? std::stoi(env) : -1;
    }();
    return idx;
}

inline bool& is_cell_file() {
    static bool is_cell = []() {
        const char* env = std::getenv("CM_IS_CELL_FILE");
        return env && std::string(env) == "true";
    }();
    return is_cell;
}

inline std::string& workspace_dir() {
    static std::string dir = []() {
        const char* env = std::getenv("CM_WORKSPACE_DIR");
        return env ? std::string(env) : "";
    }();
    return dir;
}

// Storage for CURRENT cell outputs only (each cell runs in separate process)
inline std::vector<std::string>& current_cell_outputs() {
    static std::vector<std::string> outputs;
    return outputs;
}

// HTML escape
inline std::string html_escape(const std::string& str) {
    std::string result;
    result.reserve(str.size());
    for (char c : str) {
        switch (c) {
            case '&':  result += "&amp;";  break;
            case '<':  result += "&lt;";   break;
            case '>':  result += "&gt;";   break;
            case '"':  result += "&quot;"; break;
            case '\'': result += "&#39;";  break;
            default:   result += c;
        }
    }
    return result;
}

// Base64 encoding
inline std::string base64_encode(const std::vector<unsigned char>& data) {
    static const char* chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string result;
    int val = 0, valb = -6;

    for (unsigned char c : data) {
        val = (val << 8) + c;
        valb += 8;
        while (valb >= 0) {
            result.push_back(chars[(val >> valb) & 0x3F]);
            valb -= 6;
        }
    }

    if (valb > -6) {
        result.push_back(chars[((val << 8) >> (valb + 8)) & 0x3F]);
    }

    while (result.size() % 4) {
        result.push_back('=');
    }

    return result;
}

// Cell delimiter for separating outputs between cells
const std::string CELL_DELIMITER = "<!-- CELL_DELIMITER -->";

// HTML template parts
const std::string HTML_HEADER = R"(<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 1rem;
            background: white;
            color: #333;
        }
        pre {
            background: #f5f5f5;
            padding: 0.5rem;
            border-radius: 4px;
            overflow-x: auto;
            margin: 0.25rem 0;
        }
        code {
            background: #f5f5f5;
            padding: 0.1rem 0.3rem;
            border-radius: 3px;
            font-family: 'Monaco', 'Menlo', monospace;
        }
        img {
            max-width: 100%;
            height: auto;
        }
        table {
            border-collapse: collapse;
            margin: 0.5rem 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 0.5rem;
            text-align: left;
        }
        th {
            background: #f5f5f5;
        }
        .cm-log {
            margin: 0.25rem 0;
            padding: 0.25rem 0.5rem;
            border-left: 3px solid #007acc;
            background: #f8f9fa;
        }
        .cm-error {
            border-left-color: #d32f2f;
            background: #ffebee;
        }
        .cm-warning {
            border-left-color: #f9a825;
            background: #fff8e1;
        }
        .cm-success {
            border-left-color: #388e3c;
            background: #e8f5e9;
        }
    </style>
</head>
<body>
)";

const std::string HTML_FOOTER = R"(
</body>
</html>)";

// Trim whitespace from string
inline std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\n\r");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\n\r");
    return str.substr(first, last - first + 1);
}

// Read existing cell outputs from HTML file
inline std::map<int, std::string> read_existing_cells() {
    std::map<int, std::string> cells;
    const std::string& file = output_file();

    if (file.empty() || !std::filesystem::exists(file)) {
        return cells;
    }

    try {
        std::ifstream in(file);
        std::stringstream buffer;
        buffer << in.rdbuf();
        std::string content = buffer.str();

        // Extract body content
        size_t body_start = content.find("<body>");
        size_t body_end = content.find("</body>");
        if (body_start == std::string::npos || body_end == std::string::npos) {
            return cells;
        }

        std::string body_content = content.substr(body_start + 6, body_end - body_start - 6);
        body_content = trim(body_content);

        // Split by cell delimiter
        std::vector<std::string> parts;
        size_t pos = 0;
        size_t delimiter_pos;
        while ((delimiter_pos = body_content.find(CELL_DELIMITER, pos)) != std::string::npos) {
            parts.push_back(trim(body_content.substr(pos, delimiter_pos - pos)));
            pos = delimiter_pos + CELL_DELIMITER.length();
        }
        parts.push_back(trim(body_content.substr(pos)));

        // Build map of cell index -> content
        for (size_t idx = 0; idx < parts.size(); ++idx) {
            if (!parts[idx].empty()) {
                cells[static_cast<int>(idx)] = parts[idx];
            }
        }
    } catch (...) {
        // Ignore errors, return empty
    }

    return cells;
}

// Write all outputs to file, preserving other cells
inline void write_outputs() {
    const std::string& file = output_file();
    if (file.empty()) return;

    // Ensure parent directory exists
    std::filesystem::path filepath(file);
    std::filesystem::create_directories(filepath.parent_path());

    int cell_idx = cell_index() >= 0 ? cell_index() : 0;

    // Build content
    std::string content;

    if (is_cell_file()) {
        // Read existing cells from file
        std::map<int, std::string> existing_cells = read_existing_cells();

        // Build current cell's content
        std::ostringstream current_content;
        for (const auto& output : current_cell_outputs()) {
            current_content << output << "\n";
        }
        existing_cells[cell_idx] = trim(current_content.str());

        // Find max index
        int max_idx = 0;
        for (const auto& [idx, _] : existing_cells) {
            if (idx > max_idx) max_idx = idx;
        }

        // Build output with all cells in order
        std::ostringstream combined;
        for (int i = 0; i <= max_idx; ++i) {
            if (i > 0) combined << "\n" << CELL_DELIMITER << "\n";
            auto it = existing_cells.find(i);
            if (it != existing_cells.end()) {
                combined << it->second;
            }
        }
        content = combined.str();
    } else {
        // Non-cell file: just output current content
        std::ostringstream ss;
        for (const auto& output : current_cell_outputs()) {
            ss << output << "\n";
        }
        content = ss.str();
    }

    // Write full HTML
    std::ofstream out(file);
    out << HTML_HEADER << content << HTML_FOOTER;
}

// Add output for the current cell
inline void add_output(const std::string& content, int = -2) {
    current_cell_outputs().push_back(content);
    write_outputs();
}

// Variadic log helper
template<typename T>
inline std::string to_log_string(const T& val) {
    std::ostringstream ss;
    ss << val;
    return ss.str();
}

template<typename T, typename... Args>
inline std::string join_args(const T& first, const Args&... rest) {
    std::string result = to_log_string(first);
    if constexpr (sizeof...(rest) > 0) {
        result += " " + join_args(rest...);
    }
    return result;
}

} // namespace detail

/**
 * Output raw HTML content.
 *
 * @param content HTML string to output
 * @param cell_index Optional cell index (-2 = use current cell)
 */
inline void html(const std::string& content, int cell_index = -2) {
    detail::add_output(content, cell_index);
}

/**
 * Output plain text (HTML-escaped).
 *
 * @param content Plain text string to output
 * @param cell_index Optional cell index
 */
inline void text(const std::string& content, int cell_index = -2) {
    detail::add_output("<pre>" + detail::html_escape(content) + "</pre>", cell_index);
}

/**
 * Log values with automatic formatting.
 *
 * @param args Values to log
 */
template<typename... Args>
inline void log(const Args&... args) {
    std::string content = detail::join_args(args...);
    std::string escaped = detail::html_escape(content);
    detail::add_output("<div class=\"cm-log\"><pre>" + escaped + "</pre></div>");
}

/**
 * Log an error message.
 */
template<typename... Args>
inline void log_error(const Args&... args) {
    std::string content = detail::join_args(args...);
    std::string escaped = detail::html_escape(content);
    detail::add_output("<div class=\"cm-log cm-error\"><pre>" + escaped + "</pre></div>");
}

/**
 * Log a warning message.
 */
template<typename... Args>
inline void log_warning(const Args&... args) {
    std::string content = detail::join_args(args...);
    std::string escaped = detail::html_escape(content);
    detail::add_output("<div class=\"cm-log cm-warning\"><pre>" + escaped + "</pre></div>");
}

/**
 * Log a success message.
 */
template<typename... Args>
inline void log_success(const Args&... args) {
    std::string content = detail::join_args(args...);
    std::string escaped = detail::html_escape(content);
    detail::add_output("<div class=\"cm-log cm-success\"><pre>" + escaped + "</pre></div>");
}

/**
 * Output an image from file path.
 *
 * @param filepath Path to image file
 * @param alt Alt text
 * @param cell_index Optional cell index
 */
inline void image(const std::string& filepath, const std::string& alt = "", int cell_index = -2) {
    namespace fs = std::filesystem;

    fs::path path(filepath);
    if (!path.is_absolute() && !detail::workspace_dir().empty()) {
        path = fs::path(detail::workspace_dir()) / filepath;
    }

    if (!fs::exists(path)) {
        log_error("Image not found:", filepath);
        return;
    }

    // Read file
    std::ifstream file(path, std::ios::binary);
    std::vector<unsigned char> data(
        (std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>()
    );

    // Detect MIME type
    std::string mime = "image/png";
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    if (ext == ".jpg" || ext == ".jpeg") mime = "image/jpeg";
    else if (ext == ".gif") mime = "image/gif";
    else if (ext == ".svg") mime = "image/svg+xml";
    else if (ext == ".webp") mime = "image/webp";

    // Base64 encode
    std::string b64 = detail::base64_encode(data);
    std::string src = "data:" + mime + ";base64," + b64;
    std::string alt_escaped = detail::html_escape(alt);

    detail::add_output("<img src=\"" + src + "\" alt=\"" + alt_escaped + "\">", cell_index);
}

/**
 * Output an image from binary data.
 *
 * @param data Image data
 * @param mime_type MIME type
 * @param alt Alt text
 * @param cell_index Optional cell index
 */
inline void image_data(const std::vector<unsigned char>& data,
                       const std::string& mime_type = "image/png",
                       const std::string& alt = "",
                       int cell_index = -2) {
    std::string b64 = detail::base64_encode(data);
    std::string src = "data:" + mime_type + ";base64," + b64;
    std::string alt_escaped = detail::html_escape(alt);

    detail::add_output("<img src=\"" + src + "\" alt=\"" + alt_escaped + "\">", cell_index);
}

/**
 * Output a table.
 *
 * @param data 2D vector of cell values (as strings)
 * @param headers Optional header row
 * @param cell_index Optional cell index
 */
inline void table(const std::vector<std::vector<std::string>>& data,
                  const std::vector<std::string>& headers = {},
                  int cell_index = -2) {
    std::ostringstream html;
    html << "<table>";

    if (!headers.empty()) {
        html << "<tr>";
        for (const auto& h : headers) {
            html << "<th>" << detail::html_escape(h) << "</th>";
        }
        html << "</tr>";
    }

    for (const auto& row : data) {
        html << "<tr>";
        for (const auto& cell : row) {
            html << "<td>" << detail::html_escape(cell) << "</td>";
        }
        html << "</tr>";
    }

    html << "</table>";
    detail::add_output(html.str(), cell_index);
}

/**
 * Clear all outputs for the current cell.
 */
inline void clear() {
    detail::current_cell_outputs().clear();
    detail::write_outputs();
}

/**
 * Clear all outputs for all cells (deletes the output file).
 */
inline void clear_all() {
    const std::string& file = detail::output_file();
    if (!file.empty() && std::filesystem::exists(file)) {
        std::filesystem::remove(file);
    }
}

/**
 * Output WebGL/3D content to the main visualization panel.
 *
 * This writes to a special .out/main.webgl.html file that is displayed
 * in the collapsible WebGL panel at the top of the workspace.
 *
 * @param content Full HTML content including WebGL/Three.js code
 */
inline void webgl(const std::string& content) {
    const std::string& workspace = detail::workspace_dir();
    if (workspace.empty()) {
        log_error("CM_WORKSPACE_DIR not set, cannot write WebGL output");
        return;
    }

    namespace fs = std::filesystem;
    fs::path webgl_path = fs::path(workspace) / ".out" / "main.webgl.html";
    fs::create_directories(webgl_path.parent_path());

    std::ofstream out(webgl_path);
    out << content;
}

/**
 * Output a Three.js scene with common boilerplate handled.
 *
 * @param scene_setup JavaScript code to set up the scene
 * @param animate_loop Optional JavaScript code to run each frame
 * @param background Background color (default: dark theme)
 * @param camera_x Initial camera X position
 * @param camera_y Initial camera Y position
 * @param camera_z Initial camera Z position
 * @param controls Enable OrbitControls (default: true)
 */
inline void webgl_threejs(
    const std::string& scene_setup,
    const std::string& animate_loop = "",
    const std::string& background = "#1e1e2e",
    float camera_x = 5.0f,
    float camera_y = 5.0f,
    float camera_z = 5.0f,
    bool controls = true
) {
    std::string controls_import = controls ?
        "<script src=\"https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js\"></script>" : "";

    std::string controls_setup = controls ? R"(
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
    )" : "";

    std::string controls_update = controls ? "controls.update();" : "";

    std::ostringstream content;
    content << R"(<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            width: 100%;
            height: 100%;
            overflow: hidden;
            background: )" << background << R"(;
        }
        canvas { display: block; width: 100% !important; height: 100% !important; }
        .nav-controls {
            position: absolute;
            bottom: 10px;
            right: 10px;
            display: flex;
            gap: 4px;
            z-index: 100;
        }
        .nav-btn {
            width: 32px;
            height: 32px;
            background: rgba(40, 40, 55, 0.85);
            border: 1px solid #444;
            border-radius: 4px;
            color: #aaa;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
            transition: all 0.15s;
        }
        .nav-btn:hover {
            background: rgba(60, 60, 80, 0.9);
            color: #fff;
            border-color: #666;
        }
        .nav-btn:active {
            background: rgba(80, 80, 100, 0.9);
        }
        .nav-btn svg {
            width: 16px;
            height: 16px;
            fill: currentColor;
        }
        .nav-separator {
            width: 1px;
            background: #444;
            margin: 4px 2px;
        }
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    )" << controls_import << R"(
</head>
<body>
    <!-- Navigation controls -->
    <div class="nav-controls">
        <button class="nav-btn" id="viewTop" title="Top view (Z+)">
            <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"/><path d="M12 2v4M12 18v4"/></svg>
        </button>
        <button class="nav-btn" id="viewFront" title="Front view (Y-)">
            <svg viewBox="0 0 24 24"><rect x="6" y="6" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2"/></svg>
        </button>
        <button class="nav-btn" id="viewSide" title="Side view (X+)">
            <svg viewBox="0 0 24 24"><path d="M4 6h16v12H4z" fill="none" stroke="currentColor" stroke-width="2"/><path d="M4 6l4-4h12l-4 4" fill="none" stroke="currentColor" stroke-width="1.5"/></svg>
        </button>
        <button class="nav-btn" id="viewIso" title="Isometric view">
            <svg viewBox="0 0 24 24"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" fill="none" stroke="currentColor" stroke-width="2"/></svg>
        </button>
        <div class="nav-separator"></div>
        <button class="nav-btn" id="rotateLeft" title="Rotate left">
            <svg viewBox="0 0 24 24"><path d="M12 5V1L7 6l5 5V7c3.31 0 6 2.69 6 6s-2.69 6-6 6-6-2.69-6-6H4c0 4.42 3.58 8 8 8s8-3.58 8-8-3.58-8-8-8z"/></svg>
        </button>
        <button class="nav-btn" id="rotateRight" title="Rotate right">
            <svg viewBox="0 0 24 24"><path d="M12 5V1l5 5-5 5V7c-3.31 0-6 2.69-6 6s2.69 6 6 6 6-2.69 6-6h2c0 4.42-3.58 8-8 8s-8-3.58-8-8 3.58-8 8-8z"/></svg>
        </button>
        <div class="nav-separator"></div>
        <button class="nav-btn" id="resetView" title="Reset view">
            <svg viewBox="0 0 24 24"><path d="M12 5V1L7 6l5 5V7c3.31 0 6 2.69 6 6s-2.69 6-6 6-6-2.69-6-6H4c0 4.42 3.58 8 8 8s8-3.58 8-8-3.58-8-8-8z"/><circle cx="12" cy="13" r="2"/></svg>
        </button>
    </div>

    <script>
        // Scene setup
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(')" << background << R"(');

        // Camera (Z-up orientation)
        const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.up.set(0, 0, 1);  // Z is up
        const initialCameraPos = new THREE.Vector3()" << camera_x << ", " << camera_y << ", " << camera_z << R"();
        camera.position.copy(initialCameraPos);
        camera.lookAt(0, 0, 0);

        // Renderer
        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        document.body.appendChild(renderer.domElement);

        )" << controls_setup << R"(

        // User scene setup
        )" << scene_setup << R"(

        // Navigation control functions
        const cameraDistance = initialCameraPos.length();

        function setCameraView(x, y, z) {
            const dir = new THREE.Vector3(x, y, z).normalize().multiplyScalar(cameraDistance);
            camera.position.copy(dir);
            controls.update();
        }

        document.getElementById('viewTop').onclick = () => setCameraView(0, 0, 1);
        document.getElementById('viewFront').onclick = () => setCameraView(0, -1, 0.3);
        document.getElementById('viewSide').onclick = () => setCameraView(1, 0, 0.3);
        document.getElementById('viewIso').onclick = () => setCameraView(1, 1, 1);

        document.getElementById('rotateLeft').onclick = () => {
            const spherical = new THREE.Spherical().setFromVector3(camera.position);
            spherical.theta += Math.PI / 8;
            camera.position.setFromSpherical(spherical);
            controls.update();
        };

        document.getElementById('rotateRight').onclick = () => {
            const spherical = new THREE.Spherical().setFromVector3(camera.position);
            spherical.theta -= Math.PI / 8;
            camera.position.setFromSpherical(spherical);
            controls.update();
        };

        document.getElementById('resetView').onclick = () => {
            camera.position.copy(initialCameraPos);
            controls.target.set(0, 0, 0);
            controls.update();
        };

        // Handle resize
        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });

        // Animation loop
        function animate() {
            requestAnimationFrame(animate);
            )" << controls_update << R"(
            )" << animate_loop << R"(
            renderer.render(scene, camera);
        }
        animate();
    </script>
</body>
</html>)";

    webgl(content.str());
}

// =============================================================================
// Scientific Visualization Library
// =============================================================================

namespace detail {

// Colormap data (viridis)
inline const std::vector<std::string>& viridis_colors() {
    static std::vector<std::string> colors = {
        "#440154", "#482878", "#3E4A89", "#31688E", "#26838E",
        "#1F9E89", "#35B779", "#6DCD59", "#B4DE2C", "#FDE725"
    };
    return colors;
}

// Interpolate colormap
inline std::string interpolate_colormap(double value, const std::vector<std::string>& colors) {
    value = std::max(0.0, std::min(1.0, value));
    size_t n = colors.size();
    double idx = value * (n - 1);
    size_t idx_low = static_cast<size_t>(idx);
    size_t idx_high = std::min(idx_low + 1, n - 1);
    double t = idx - idx_low;

    // Parse hex colors
    auto parse_hex = [](const std::string& c, int offset) {
        return std::stoi(c.substr(offset, 2), nullptr, 16);
    };

    int r1 = parse_hex(colors[idx_low], 1);
    int g1 = parse_hex(colors[idx_low], 3);
    int b1 = parse_hex(colors[idx_low], 5);
    int r2 = parse_hex(colors[idx_high], 1);
    int g2 = parse_hex(colors[idx_high], 3);
    int b2 = parse_hex(colors[idx_high], 5);

    int r = static_cast<int>(r1 + t * (r2 - r1));
    int g = static_cast<int>(g1 + t * (g2 - g1));
    int b = static_cast<int>(b1 + t * (b2 - b1));

    std::ostringstream ss;
    ss << "#" << std::hex << std::setfill('0')
       << std::setw(2) << r
       << std::setw(2) << g
       << std::setw(2) << b;
    return ss.str();
}

// Element data for molecular visualization
struct ElementInfo {
    std::string color;
    double radius;
};

inline const std::map<std::string, ElementInfo>& element_data() {
    static std::map<std::string, ElementInfo> data = {
        {"H",  {"#FFFFFF", 0.31}}, {"He", {"#D9FFFF", 0.28}},
        {"Li", {"#CC80FF", 1.28}}, {"Be", {"#C2FF00", 0.96}},
        {"B",  {"#FFB5B5", 0.84}}, {"C",  {"#909090", 0.77}},
        {"N",  {"#3050F8", 0.71}}, {"O",  {"#FF0D0D", 0.66}},
        {"F",  {"#90E050", 0.57}}, {"Ne", {"#B3E3F5", 0.58}},
        {"Na", {"#AB5CF2", 1.66}}, {"Mg", {"#8AFF00", 1.41}},
        {"Al", {"#BFA6A6", 1.21}}, {"Si", {"#F0C8A0", 1.11}},
        {"P",  {"#FF8000", 1.07}}, {"S",  {"#FFFF30", 1.05}},
        {"Cl", {"#1FF01F", 1.02}}, {"Ar", {"#80D1E3", 1.06}},
        {"K",  {"#8F40D4", 2.03}}, {"Ca", {"#3DFF00", 1.76}},
        {"Fe", {"#E06633", 1.32}}, {"Cu", {"#C88033", 1.32}},
        {"Zn", {"#7D80B0", 1.22}}, {"Br", {"#A62929", 1.20}},
        {"Ag", {"#C0C0C0", 1.45}}, {"Au", {"#FFD123", 1.36}},
    };
    return data;
}

inline ElementInfo get_element(const std::string& symbol) {
    const auto& data = element_data();
    auto it = data.find(symbol);
    if (it != data.end()) return it->second;
    return {"#FF00FF", 1.0};
}

// Generate unit box JavaScript (cubic, Z-up orientation, centered at given point)
inline std::string generate_unit_box_js(double w, double h, double d,
                                        const std::string& color = "#444444",
                                        double cx = 0.0, double cy = 0.0, double cz = 0.0) {
    std::ostringstream js;
    // Use maximum dimension to create a cube
    double max_dim = std::max({w, h, d});
    double s = max_dim / 2;  // half-size for cube

    // Box vertices centered around (cx, cy, cz)
    double x_min = cx - s, x_max = cx + s;
    double y_min = cy - s, y_max = cy + s;
    double z_min = cz - s, z_max = cz + s;

    js << R"(
        // Unit box (cubic, Z-up, centered at ()" << cx << ", " << cy << ", " << cz << R"())
        const boxGeometry = new THREE.BufferGeometry();
        const boxVertices = new Float32Array([
            // Bottom face (Z = z_min)
            )" << x_min << ", " << y_min << ", " << z_min << ", " << x_max << ", " << y_min << ", " << z_min << R"(,
            )" << x_max << ", " << y_min << ", " << z_min << ", " << x_max << ", " << y_max << ", " << z_min << R"(,
            )" << x_max << ", " << y_max << ", " << z_min << ", " << x_min << ", " << y_max << ", " << z_min << R"(,
            )" << x_min << ", " << y_max << ", " << z_min << ", " << x_min << ", " << y_min << ", " << z_min << R"(,
            // Top face (Z = z_max)
            )" << x_min << ", " << y_min << ", " << z_max << ", " << x_max << ", " << y_min << ", " << z_max << R"(,
            )" << x_max << ", " << y_min << ", " << z_max << ", " << x_max << ", " << y_max << ", " << z_max << R"(,
            )" << x_max << ", " << y_max << ", " << z_max << ", " << x_min << ", " << y_max << ", " << z_max << R"(,
            )" << x_min << ", " << y_max << ", " << z_max << ", " << x_min << ", " << y_min << ", " << z_max << R"(,
            // Vertical edges (connecting bottom to top)
            )" << x_min << ", " << y_min << ", " << z_min << ", " << x_min << ", " << y_min << ", " << z_max << R"(,
            )" << x_max << ", " << y_min << ", " << z_min << ", " << x_max << ", " << y_min << ", " << z_max << R"(,
            )" << x_max << ", " << y_max << ", " << z_min << ", " << x_max << ", " << y_max << ", " << z_max << R"(,
            )" << x_min << ", " << y_max << ", " << z_min << ", " << x_min << ", " << y_max << ", " << z_max << R"(
        ]);
        boxGeometry.setAttribute('position', new THREE.BufferAttribute(boxVertices, 3));
        const boxMaterial = new THREE.LineBasicMaterial({ color: ')" << color << R"(', transparent: true, opacity: 0.5 });
        const unitBox = new THREE.LineSegments(boxGeometry, boxMaterial);
        scene.add(unitBox);

        // Axis labels using sprites (Z-up coordinate system)
        function createTextSprite(text, position) {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            canvas.width = 64;
            canvas.height = 32;
            ctx.fillStyle = '#888888';
            ctx.font = '20px Arial';
            ctx.textAlign = 'center';
            ctx.fillText(text, 32, 24);
            const texture = new THREE.CanvasTexture(canvas);
            const material = new THREE.SpriteMaterial({ map: texture });
            const sprite = new THREE.Sprite(material);
            sprite.position.copy(position);
            sprite.scale.set(1, 0.5, 1);
            return sprite;
        }
        scene.add(createTextSprite('X', new THREE.Vector3()" << (x_max + 0.3) << ", " << cy << ", " << cz << R"()));
        scene.add(createTextSprite('Y', new THREE.Vector3()" << cx << ", " << (y_max + 0.3) << ", " << cz << R"()));
        scene.add(createTextSprite('Z', new THREE.Vector3()" << cx << ", " << cy << ", " << (z_max + 0.3) << R"()));
    )";

    return js.str();
}

} // namespace detail

/**
 * Atom structure for molecule visualization.
 */
struct Atom {
    std::string element;
    double x, y, z;
};

/**
 * Bond structure for molecule visualization.
 */
struct Bond {
    size_t atom1, atom2;
};

/**
 * Render a 3D scatter plot of points.
 *
 * @param points Vector of {x, y, z} coordinates
 * @param point_size Size of each point
 * @param show_box Show bounding box
 * @param background Background color
 */
inline void scatter_3d(
    const std::vector<std::array<double, 3>>& points,
    double point_size = 0.1,
    bool show_box = true,
    const std::string& background = "#1e1e2e"
) {
    if (points.empty()) return;

    // Calculate bounds
    double min_x = points[0][0], max_x = points[0][0];
    double min_y = points[0][1], max_y = points[0][1];
    double min_z = points[0][2], max_z = points[0][2];

    for (const auto& p : points) {
        min_x = std::min(min_x, p[0]); max_x = std::max(max_x, p[0]);
        min_y = std::min(min_y, p[1]); max_y = std::max(max_y, p[1]);
        min_z = std::min(min_z, p[2]); max_z = std::max(max_z, p[2]);
    }

    double cx = (min_x + max_x) / 2;
    double cy = (min_y + max_y) / 2;
    double cz = (min_z + max_z) / 2;
    double extent = std::max({max_x - min_x, max_y - min_y, max_z - min_z});

    std::ostringstream scene_js;

    // Add unit box if requested (centered around data with padding)
    if (show_box) {
        scene_js << detail::generate_unit_box_js(
            (max_x - min_x) * 1.1,
            (max_y - min_y) * 1.1,
            (max_z - min_z) * 1.1,
            "#444444",
            cx, cy, cz
        );
    }

    // Add points as spheres
    scene_js << "const pointsGroup = new THREE.Group();\n";

    for (size_t i = 0; i < points.size(); ++i) {
        double z_norm = (max_z > min_z) ? (points[i][2] - min_z) / (max_z - min_z) : 0.5;
        std::string color = detail::interpolate_colormap(z_norm, detail::viridis_colors());

        scene_js << "{\n"
                 << "  const geo = new THREE.SphereGeometry(" << point_size << ", 16, 12);\n"
                 << "  const mat = new THREE.MeshPhongMaterial({ color: '" << color << "' });\n"
                 << "  const sphere = new THREE.Mesh(geo, mat);\n"
                 << "  sphere.position.set(" << points[i][0] << ", " << points[i][1] << ", " << points[i][2] << ");\n"
                 << "  pointsGroup.add(sphere);\n"
                 << "}\n";
    }

    scene_js << "scene.add(pointsGroup);\n";

    // Add lighting
    scene_js << R"(
        const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
        scene.add(ambientLight);
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(5, 10, 7);
        scene.add(directionalLight);
    )";

    webgl_threejs(scene_js.str(), "", background,
                  cx + extent * 1.5, cy + extent * 0.5, cz + extent * 1.5);
}

/**
 * Render a 3D line path.
 *
 * @param points Vector of {x, y, z} coordinates forming the path
 * @param color Line color
 * @param show_box Show bounding box
 * @param background Background color
 */
inline void line_3d(
    const std::vector<std::array<double, 3>>& points,
    const std::string& color = "#00d4ff",
    bool show_box = true,
    const std::string& background = "#1e1e2e"
) {
    if (points.size() < 2) return;

    // Calculate bounds
    double min_x = points[0][0], max_x = points[0][0];
    double min_y = points[0][1], max_y = points[0][1];
    double min_z = points[0][2], max_z = points[0][2];

    for (const auto& p : points) {
        min_x = std::min(min_x, p[0]); max_x = std::max(max_x, p[0]);
        min_y = std::min(min_y, p[1]); max_y = std::max(max_y, p[1]);
        min_z = std::min(min_z, p[2]); max_z = std::max(max_z, p[2]);
    }

    double cx = (min_x + max_x) / 2;
    double cy = (min_y + max_y) / 2;
    double cz = (min_z + max_z) / 2;
    double extent = std::max({max_x - min_x, max_y - min_y, max_z - min_z});

    std::ostringstream scene_js;

    // Add unit box if requested (centered around data with padding)
    if (show_box) {
        scene_js << detail::generate_unit_box_js(
            (max_x - min_x) * 1.1,
            (max_y - min_y) * 1.1,
            (max_z - min_z) * 1.1,
            "#444444",
            cx, cy, cz
        );
    }

    // Build vertex array
    scene_js << "{\n"
             << "  const lineGeometry = new THREE.BufferGeometry();\n"
             << "  const lineVertices = new Float32Array([";

    for (size_t i = 0; i < points.size(); ++i) {
        if (i > 0) scene_js << ", ";
        scene_js << points[i][0] << ", " << points[i][1] << ", " << points[i][2];
    }

    scene_js << "]);\n"
             << "  lineGeometry.setAttribute('position', new THREE.BufferAttribute(lineVertices, 3));\n"
             << "  const lineMaterial = new THREE.LineBasicMaterial({ color: '" << color << "' });\n"
             << "  const line = new THREE.Line(lineGeometry, lineMaterial);\n"
             << "  scene.add(line);\n"
             << "}\n";

    webgl_threejs(scene_js.str(), "", background,
                  cx + extent * 1.5, cy + extent * 0.5, cz + extent * 1.5);
}

/**
 * Render a molecular structure.
 *
 * @param atoms Vector of Atom structs
 * @param bonds Vector of Bond structs (optional)
 * @param atom_scale Scale factor for atom radii
 * @param bond_radius Bond cylinder radius
 * @param auto_rotate Enable auto-rotation
 * @param background Background color
 */
inline void molecule(
    const std::vector<Atom>& atoms,
    const std::vector<Bond>& bonds = {},
    double atom_scale = 0.4,
    double bond_radius = 0.1,
    bool auto_rotate = true,
    const std::string& background = "#1e1e2e"
) {
    if (atoms.empty()) return;

    // Calculate bounds
    double min_x = atoms[0].x, max_x = atoms[0].x;
    double min_y = atoms[0].y, max_y = atoms[0].y;
    double min_z = atoms[0].z, max_z = atoms[0].z;

    for (const auto& a : atoms) {
        min_x = std::min(min_x, a.x); max_x = std::max(max_x, a.x);
        min_y = std::min(min_y, a.y); max_y = std::max(max_y, a.y);
        min_z = std::min(min_z, a.z); max_z = std::max(max_z, a.z);
    }

    double cx = (min_x + max_x) / 2;
    double cy = (min_y + max_y) / 2;
    double cz = (min_z + max_z) / 2;
    double extent = std::max({max_x - min_x, max_y - min_y, max_z - min_z}) + 5;

    std::ostringstream scene_js;
    scene_js << "const moleculeGroup = new THREE.Group();\n";

    // Add atoms
    for (const auto& atom : atoms) {
        auto elem = detail::get_element(atom.element);
        double radius = elem.radius * atom_scale;

        scene_js << "{\n"
                 << "  const atomGeo = new THREE.SphereGeometry(" << radius << ", 32, 24);\n"
                 << "  const atomMat = new THREE.MeshPhongMaterial({ color: '" << elem.color
                 << "', shininess: 80, specular: 0x444444 });\n"
                 << "  const atom = new THREE.Mesh(atomGeo, atomMat);\n"
                 << "  atom.position.set(" << atom.x << ", " << atom.y << ", " << atom.z << ");\n"
                 << "  moleculeGroup.add(atom);\n"
                 << "}\n";
    }

    // Add bonds
    for (const auto& bond : bonds) {
        if (bond.atom1 >= atoms.size() || bond.atom2 >= atoms.size()) continue;

        const auto& a1 = atoms[bond.atom1];
        const auto& a2 = atoms[bond.atom2];

        double mx = (a1.x + a2.x) / 2;
        double my = (a1.y + a2.y) / 2;
        double mz = (a1.z + a2.z) / 2;

        double dx = a2.x - a1.x;
        double dy = a2.y - a1.y;
        double dz = a2.z - a1.z;
        double length = std::sqrt(dx*dx + dy*dy + dz*dz);

        if (length < 0.01) continue;

        dx /= length; dy /= length; dz /= length;

        scene_js << "{\n"
                 << "  const bondGeo = new THREE.CylinderGeometry(" << bond_radius << ", "
                 << bond_radius << ", " << length << ", 8);\n"
                 << "  const bondMat = new THREE.MeshPhongMaterial({ color: '#888888', shininess: 30 });\n"
                 << "  const bond = new THREE.Mesh(bondGeo, bondMat);\n"
                 << "  bond.position.set(" << mx << ", " << my << ", " << mz << ");\n"
                 << "  const direction = new THREE.Vector3(" << dx << ", " << dy << ", " << dz << ");\n"
                 << "  const axis = new THREE.Vector3(0, 1, 0);\n"
                 << "  bond.quaternion.setFromUnitVectors(axis, direction);\n"
                 << "  moleculeGroup.add(bond);\n"
                 << "}\n";
    }

    scene_js << "scene.add(moleculeGroup);\n";

    // Add lighting
    scene_js << R"(
        const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
        scene.add(ambientLight);
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(5, 10, 7);
        scene.add(directionalLight);
    )";

    std::string animate = auto_rotate ? "moleculeGroup.rotation.y += 0.002;" : "";

    webgl_threejs(scene_js.str(), animate, background,
                  cx + extent, cy + extent * 0.3, cz + extent);
}

} // namespace views
} // namespace cm

#endif // CM_VIEWS_HPP
