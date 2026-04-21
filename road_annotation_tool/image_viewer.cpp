#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <map>
#include <chrono>
#include <iomanip>
#include <ctime>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Hard-coded constants
// ---------------------------------------------------------------------------
const std::string WINDOW_NAME    = "Image Viewer";
const cv::Scalar  GREEN          = cv::Scalar(0, 255, 0);
const int         LINE_THICKNESS = 2;

// ---------------------------------------------------------------------------
// Config — populated by loadConfig()
// ---------------------------------------------------------------------------
struct Config {
    // Parameters read from config file
    float       scale_factor  = 1.0f;          // float  param: resize factor applied on load
    int         circle_radius = 10;            // int    param: radius of click circles

    // Fractional row positions [0.0, 1.0] for horizontal green lines.
    // Defaults match the previous hard-coded values; overridden by config.
    std::vector<double> line_rows = { 0.22, 0.35, 0.55 };

    // Fraction of the raw image width used as a movement step [0.0, 1.0].
    float move_step_pct = 0.01f;

    // Cropping fractions [0.0, 1.0].  crop_top is the fraction of image height
    // removed from the top; crop_bottom from the bottom.  Stored in dataset.info
    // for use by downstream tools.  Default = no cropping.
    double crop_top    = 0.1;
    double crop_bottom = 0.8;

    // Paths (also overridable in config)
    std::string image_dir     = "images";
    std::string output_file   = "labels.csv";
    std::string config_file   = "config.ini";
};

// ---------------------------------------------------------------------------
// INI parser: key = value, # comments, surrounding whitespace stripped
// ---------------------------------------------------------------------------
std::map<std::string, std::string> parseIni(const std::string& path)
{
    std::map<std::string, std::string> kv;
    std::ifstream f(path);
    if (!f.is_open()) return kv;

    auto trim = [](const std::string& s) -> std::string {
        size_t a = s.find_first_not_of(" \t\r\n");
        size_t b = s.find_last_not_of(" \t\r\n");
        return (a == std::string::npos) ? "" : s.substr(a, b - a + 1);
    };

    std::string line;
    while (std::getline(f, line)) {
        auto comment = line.find('#');
        if (comment != std::string::npos) line = line.substr(0, comment);
        line = trim(line);
        if (line.empty()) continue;

        auto eq = line.find('=');
        if (eq == std::string::npos) continue;

        std::string key = trim(line.substr(0, eq));
        std::string val = trim(line.substr(eq + 1));
        if (!key.empty()) kv[key] = val;
    }
    return kv;
}

bool loadConfig(Config& cfg, const std::string& path)
{
    auto kv = parseIni(path);
    if (kv.empty()) {
        std::cerr << "[CONFIG] Could not open or parse \"" << path
                  << "\" — using defaults.\n";
        return false;
    }

    try {
        if (kv.count("scale_factor"))  cfg.scale_factor  = std::stof(kv.at("scale_factor"));
        if (kv.count("circle_radius")) cfg.circle_radius = std::stoi(kv.at("circle_radius"));
        if (kv.count("image_dir"))     cfg.image_dir     = kv.at("image_dir");
        if (kv.count("crop_top"))      cfg.crop_top      = std::stod(kv.at("crop_top"));
        if (kv.count("crop_bottom"))   cfg.crop_bottom   = std::stod(kv.at("crop_bottom"));
        if (kv.count("move_step_pct")) {
            float v = std::stof(kv.at("move_step_pct"));
            if (v <= 0.0f || v > 1.0f)
                std::cerr << "[CONFIG] move_step_pct=" << v
                          << " is outside (0.0, 1.0] — keeping default.\n";
            else
                cfg.move_step_pct = v;
        }

        // line_rows = 0.1, 0.5, 0.9   (comma-separated floats in [0.0, 1.0])
        if (kv.count("line_rows")) {
            std::vector<double> parsed;
            std::istringstream ss(kv.at("line_rows"));
            std::string token;
            while (std::getline(ss, token, ',')) {
                // trim whitespace around each token
                size_t a = token.find_first_not_of(" \t");
                size_t b = token.find_last_not_of(" \t");
                if (a == std::string::npos) continue;
                double v = std::stod(token.substr(a, b - a + 1));
                if (v < 0.0 || v > 1.0) {
                    std::cerr << "[CONFIG] line_rows value " << v
                              << " is outside [0.0, 1.0] — skipped.\n";
                    continue;
                }
                parsed.push_back(v);
            }
            if (!parsed.empty())
                cfg.line_rows = std::move(parsed);
            else
                std::cerr << "[CONFIG] line_rows had no valid values — keeping defaults.\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "[CONFIG] Parse error: " << e.what() << "\n";
        return false;
    }

    std::cout << "[CONFIG] Loaded \"" << path << "\"\n"
              << "         scale_factor  = " << cfg.scale_factor  << "\n"
              << "         circle_radius = " << cfg.circle_radius << "\n"
              << "         move_step_pct = " << cfg.move_step_pct << "\n"
              << "         crop_top      = " << cfg.crop_top      << "\n"
              << "         crop_bottom   = " << cfg.crop_bottom   << "\n"
              << "         image_dir     = " << cfg.image_dir     << "\n"
              << "         output_file   = " << cfg.output_file   << "\n"
              << "         line_rows     = ";
    for (size_t i = 0; i < cfg.line_rows.size(); ++i)
        std::cout << (i ? ", " : "") << cfg.line_rows[i];
    std::cout << "\n";
    return true;
}

// ---------------------------------------------------------------------------
// Output file helpers — labels.csv format
//
// Each row: filename, road, row0_x0, row0_x1, ..., row1_x0, ..., rowN_...
//
// Columns are built from line_rows sorted furthest (largest fraction) first.
// For each line, point x-positions are expressed as a decimal fraction of
// image width [0.0, 1.0], sorted left-to-right.  Columns are fixed-width per
// line (one column per line) when only one point per line is expected, but
// multiple points on the same line are written as additional comma-separated
// columns.  A header is written once if the file is new/empty.
// ---------------------------------------------------------------------------

// Return line_rows indices sorted largest-fraction first (furthest row first).
std::vector<int> sortedLineIndices(const std::vector<double>& lineRows)
{
    std::vector<int> idx(lineRows.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b){
        return lineRows[a] > lineRows[b];
    });
    return idx;
}

// Collect points snapped to a given pixel row.
std::vector<double> pointsOnRow(const std::vector<cv::Point>& circles,
                                int rowPx, int imgWidth)
{
    std::vector<double> xs;
    for (const auto& p : circles)
        if (p.y == rowPx)
            xs.push_back(static_cast<double>(p.x) / imgWidth);
    std::sort(xs.begin(), xs.end());
    return xs;
}

// ---------------------------------------------------------------------------
// Write dataset.info — records the row fractions alongside labels.csv
// so downstream tools know what image-height percentage each loc_rN maps to.
// ---------------------------------------------------------------------------
void writeDatasetInfo(const std::string& imageDir,
                      const std::vector<double>& lineRows,
                      double cropTop,
                      double cropBottom)
{
    std::string infoPath = (fs::path(imageDir) / "dataset.info").string();

    auto idx = sortedLineIndices(lineRows);  // largest fraction first
    int n    = static_cast<int>(lineRows.size());

    std::ofstream f(infoPath, std::ios::trunc);
    if (!f.is_open()) {
        std::cerr << "[INFO]  Cannot write \"" << infoPath << "\"\n";
        return;
    }

    f << "# dataset.info — geometry metadata for this dataset\n";
    f << "#\n";
    f << "# Row naming convention:\n";
    f << "#   r0       = furthest from camera (smallest y fraction, top of image)\n";
    f << "#   r(N-1)   = closest to camera  (largest y fraction, bottom of image)\n";
    f << "#\n";
    f << "# Each loc_rN value is the fraction of image height at which\n";
    f << "# the horizontal label line sits (0.0 = top, 1.0 = bottom).\n";
    f << "# Columns in labels.csv are written closest-first (loc_r(N-1) ... loc_r0).\n";
    f << "#\n";
    f << "# crop_top and crop_bottom are fractions of image height that are\n";
    f << "# removed before the image is used for training (0.0 = no crop).\n";
    f << "#\n";

    // Row fractions: r0 (furthest) first
    for (int col = 0; col < n; ++col) {
        int rowIdx = idx[n - 1 - col];
        f << "loc_r" << col << " = "
          << std::fixed << std::setprecision(4) << lineRows[rowIdx] << "\n";
    }

    f << "crop_top    = " << std::fixed << std::setprecision(4) << cropTop    << "\n";
    f << "crop_bottom = " << std::fixed << std::setprecision(4) << cropBottom << "\n";

    std::cout << "[INFO]  dataset.info  : " << infoPath << "\n";
    for (int col = 0; col < n; ++col) {
        int rowIdx = idx[n - 1 - col];
        std::cout << "         loc_r" << col << " = "
                  << std::fixed << std::setprecision(4) << lineRows[rowIdx] << "\n";
    }
    std::cout << "         crop_top    = " << std::fixed << std::setprecision(4) << cropTop    << "\n";
    std::cout << "         crop_bottom = " << std::fixed << std::setprecision(4) << cropBottom << "\n";
}

void ensureOutputHeader(const std::string& path,
                        const std::vector<double>& lineRows)
{
    // Column naming convention:
    //   r0 = furthest from camera (smallest y fraction, top of image)
    //   r1 = middle row
    //   r(N-1) = closest to camera (largest y fraction, bottom of image)
    // Columns are written largest-fraction first in the CSV, so the order is:
    //   loc_r(N-1), loc_r(N-2), ..., loc_r0
    // Example with 3 rows: filename,road_present,loc_r2,loc_r1,loc_r0
    std::ostringstream hdr;
    hdr << "filename,road_present";
    int n = static_cast<int>(lineRows.size());
    for (int col = 0; col < n; ++col)
        hdr << ",loc_r" << (n - 1 - col);
    const std::string correctHeader = hdr.str();

    // Read existing file
    std::vector<std::string> lines;
    {
        std::ifstream f(path);
        if (f.is_open()) {
            std::string line;
            while (std::getline(f, line))
                if (!line.empty()) lines.push_back(line);
        }
    }

    if (lines.empty()) {
        // New file — just write the header
        std::ofstream f(path, std::ios::trunc);
        if (!f.is_open()) {
            std::cerr << "[OUTPUT] Cannot create \"" << path << "\".\n";
            return;
        }
        f << correctHeader << '\n';
        std::cout << "[OUTPUT] Created \"" << path << "\" with header:\n"
                  << "         " << correctHeader << "\n";
        return;
    }

    // File exists — check if the first line matches the correct header
    if (lines[0] == correctHeader) return;  // already correct

    // Header mismatch — update it and rewrite the file
    std::cout << "[OUTPUT] Updating header in \"" << path << "\":\n"
              << "         old: " << lines[0] << "\n"
              << "         new: " << correctHeader << "\n";
    lines[0] = correctHeader;
    std::ofstream f(path, std::ios::trunc);
    if (!f.is_open()) {
        std::cerr << "[OUTPUT] Cannot rewrite \"" << path << "\".\n";
        return;
    }
    for (const auto& line : lines)
        f << line << '\n';
}

void appendLabel(const std::string&            outputPath,
                 const std::string&            imagePath,
                 const cv::Mat&                img,
                 const std::vector<cv::Point>& circles,
                 const Config&                 cfg,
                 const std::vector<int>&       linePixelRows)
{
    const std::string fname = fs::path(imagePath).filename().string();
    int road = circles.empty() ? 0 : 1;

    // Build the new data line
    std::ostringstream newLine;
    newLine << fname << ',' << road;
    auto idx = sortedLineIndices(cfg.line_rows);
    for (int i : idx) {
        int rowPx = linePixelRows[i];
        auto xs   = pointsOnRow(circles, rowPx, img.cols);
        newLine << ',';
        for (size_t k = 0; k < xs.size(); ++k) {
            if (k) newLine << ';';
            newLine << std::fixed << std::setprecision(4) << xs[k];
        }
    }
    newLine << '\n';

    // Read existing file into lines
    std::vector<std::string> lines;
    {
        std::ifstream in(outputPath);
        if (in.is_open()) {
            std::string line;
            while (std::getline(in, line))
                if (!line.empty()) lines.push_back(line);
        }
    }

    // Check if a line for this filename already exists
    bool replaced = false;
    for (auto& line : lines) {
        // The filename is the first comma-separated field
        auto comma = line.find(',');
        std::string existingFname = (comma != std::string::npos)
                                    ? line.substr(0, comma) : line;
        if (existingFname == fname) {
            line     = newLine.str();
            // strip trailing newline for storage in vector
            if (!line.empty() && line.back() == '\n')
                line.pop_back();
            replaced = true;
            std::cout << "[OUTPUT] Updated existing entry for \""
                      << fname << "\"\n";
            break;
        }
    }

    if (!replaced) {
        std::string l = newLine.str();
        if (!l.empty() && l.back() == '\n') l.pop_back();
        lines.push_back(l);
        std::cout << "[OUTPUT] Appended new entry for \""
                  << fname << "\"\n";
    }

    // Write all lines back
    std::ofstream out(outputPath, std::ios::trunc);
    if (!out.is_open()) {
        std::cerr << "[OUTPUT] Cannot write \"" << outputPath << "\"\n";
        return;
    }
    for (const auto& line : lines)
        out << line << '\n';

    std::cout << "[OUTPUT] labels.csv  road=" << road
              << "  pts=" << circles.size() << "\n";
}

// ---------------------------------------------------------------------------
// Application state (shared with mouse callback)
// ---------------------------------------------------------------------------
struct AppState {
    cv::Mat               base;           // image + green lines
    cv::Mat               display;        // base + circles
    std::vector<cv::Point> circles;
    bool                  dirty        = false;
    int                   circleRadius = 10;

    // Absolute pixel rows of the green lines for the current image.
    // Populated by loadImage; used by onMouse to snap clicks.
    std::vector<int>      linePixelRows;

    // Index of the currently active (selected) circle, or -1 for none.
    // Active circle is drawn as an open (unfilled) circle.
    int                   activeIndex  = -1;

    // True while a drag is in progress on the active point.
    bool                  dragging     = false;

    // Pixel step derived from cfg.move_step_pct * image width.
    int                   moveStepPx   = 1;

    // Label overlaid on the image: "filename  (index / total)"
    std::string           imageLabel;

    // Window name — needed so onMouse can push frames during drag
    // without waiting for the main loop's waitKeyEx to return.
    std::string           windowName;
};

// ---------------------------------------------------------------------------
// Mouse callback
// ---------------------------------------------------------------------------
// Forward-declare redraw so onMouse can call it directly during drag.
void redraw(AppState& state);

void onMouse(int event, int x, int y, int /*flags*/, void* userdata)
{
    AppState* s = reinterpret_cast<AppState*>(userdata);
    const int activateTol = s->circleRadius + 4;

    // ---- Left-button down --------------------------------------------------
    if (event == cv::EVENT_LBUTTONDOWN) {
        // Hit-test: clicking on an existing point starts a drag and activates it
        for (int i = 0; i < static_cast<int>(s->circles.size()); ++i) {
            const cv::Point& p = s->circles[i];
            if (std::abs(x - p.x) <= activateTol &&
                std::abs(y - p.y) <= activateTol) {
                s->activeIndex = i;
                s->dragging    = true;
                s->dirty       = true;
                std::cout << "[DRAG]  Started on point " << i
                          << " at (" << p.x << ", " << p.y << ")\n";
                return;
            }
        }

        // No hit — snap y to nearest green line and place a new point
        int snappedY = y;
        if (!s->linePixelRows.empty()) {
            int bestRow  = s->linePixelRows[0];
            int bestDist = std::abs(y - bestRow);
            for (int row : s->linePixelRows) {
                int dist = std::abs(y - row);
                if (dist < bestDist) { bestDist = dist; bestRow = row; }
            }
            snappedY = bestRow;
        }
        s->activeIndex = static_cast<int>(s->circles.size());
        s->circles.emplace_back(x, snappedY);
        s->dirty = true;
        std::cout << "[MOUSE] Left-click at (" << x << ", " << y << ")"
                  << "  →  new point " << s->activeIndex
                  << " snapped to row " << snappedY << "\n";
        return;
    }

    // ---- Mouse move during drag: update x and push frame immediately -------
    if (event == cv::EVENT_MOUSEMOVE && s->dragging && s->activeIndex >= 0) {
        auto& pt = s->circles[s->activeIndex];
        pt.x = std::max(0, std::min(s->display.cols - 1, x));
        // Redraw and show immediately — don't wait for waitKeyEx to return
        redraw(*s);
        cv::imshow(s->windowName, s->display);
        return;
    }

    // ---- Left-button up: finalise drag -------------------------------------
    if (event == cv::EVENT_LBUTTONUP && s->dragging && s->activeIndex >= 0) {
        auto& pt    = s->circles[s->activeIndex];
        pt.x        = std::max(0, std::min(s->display.cols - 1, x));
        s->dragging = false;
        s->dirty    = true;
        std::cout << "[DRAG]  Released point " << s->activeIndex
                  << " at (" << pt.x << ", " << pt.y << ")\n";
        return;
    }

    // ---- Right-button down: toggle activate/deactivate (no drag) -----------
    if (event == cv::EVENT_RBUTTONDOWN) {
        for (int i = 0; i < static_cast<int>(s->circles.size()); ++i) {
            const cv::Point& p = s->circles[i];
            if (std::abs(x - p.x) <= activateTol &&
                std::abs(y - p.y) <= activateTol) {
                s->activeIndex = (s->activeIndex == i) ? -1 : i;
                s->dirty = true;
                std::cout << "[MOUSE] "
                          << (s->activeIndex == i ? "Activated" : "Deactivated")
                          << " point " << i
                          << " at (" << p.x << ", " << p.y << ")\n";
                return;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Drawing helpers
// ---------------------------------------------------------------------------
void redraw(AppState& state)
{
    state.base.copyTo(state.display);
    for (int i = 0; i < static_cast<int>(state.circles.size()); ++i) {
        const cv::Point& pt = state.circles[i];
        if (i == state.activeIndex)
            cv::circle(state.display, pt, state.circleRadius, GREEN, 2);
        else
            cv::circle(state.display, pt, state.circleRadius, GREEN, cv::FILLED);
    }

    // Overlay filename + index in the top-left corner
    if (!state.imageLabel.empty()) {
        const int   margin    = 8;
        const double fontScale = 1.2;
        const int   thickness = 1;
        const int   font      = cv::FONT_HERSHEY_SIMPLEX;
        // Dark shadow for readability over any background
        cv::putText(state.display, state.imageLabel,
                    cv::Point(margin + 1, margin + 30),
                    font, fontScale, cv::Scalar(0, 0, 0), thickness + 1);
        cv::putText(state.display, state.imageLabel,
                    cv::Point(margin, margin + 29),
                    font, fontScale, cv::Scalar(255, 255, 255), thickness);
    }
}

void drawLines(const cv::Mat& src, cv::Mat& dst,
               const std::vector<double>& lineRows,
               double cropTop    = 0.0,
               double cropBottom = 0.0)
{
    const cv::Scalar VIOLET(211, 0, 148);   // BGR violet
    src.copyTo(dst);
    int w = src.cols, h = src.rows;

    // Draw green label lines
    for (double frac : lineRows) {
        int row = static_cast<int>(frac * h);
        cv::line(dst, cv::Point(0, row), cv::Point(w - 1, row),
                 GREEN, LINE_THICKNESS);
    }

    // Draw violet crop lines
    if (cropTop > 0.0) {
        int row = static_cast<int>(cropTop * h);
        cv::line(dst, cv::Point(0, row), cv::Point(w - 1, row),
                 VIOLET, LINE_THICKNESS);
    }
    if (cropBottom > 0.0) {
        int row = static_cast<int>((1.0 - cropBottom) * h);
        cv::line(dst, cv::Point(0, row), cv::Point(w - 1, row),
                 VIOLET, LINE_THICKNESS);
    }
}

// ---------------------------------------------------------------------------
// Filesystem helpers
// ---------------------------------------------------------------------------
std::vector<std::string> collectPNGs(const std::string& dir)
{
    std::vector<std::string> paths;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        auto ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".png") paths.push_back(entry.path().string());
    }
    std::sort(paths.begin(), paths.end());
    return paths;
}

// ---------------------------------------------------------------------------
// Argument parsing helpers
// ---------------------------------------------------------------------------

// Print usage and exit
void printUsage(const char* prog)
{
    std::cout << "Usage: " << prog << " [options]\n\n"
              << "Options:\n"
              << "  -ds  <dataset>   Dataset name — images are loaded from\n"
              << "                   training_data/<dataset>/\n"
              << "  -cfg <file>      Path to config file  (default: config.ini)\n"
              << "  -h, --help       Show this help and exit\n\n"
              << "Examples:\n"
              << "  " << prog << " -ds cats\n"
              << "       Loads PNGs from  training_data/cats/\n"
              << "  " << prog << " -ds dogs -cfg my.ini\n";
}

// Parse argv into a string→string map of flag→value pairs.
// Boolean flags (no value) map to "true".
// Returns false if an unknown flag or a missing value is detected.
bool parseArgs(int argc, char* argv[],
               std::map<std::string, std::string>& out)
{
    for (int i = 1; i < argc; ++i) {
        std::string tok = argv[i];

        if (tok == "-h" || tok == "--help") {
            out["help"] = "true";
            return true;
        }

        if (tok[0] == '-') {
            // Expect a value for every flag we know about
            if (i + 1 >= argc) {
                std::cerr << "ERROR: flag \"" << tok
                          << "\" requires a value.\n";
                return false;
            }
            // Strip leading dash(es) for the key
            std::string key = tok.substr(tok.find_first_not_of('-'));
            out[key] = argv[++i];
        } else {
            std::cerr << "ERROR: unexpected argument \"" << tok << "\".\n";
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    // ---- Parse command-line arguments --------------------------------------
    std::map<std::string, std::string> args;
    if (!parseArgs(argc, argv, args)) {
        printUsage(argv[0]);
        return 1;
    }
    if (args.count("help")) {
        printUsage(argv[0]);
        return 0;
    }

    // ---- Load config -------------------------------------------------------
    Config cfg;
    std::string cfgPath = args.count("cfg") ? args.at("cfg") : cfg.config_file;
    loadConfig(cfg, cfgPath);

    // ---- Resolve image directory -------------------------------------------
    // -ds <dataset>  →  training_data/<dataset>/
    // Fallback: cfg.image_dir (which defaults to "images")
    if (args.count("ds")) {
        const std::string trainingRoot = "training_data";
        cfg.image_dir = (fs::path(trainingRoot) / args.at("ds")).string();
        std::cout << "[ARGS]  Dataset    : " << args.at("ds") << "\n"
                  << "[ARGS]  Image dir  : " << cfg.image_dir  << "\n";
    }

    // ---- Validate image directory ------------------------------------------
    if (!fs::exists(cfg.image_dir) || !fs::is_directory(cfg.image_dir)) {
        std::cerr << "ERROR: directory \"" << cfg.image_dir << "\" not found.\n";
        if (args.count("ds"))
            std::cerr << "       (looked for training_data/"
                      << args.at("ds") << "/)\n";
        return 1;
    }

    auto paths = collectPNGs(cfg.image_dir);
    if (paths.empty()) {
        std::cerr << "ERROR: no PNG files found in \"" << cfg.image_dir << "\".\n";
        return 1;
    }

    std::cout << "\nFound " << paths.size() << " PNG(s) in \""
              << cfg.image_dir << "\"\n";
    std::cout << "Controls:\n"
              << "  Left-click on empty  : place a new snapped circle (auto-activated)\n"
              << "  Left-click + drag    : grab a circle and drag it horizontally\n"
              << "  Right-click          : activate / deactivate a circle\n"
              << "  l                    : move active point left by move_step_pct\n"
              << "  r                    : move active point right by move_step_pct\n"
              << "  d                    : delete the active point\n"
              << "  n                    : advance to next image (no save)\n"
              << "  p                    : previous image (no label written)\n"
              << "  c                    : clear all circles (not carried to next image)\n"
              << "  s                    : write label for current image\n"
              << "  q                    : quit\n\n";

    // ---- Output file — always lives in the image directory -----------------
    // Take only the filename portion of cfg.output_file (so a config value
    // like "results.csv" or "/some/path/labels.csv" both resolve to
    // <image_dir>/labels.csv).
    cfg.output_file = (fs::path(cfg.image_dir)
                       / fs::path(cfg.output_file).filename()).string();
    std::cout << "[OUTPUT] Label file   : " << cfg.output_file << "\n";
    ensureOutputHeader(cfg.output_file, cfg.line_rows);
    writeDatasetInfo(cfg.image_dir, cfg.line_rows, cfg.crop_top, cfg.crop_bottom);

    // ---- Window & state ----------------------------------------------------
    cv::namedWindow(WINDOW_NAME, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
    AppState state;
    state.circleRadius = cfg.circle_radius;
    state.windowName   = WINDOW_NAME;
    cv::setMouseCallback(WINDOW_NAME, onMouse, &state);

    int    index = 0;
    cv::Mat rawImage;   // unscaled+unlined copy kept for record-writing

    auto loadImage = [&]() {
        cv::Mat raw = cv::imread(paths[index]);
        if (raw.empty()) {
            std::cerr << "WARNING: could not read " << paths[index] << "\n";
            return;
        }
        // Apply optional scale factor
        if (std::abs(cfg.scale_factor - 1.0f) > 1e-4f)
            cv::resize(raw, raw, cv::Size(), cfg.scale_factor, cfg.scale_factor);

        rawImage = raw.clone();

        // Carry over x-positions from the previous image, re-snapped to this
        // image's line rows.  'c' clears state.circles before loadImage runs.
        std::vector<cv::Point> prev = state.circles;

        state.circles.clear();
        state.activeIndex = -1;
        state.dragging    = false;

        // Pixel step = percentage of image width, minimum 1px
        state.moveStepPx = std::max(1,
            static_cast<int>(std::round(cfg.move_step_pct * rawImage.cols)));

        // Pre-compute pixel rows for snap logic (must match drawLines exactly)
        state.linePixelRows.clear();
        for (double frac : cfg.line_rows)
            state.linePixelRows.push_back(static_cast<int>(frac * rawImage.rows));

        // Restore carried-over points: keep x, re-snap y to closest new row
        for (const auto& p : prev) {
            // Find which new row this point belonged to (closest match)
            int snappedY  = state.linePixelRows[0];
            int bestDist  = std::abs(p.y - snappedY);
            for (int row : state.linePixelRows) {
                int dist = std::abs(p.y - row);
                if (dist < bestDist) { bestDist = dist; snappedY = row; }
            }
            // Clamp x to new image width
            int clampedX = std::max(0, std::min(rawImage.cols - 1, p.x));
            state.circles.emplace_back(clampedX, snappedY);
        }

        drawLines(rawImage, state.base, cfg.line_rows, cfg.crop_top, cfg.crop_bottom);

        // Set overlay text: filename and position in sequence
        state.imageLabel = fs::path(paths[index]).filename().string()
                           + "  (" + std::to_string(index + 1)
                           + " / " + std::to_string(paths.size()) + ")";

        redraw(state);
        state.dirty = false;
        std::cout << "[IMAGE] " << (index + 1) << "/" << paths.size()
                  << "  " << paths[index]
                  << "  (" << rawImage.cols << " x " << rawImage.rows << ")\n";
    };

    loadImage();

    // ---- Event loop --------------------------------------------------------
    while (true) {
        if (state.dirty) { redraw(state); state.dirty = false; }

        std::string title = WINDOW_NAME;
        title += (state.activeIndex >= 0)
               ? ("  [ACTIVE pt " + std::to_string(state.activeIndex) + "]")
               : "  [click point to activate]";
        cv::setWindowTitle(WINDOW_NAME, title);

        cv::imshow(WINDOW_NAME, state.display);

        int key = cv::waitKeyEx(16);
        if (key < 0) continue;

        // Mask to low byte for all ASCII key handling.
        // Skip extended keys (>255) — they are not used.
        int lkey = key & 0xFF;
        if (key > 255) continue;
        if (lkey >= 'A' && lkey <= 'Z') lkey += 32;

        switch (lkey) {
            case 'l':   // move active point left by moveStepPx
            case 'r':   // move active point right by moveStepPx
                if (state.activeIndex >= 0) {
                    auto& pt = state.circles[state.activeIndex];
                    if (lkey == 'l')
                        pt.x = std::max(0, pt.x - state.moveStepPx);
                    else
                        pt.x = std::min(rawImage.cols - 1, pt.x + state.moveStepPx);
                    state.dirty = true;
                    std::cout << "[MOVE]  pt=" << state.activeIndex
                              << "  dir=" << (lkey == 'l' ? "LEFT" : "RIGHT")
                              << "  step=" << state.moveStepPx
                              << "  pos=(" << pt.x << "," << pt.y << ")\n"
                              << std::flush;
                } else {
                    std::cout << "[MOVE]  no active point — click a circle first.\n";
                }
                break;

            case 'd':   // delete active point
                if (state.activeIndex >= 0) {
                    std::cout << "[ACTION] Deleted point " << state.activeIndex
                              << " at (" << state.circles[state.activeIndex].x
                              << "," << state.circles[state.activeIndex].y << ")\n";
                    state.circles.erase(state.circles.begin() + state.activeIndex);
                    state.activeIndex = -1;
                    state.dirty = true;
                } else {
                    std::cout << "[ACTION] No active point to delete.\n";
                }
                break;

            case 'n':
                // Advance to next image without saving
                index = (index + 1) % static_cast<int>(paths.size());
                loadImage();
                break;

            case 'p':
                index = (index - 1 + static_cast<int>(paths.size()))
                        % static_cast<int>(paths.size());
                loadImage();
                break;

            case 27:    // ESC — deactivate point, or quit
                if (state.activeIndex >= 0) {
                    state.activeIndex = -1;
                    state.dirty = true;
                    std::cout << "[ACTION] Point deactivated.\n";
                } else {
                    std::cout << "Exiting.\n";
                    cv::destroyAllWindows();
                    return 0;
                }
                break;

            case 'c':
                state.circles.clear();
                state.activeIndex = -1;
                redraw(state);
                std::cout << "[ACTION] Circles cleared.\n";
                break;

            case 's':
                // Write label for current image
                appendLabel(cfg.output_file, paths[index],
                            rawImage, state.circles, cfg, state.linePixelRows);
                break;

            case 'q':
                std::cout << "Exiting.\n";
                cv::destroyAllWindows();
                return 0;

            default: break;
        }
    }
}
