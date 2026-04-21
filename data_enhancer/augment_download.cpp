#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <random>
#include <iomanip>
#include <map>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Configuration — loaded from augment.ini, overridable by CLI
// ---------------------------------------------------------------------------
struct AugConfig {
    // Maximum horizontal shift as a fraction of image width [0.0, 1.0]
    double max_shift_frac   = 0.05;

    // Number of augmented versions to generate per source image.
    // Each version is randomly either a horizontal shift or a warp.
    int    num_augmentations = 5;

    // Inpaint radius used to fill void regions (pixels)
    int    inpaint_radius   = 7;

    // Horizontal flip threshold — set to -1.0 to disable (default: disabled)
    double flip_threshold   = -1.0;

    // Row-varying lateral warp — resolution-independent perspective shift.
    // Formula: colShiftFrac = sign * A * (exp(k * (lowestFrac - rowFrac)^n) - 1.0)
    //   A     : column shift fraction, drawn randomly from [0, warp_A_max] per version
    //   k     : steepness, drawn randomly from [0, warp_k_max] per version
    //   n     : power on the distance term — higher n = flatter near zero,
    //           steeper near the top row
    //   sign  : randomly +1 or -1 per version
    // Set warp_A_max = 0.0 to disable warping.
    double warp_A_max = 0.3;
    double warp_k_max = 8.0;
    double warp_n     = 2.0;

    // Linear warp — column shift grows linearly from 0 at lowestFrac to a
    // maximum at clampFrac (10% above highestFrac), then holds constant.
    // Formula: colShiftFrac = sign * linMax * (lowestFrac - rowFrac)
    //                                        / (lowestFrac - clampFrac)
    //   linear_warp_max : maximum shift fraction of image width, drawn
    //                     randomly from [0, linear_warp_max] per version.
    //   sign            : randomly +1 or -1 per version.
    // Set linear_warp_max = 0.0 to disable.
    double linear_warp_max = 0.15;

    // Show side-by-side visualisation window for each augmented pair
    bool   visualize        = false;

    // Source dataset directory (set via -ds on the command line)
    std::string dataset_dir;

    // Config file path
    std::string config_file = "augment.ini";
};

// ---------------------------------------------------------------------------
// INI parser (same pattern as image_viewer.cpp)
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

bool loadAugConfig(AugConfig& cfg, const std::string& path)
{
    auto kv = parseIni(path);
    if (kv.empty()) {
        std::cerr << "[CONFIG] Could not open \"" << path
                  << "\" — using defaults.\n";
        return false;
    }
    try {
        if (kv.count("max_shift_frac"))    cfg.max_shift_frac    = std::stod(kv.at("max_shift_frac"));
        if (kv.count("num_augmentations")) cfg.num_augmentations = std::stoi(kv.at("num_augmentations"));
        if (kv.count("inpaint_radius"))    cfg.inpaint_radius    = std::stoi(kv.at("inpaint_radius"));
        if (kv.count("flip_threshold"))    cfg.flip_threshold    = std::stod(kv.at("flip_threshold"));
        if (kv.count("warp_A_max"))          cfg.warp_A_max          = std::stod(kv.at("warp_A_max"));
        if (kv.count("warp_k_max"))          cfg.warp_k_max          = std::stod(kv.at("warp_k_max"));
        if (kv.count("warp_n"))              cfg.warp_n              = std::stod(kv.at("warp_n"));
        if (kv.count("linear_warp_max"))     cfg.linear_warp_max     = std::stod(kv.at("linear_warp_max"));
    } catch (const std::exception& e) {
        std::cerr << "[CONFIG] Parse error: " << e.what() << "\n";
        return false;
    }
    std::cout << "[CONFIG] Loaded \"" << path << "\"\n"
              << "         max_shift_frac    = " << cfg.max_shift_frac    << "\n"
              << "         num_augmentations = " << cfg.num_augmentations << "\n"
              << "         inpaint_radius    = " << cfg.inpaint_radius    << "\n"
              << "         flip_threshold    = " << cfg.flip_threshold    << "\n"
              << "         warp_A_max        = " << cfg.warp_A_max        << "\n"
              << "         warp_k_max        = " << cfg.warp_k_max        << "\n"
              << "         warp_n            = " << cfg.warp_n            << "\n"
              << "         linear_warp_max   = " << cfg.linear_warp_max   << "\n";
    return true;
}

// ---------------------------------------------------------------------------
// Label row — one entry in labels.csv
// ---------------------------------------------------------------------------
struct LabelRow {
    std::string filename;
    int         road = 0;
    // x fractions per column (column order matches the CSV header)
    // Each element is a semicolon-joined list of values, or "" if absent
    std::vector<std::string> cols;
};

// ---------------------------------------------------------------------------
// Parse a semicolon-separated list of doubles from a cell string
// ---------------------------------------------------------------------------
std::vector<double> parseCellValues(const std::string& cell)
{
    std::vector<double> vals;
    if (cell.empty()) return vals;
    std::istringstream ss(cell);
    std::string tok;
    while (std::getline(ss, tok, ';')) {
        try { vals.push_back(std::stod(tok)); } catch (...) {}
    }
    return vals;
}

// ---------------------------------------------------------------------------
// Format a vector of doubles back into a semicolon-separated cell string
// ---------------------------------------------------------------------------
std::string formatCell(const std::vector<double>& vals)
{
    std::ostringstream oss;
    for (size_t i = 0; i < vals.size(); ++i) {
        if (i) oss << ';';
        oss << std::fixed << std::setprecision(4) << vals[i];
    }
    return oss.str();
}

// ---------------------------------------------------------------------------
// Read labels.csv → header line + vector of LabelRows
// ---------------------------------------------------------------------------
bool readLabels(const std::string& path,
                std::string&             header,
                std::vector<LabelRow>&   rows)
{
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "[LABELS] Cannot open \"" << path << "\"\n";
        return false;
    }

    bool firstLine = true;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;

        if (firstLine) {
            header    = line;
            firstLine = false;
            continue;
        }

        std::istringstream ss(line);
        std::string tok;
        LabelRow row;

        // filename
        std::getline(ss, row.filename, ',');
        // road flag
        std::string roadStr;
        std::getline(ss, roadStr, ',');
        try { row.road = std::stoi(roadStr); } catch (...) {}
        // remaining columns (one per line_row)
        while (std::getline(ss, tok, ','))
            row.cols.push_back(tok);

        rows.push_back(row);
    }
    return true;
}

// ---------------------------------------------------------------------------
// Write labels.csv
// ---------------------------------------------------------------------------
void writeLabels(const std::string&            path,
                 const std::string&            header,
                 const std::vector<LabelRow>&  rows)
{
    std::ofstream f(path, std::ios::trunc);
    if (!f.is_open()) {
        std::cerr << "[LABELS] Cannot write \"" << path << "\"\n";
        return;
    }
    f << header << '\n';
    for (const auto& r : rows) {
        f << r.filename << ',' << r.road;
        for (const auto& c : r.cols)
            f << ',' << c;
        f << '\n';
    }
    std::cout << "[LABELS] Wrote " << rows.size()
              << " rows to \"" << path << "\"\n";
}

// ---------------------------------------------------------------------------
// Inpaint void regions.
// mask: 255 where pixels are void (came from outside the original image).
// ---------------------------------------------------------------------------
cv::Mat inpaintVoid(const cv::Mat& img, const cv::Mat& mask, int radius)
{
    cv::Mat result;
    cv::inpaint(img, mask, result, radius, cv::INPAINT_TELEA);
    return result;
}

// ---------------------------------------------------------------------------
// Apply a horizontal shift to an image.
// Returns the shifted image (inpainted) and the pixel shift applied.
// shiftFrac: positive = shift right, negative = shift left.
// ---------------------------------------------------------------------------
cv::Mat applyShift(const cv::Mat& src, double shiftFrac, int inpaintRadius)
{
    int shiftPx = static_cast<int>(std::round(shiftFrac * src.cols));

    // Translation matrix
    cv::Mat M = (cv::Mat_<double>(2, 3) <<
        1, 0, static_cast<double>(shiftPx),
        0, 1, 0.0);

    cv::Mat shifted;
    cv::warpAffine(src, shifted, M, src.size(),
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT,
                   cv::Scalar(0, 0, 0));

    // Build void mask
    cv::Mat mask = cv::Mat::zeros(src.size(), CV_8UC1);
    if (shiftPx > 0)
        mask(cv::Rect(0, 0, shiftPx, src.rows)) = 255;
    else if (shiftPx < 0)
        mask(cv::Rect(src.cols + shiftPx, 0, -shiftPx, src.rows)) = 255;

    if (cv::countNonZero(mask) == 0) return shifted;
    return inpaintVoid(shifted, mask, inpaintRadius);
}

// ---------------------------------------------------------------------------
// Apply a horizontal flip to an image (no void regions — no inpainting needed)
// ---------------------------------------------------------------------------
cv::Mat applyFlip(const cv::Mat& src)
{
    cv::Mat flipped;
    cv::flip(src, flipped, 1);   // 1 = horizontal
    return flipped;
}

// ---------------------------------------------------------------------------
// Transform a full LabelRow for a horizontal flip.
// Each x-fraction is reflected: new_x = 1.0 - old_x
// Point order within each cell is reversed so values stay sorted left-to-right.
// ---------------------------------------------------------------------------
LabelRow transformLabel_flip(const LabelRow&                  src,
                             const std::string&               newFilename,
                             const std::vector<std::string>&  headerCols)
{
    LabelRow out;
    out.filename = newFilename;
    out.road     = src.road;
    out.cols.resize(src.cols.size());

    for (size_t ci = 0; ci < src.cols.size(); ++ci) {
        auto vals = parseCellValues(src.cols[ci]);
        if (vals.empty()) { out.cols[ci] = ""; continue; }

        std::vector<double> flipped;
        for (double x : vals)
            flipped.push_back(1.0 - x);
        // Re-sort left-to-right after reflection
        std::sort(flipped.begin(), flipped.end());
        out.cols[ci] = formatCell(flipped);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Row-varying lateral warp — resolution-independent perspective shift.
//
// colShiftFrac = sign * A * (exp(k * (lowestFrac - rowFrac)^n) - 1.0)
//
// Three zones:
//   rowFrac >= lowestFrac              : colShiftFrac = 0  (below bottom label row)
//   clampFrac < rowFrac < lowestFrac   : formula above     (active warp zone,
//                                         extends 10% of image height above the
//                                         top label row, i.e. clampFrac = highestFrac - 0.10)
//   rowFrac <= clampFrac               : clamped at clampFrac value
// ---------------------------------------------------------------------------
cv::Mat applyPerspectiveWarp(const cv::Mat& src,
                              double lowestFrac,
                              double highestFrac,
                              double A,
                              double k,
                              double n,
                              double sign,
                              int    inpaintRadius)
{
    int W = src.cols, H = src.rows;

    // Warp continues 10% of image height above the top label row
    double clampFrac = std::max(0.0, highestFrac - 0.10);

    auto warpShift = [&](double x) {
        return sign * A * (std::exp(k * std::pow(x, n)) - 1.0);
    };

    double colShiftFracTop = warpShift(lowestFrac - clampFrac);

    cv::Mat map1(H, W, CV_32FC1);
    cv::Mat map2(H, W, CV_32FC1);
    cv::Mat voidMask = cv::Mat::zeros(H, W, CV_8UC1);

    for (int y = 0; y < H; ++y) {
        double rowFrac = static_cast<double>(y) / H;

        double colShiftFrac;
        if (rowFrac >= lowestFrac)
            colShiftFrac = 0.0;
        else if (rowFrac <= clampFrac)
            colShiftFrac = colShiftFracTop;
        else
            colShiftFrac = warpShift(lowestFrac - rowFrac);

        double colShiftPx = colShiftFrac * W;

        for (int x = 0; x < W; ++x) {
            float srcX = static_cast<float>(x - colShiftPx);
            map1.at<float>(y, x) = srcX;
            map2.at<float>(y, x) = static_cast<float>(y);
            if (srcX < 0.0f || srcX >= static_cast<float>(W))
                voidMask.at<uchar>(y, x) = 255;
        }
    }

    cv::Mat warped;
    cv::remap(src, warped, map1, map2, cv::INTER_LINEAR,
              cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    if (cv::countNonZero(voidMask) == 0) return warped;
    return inpaintVoid(warped, voidMask, inpaintRadius);
}

// ---------------------------------------------------------------------------
// Transform a LabelRow for the perspective warp.
// Uses the same formula as applyPerspectiveWarp.
// ---------------------------------------------------------------------------
LabelRow transformLabel_warp(const LabelRow&                  src,
                              const std::string&               newFilename,
                              const std::vector<std::string>&  headerCols,
                              double                           lowestFrac,
                              double                           highestFrac,
                              double                           A,
                              double                           k,
                              double                           n,
                              double                           sign)
{
    LabelRow out;
    out.filename = newFilename;
    out.road     = src.road;
    out.cols.resize(src.cols.size());

    auto warpShift = [&](double x) {
        return sign * A * (std::exp(k * std::pow(x, n)) - 1.0);
    };
    double clampFrac       = std::max(0.0, highestFrac - 0.10);
    double colShiftFracTop = warpShift(lowestFrac - clampFrac);

    for (size_t ci = 0; ci < src.cols.size(); ++ci) {
        auto vals = parseCellValues(src.cols[ci]);
        if (vals.empty()) { out.cols[ci] = ""; continue; }

        double rowFrac = 0.5;
        if (ci < headerCols.size()) {
            const std::string& colName = headerCols[ci];
            auto rowPos = colName.find("row");
            auto xPos   = colName.find("_x");
            if (rowPos != std::string::npos && xPos != std::string::npos) {
                try { rowFrac = std::stod(colName.substr(rowPos + 3, xPos - rowPos - 3)); }
                catch (...) {}
            }
        }

        double colShiftFrac;
        if (rowFrac >= lowestFrac)
            colShiftFrac = 0.0;
        else if (rowFrac <= clampFrac)
            colShiftFrac = colShiftFracTop;
        else
            colShiftFrac = warpShift(lowestFrac - rowFrac);

        std::vector<double> newVals;
        for (double x : vals)
            newVals.push_back(std::max(0.0, std::min(1.0, x + colShiftFrac)));
        out.cols[ci] = formatCell(newVals);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Linear warp — column shift grows linearly from 0 at lowestFrac up to
// linMax (as a fraction of image width) at clampFrac, then holds constant.
//
//   clampFrac     = max(0, highestFrac - 0.10)   (10% above top label row)
//   t(rowFrac)    = (lowestFrac - rowFrac) / (lowestFrac - clampFrac)  [0,1]
//   colShiftFrac  = sign * linMax * t            for clampFrac < rowFrac < lowestFrac
//   colShiftFrac  = sign * linMax                for rowFrac <= clampFrac
//   colShiftFrac  = 0                            for rowFrac >= lowestFrac
// ---------------------------------------------------------------------------
cv::Mat applyLinearWarp(const cv::Mat& src,
                         double lowestFrac,
                         double highestFrac,
                         double linMax,
                         double sign,
                         int    inpaintRadius)
{
    int W = src.cols, H = src.rows;
    double clampFrac = std::max(0.0, highestFrac - 0.10);
    double span      = lowestFrac - clampFrac;

    cv::Mat map1(H, W, CV_32FC1);
    cv::Mat map2(H, W, CV_32FC1);
    cv::Mat voidMask = cv::Mat::zeros(H, W, CV_8UC1);

    for (int y = 0; y < H; ++y) {
        double rowFrac = static_cast<double>(y) / H;

        double colShiftFrac;
        if (rowFrac >= lowestFrac)
            colShiftFrac = 0.0;
        else if (rowFrac <= clampFrac)
            colShiftFrac = sign * linMax;
        else
            colShiftFrac = sign * linMax * (lowestFrac - rowFrac) / span;

        double colShiftPx = colShiftFrac * W;

        for (int x = 0; x < W; ++x) {
            float srcX = static_cast<float>(x - colShiftPx);
            map1.at<float>(y, x) = srcX;
            map2.at<float>(y, x) = static_cast<float>(y);
            if (srcX < 0.0f || srcX >= static_cast<float>(W))
                voidMask.at<uchar>(y, x) = 255;
        }
    }

    cv::Mat warped;
    cv::remap(src, warped, map1, map2, cv::INTER_LINEAR,
              cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    if (cv::countNonZero(voidMask) == 0) return warped;
    return inpaintVoid(warped, voidMask, inpaintRadius);
}

// ---------------------------------------------------------------------------
// Transform a LabelRow for the linear warp.
// ---------------------------------------------------------------------------
LabelRow transformLabel_linearWarp(const LabelRow&                  src,
                                    const std::string&               newFilename,
                                    const std::vector<std::string>&  headerCols,
                                    double                           lowestFrac,
                                    double                           highestFrac,
                                    double                           linMax,
                                    double                           sign)
{
    LabelRow out;
    out.filename = newFilename;
    out.road     = src.road;
    out.cols.resize(src.cols.size());

    double clampFrac = std::max(0.0, highestFrac - 0.10);
    double span      = lowestFrac - clampFrac;

    for (size_t ci = 0; ci < src.cols.size(); ++ci) {
        auto vals = parseCellValues(src.cols[ci]);
        if (vals.empty()) { out.cols[ci] = ""; continue; }

        double rowFrac = 0.5;
        if (ci < headerCols.size()) {
            const std::string& colName = headerCols[ci];
            auto rowPos = colName.find("row");
            auto xPos   = colName.find("_x");
            if (rowPos != std::string::npos && xPos != std::string::npos) {
                try { rowFrac = std::stod(colName.substr(rowPos + 3, xPos - rowPos - 3)); }
                catch (...) {}
            }
        }

        double colShiftFrac;
        if (rowFrac >= lowestFrac)
            colShiftFrac = 0.0;
        else if (rowFrac <= clampFrac)
            colShiftFrac = sign * linMax;
        else
            colShiftFrac = sign * linMax * (lowestFrac - rowFrac) / span;

        std::vector<double> newVals;
        for (double x : vals)
            newVals.push_back(std::max(0.0, std::min(1.0, x + colShiftFrac)));
        out.cols[ci] = formatCell(newVals);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Draw linear warp debug overlay — yellow centre line + yellow linear curve.
// ---------------------------------------------------------------------------
cv::Mat drawLinearWarpDebug(const cv::Mat& img,
                             double lowestFrac,
                             double highestFrac,
                             double linMax,
                             double sign)
{
    const cv::Scalar YELLOW(0, 255, 255);
    cv::Mat out = img.clone();
    int W = out.cols, H = out.rows;
    int cx = W / 2;

    double clampFrac = std::max(0.0, highestFrac - 0.10);
    double span      = lowestFrac - clampFrac;

    cv::line(out, cv::Point(cx, 0), cv::Point(cx, H - 1), YELLOW, 1);

    cv::Point prev(-1, -1);
    for (int y = 0; y < H; ++y) {
        double rowFrac = static_cast<double>(y) / H;
        double colShiftFrac;
        if (rowFrac >= lowestFrac)
            colShiftFrac = 0.0;
        else if (rowFrac <= clampFrac)
            colShiftFrac = sign * linMax;
        else
            colShiftFrac = sign * linMax * (lowestFrac - rowFrac) / span;

        int px = cv::saturate_cast<int>(cx + colShiftFrac * W);
        px = std::max(0, std::min(W - 1, px));
        cv::Point cur(px, y);
        if (prev.x >= 0) cv::line(out, prev, cur, YELLOW, 2);
        prev = cur;
    }
    return out;
}
double transformX_shift(double x, double shiftFrac)
{
    return std::max(0.0, std::min(1.0, x + shiftFrac));
}

// ---------------------------------------------------------------------------
// Transform a full LabelRow given a horizontal shift perturbation.
// ---------------------------------------------------------------------------
LabelRow transformLabel(const LabelRow&                  src,
                        const std::string&               newFilename,
                        const std::vector<std::string>&  headerCols,
                        double                           shiftFrac)
{
    LabelRow out;
    out.filename = newFilename;
    out.road     = src.road;
    out.cols.resize(src.cols.size());

    for (size_t ci = 0; ci < src.cols.size(); ++ci) {
        auto vals = parseCellValues(src.cols[ci]);
        if (vals.empty()) { out.cols[ci] = ""; continue; }

        std::vector<double> newVals;
        for (double x : vals)
            newVals.push_back(transformX_shift(x, shiftFrac));
        out.cols[ci] = formatCell(newVals);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Parse header columns after "filename,road"
// ---------------------------------------------------------------------------
std::vector<std::string> parseHeaderCols(const std::string& header)
{
    std::vector<std::string> cols;
    std::istringstream ss(header);
    std::string tok;
    int idx = 0;
    while (std::getline(ss, tok, ',')) {
        if (idx >= 2) cols.push_back(tok);
        ++idx;
    }
    return cols;
}

// ---------------------------------------------------------------------------
// Collect sorted PNG paths from a directory
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
// Draw horizontal lines and point markers onto a copy of an image,
// matching the style used in image_viewer.cpp.
//
// headerCols: column names after "filename,road" (e.g. "row0.75_x")
// label:      the LabelRow for this image
// ---------------------------------------------------------------------------
cv::Mat drawAnnotated(const cv::Mat&                  img,
                      const std::vector<std::string>& headerCols,
                      const LabelRow&                 label,
                      const std::string&              title = "")
{
    const cv::Scalar GREEN(0, 255, 0);
    const cv::Scalar WHITE(255, 255, 255);
    const int        LINE_THICKNESS = 2;
    const int        CIRCLE_RADIUS  = 10;

    cv::Mat out = img.clone();

    for (size_t ci = 0; ci < headerCols.size(); ++ci) {
        // Extract row fraction from header column name "row0.75_x" → 0.75
        const std::string& colName = headerCols[ci];
        double rowFrac = 0.5;
        auto rowPos = colName.find("row");
        auto xPos   = colName.find("_x");
        if (rowPos != std::string::npos && xPos != std::string::npos) {
            try {
                rowFrac = std::stod(
                    colName.substr(rowPos + 3, xPos - rowPos - 3));
            } catch (...) {}
        }

        int rowPx = static_cast<int>(rowFrac * img.rows);

        // Draw horizontal green line
        cv::line(out, cv::Point(0, rowPx),
                 cv::Point(img.cols - 1, rowPx), GREEN, LINE_THICKNESS);

        // Draw point markers if this image has them
        if (ci < label.cols.size()) {
            auto vals = parseCellValues(label.cols[ci]);
            for (double xFrac : vals) {
                int px = static_cast<int>(xFrac * img.cols);
                cv::circle(out, cv::Point(px, rowPx),
                           CIRCLE_RADIUS, GREEN, cv::FILLED);
            }
        }
    }

    // Overlay title text
    if (!title.empty()) {
        cv::putText(out, title, cv::Point(10, 28),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,0,0), 3);
        cv::putText(out, title, cv::Point(10, 28),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 1);
    }

    return out;
}

// ---------------------------------------------------------------------------
// Draw warp debug overlay — yellow centre line + yellow warp curve.
// ---------------------------------------------------------------------------
cv::Mat drawWarpDebug(const cv::Mat& img,
                      double lowestFrac,
                      double highestFrac,
                      double A,
                      double k,
                      double n,
                      double sign)
{
    const cv::Scalar YELLOW(0, 255, 255);
    cv::Mat out = img.clone();
    int W = out.cols, H = out.rows;
    int cx = W / 2;

    auto warpShift = [&](double x) {
        return sign * A * (std::exp(k * std::pow(x, n)) - 1.0);
    };
    double clampFrac       = std::max(0.0, highestFrac - 0.10);
    double colShiftFracTop = warpShift(lowestFrac - clampFrac);

    cv::line(out, cv::Point(cx, 0), cv::Point(cx, H - 1), YELLOW, 1);

    cv::Point prev(-1, -1);
    for (int y = 0; y < H; ++y) {
        double rowFrac = static_cast<double>(y) / H;
        double colShiftFrac;
        if (rowFrac >= lowestFrac)
            colShiftFrac = 0.0;
        else if (rowFrac <= clampFrac)
            colShiftFrac = colShiftFracTop;
        else
            colShiftFrac = warpShift(lowestFrac - rowFrac);

        int px = cv::saturate_cast<int>(cx + colShiftFrac * W);
        px = std::max(0, std::min(W - 1, px));
        cv::Point cur(px, y);
        if (prev.x >= 0) cv::line(out, prev, cur, YELLOW, 2);
        prev = cur;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Display two annotated images side by side in a single window.
// Returns false if the user pressed 'q' or ESC to abort, true to continue.
// Any other key (or the window timeout) advances to the next pair.
// ---------------------------------------------------------------------------
bool showSideBySide(const cv::Mat& left,  const std::string& leftTitle,
                    const cv::Mat& right, const std::string& rightTitle,
                    const std::vector<std::string>& headerCols,
                    const LabelRow& leftLabel,
                    const LabelRow& rightLabel)
{
    cv::Mat leftAnn  = drawAnnotated(left,  headerCols, leftLabel,  leftTitle);
    cv::Mat rightAnn = drawAnnotated(right, headerCols, rightLabel, rightTitle);

    // Resize right to match left height if dimensions differ
    if (leftAnn.rows != rightAnn.rows) {
        double scale = static_cast<double>(leftAnn.rows) / rightAnn.rows;
        cv::resize(rightAnn, rightAnn, cv::Size(), scale, scale);
    }

    // Draw a thin separator between the two panels
    cv::Mat sep(leftAnn.rows, 4, CV_8UC3, cv::Scalar(80, 80, 80));

    cv::Mat canvas;
    cv::hconcat(std::vector<cv::Mat>{leftAnn, sep, rightAnn}, canvas);

    const std::string WIN = "Augmentation Preview  |  any key = next  |  q = quit";
    cv::namedWindow(WIN, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
    cv::imshow(WIN, canvas);

    int key = cv::waitKeyEx(0);   // wait indefinitely for a keypress
    int k8  = key & 0xFF;
    if (k8 == 'q' || k8 == 27) {
        cv::destroyAllWindows();
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    AugConfig cfg;

    // ---- Parse command-line arguments --------------------------------------
    // Usage: ./augment -ds <dataset_dir> [-cfg <config>] [--visualize]
    if (argc < 3) {
        std::cout << "Usage: " << argv[0]
                  << " -ds <dataset_dir> [options]\n\n"
                  << "Options:\n"
                  << "  -ds  <path>     Full path to dataset directory (required)\n"
                  << "  -cfg <file>     Config file path (default: augment.ini)\n"
                  << "  --visualize     Show side-by-side preview for each pair\n\n"
                  << "Config file parameters (augment.ini):\n"
                  << "  max_shift_frac  Max horizontal shift as fraction of width "
                     "(default: 0.05)\n"
                  << "  max_angle_deg   Max rotation angle in degrees "
                     "(default: 5.0)\n"
                  << "  max_copies      Max augmented copies per image — actual\n"
                  << "                  count is random in [1, max_copies] "
                     "(default: 3)\n"
                  << "  inpaint_radius  Inpaint fill radius in pixels "
                     "(default: 7)\n\n"
                  << "Example:\n"
                  << "  " << argv[0]
                  << " -ds /data/training_data/road1 --visualize\n";
        return 1;
    }

    for (int i = 1; i < argc; ++i) {
        std::string flag = argv[i];
        if ((flag == "-ds" || flag == "--ds") && i + 1 < argc)
            cfg.dataset_dir = argv[++i];
        else if ((flag == "-cfg" || flag == "--cfg") && i + 1 < argc)
            cfg.config_file = argv[++i];
        else if (flag == "--visualize")
            cfg.visualize = true;
        else if (flag[0] == '-') {
            std::cerr << "Unknown flag: " << flag << "\n";
            return 1;
        }
    }

    if (cfg.dataset_dir.empty()) {
        std::cerr << "ERROR: -ds <dataset_dir> is required.\n";
        return 1;
    }

    // ---- Load config file --------------------------------------------------
    loadAugConfig(cfg, cfg.config_file);

    // ---- Validate dataset directory ----------------------------------------
    if (!fs::exists(cfg.dataset_dir) || !fs::is_directory(cfg.dataset_dir)) {
        std::cerr << "ERROR: directory \"" << cfg.dataset_dir
                  << "\" not found.\n";
        return 1;
    }

    // ---- Read labels.csv ---------------------------------------------------
    std::string labelsPath = (fs::path(cfg.dataset_dir) / "labels.csv").string();
    std::string header;
    std::vector<LabelRow> sourceLabels;
    if (!readLabels(labelsPath, header, sourceLabels)) return 1;

    std::cout << "\n[INFO]  Dataset            : " << cfg.dataset_dir        << "\n"
              << "[INFO]  Source images      : " << sourceLabels.size()      << "\n"
              << "[INFO]  Augmentations/image: " << cfg.num_augmentations    << "\n"
              << "[INFO]  Max shift          : ±" << cfg.max_shift_frac*100  << "% of width\n"
              << "[INFO]  Warp A_max         : " << cfg.warp_A_max           << "\n"
              << "[INFO]  Warp k_max         : " << cfg.warp_k_max           << "\n"
              << "[INFO]  Warp n             : " << cfg.warp_n               << "\n"
              << "[INFO]  Linear warp max    : " << cfg.linear_warp_max      << "\n"
              << "[INFO]  Flip threshold     : " << cfg.flip_threshold
              << (cfg.flip_threshold < 0 ? "  (disabled)" : "") << "\n"
              << "[INFO]  Inpaint radius     : " << cfg.inpaint_radius       << "px\n"
              << "[INFO]  Visualize          : " << (cfg.visualize ? "yes" : "no")
              << "\n\n";

    // ---- Build filename → LabelRow map ------------------------------------
    std::map<std::string, LabelRow> labelMap;
    for (const auto& r : sourceLabels)
        labelMap[r.filename] = r;

    auto headerCols = parseHeaderCols(header);

    // ---- Output goes into the same directory as the source images -----------
    fs::path outDir = fs::path(cfg.dataset_dir);
    std::cout << "[INFO]  Output dir    : " << outDir << " (same as source)\n\n";

    // ---- Set up RNG --------------------------------------------------------
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> shiftDist(-cfg.max_shift_frac, cfg.max_shift_frac);
    std::uniform_real_distribution<double> warpADist(0.0, cfg.warp_A_max);
    std::uniform_real_distribution<double> warpKDist(0.0, cfg.warp_k_max);
    std::uniform_real_distribution<double> linearWarpDist(0.0, cfg.linear_warp_max);
    std::uniform_int_distribution<int>     signDist(0, 1);
    std::uniform_int_distribution<int>     typeDist(0, 1);  // 0=exp warp, 1=linear warp

    // ---- Collect source PNGs -----------------------------------------------
    auto pngPaths = collectPNGs(cfg.dataset_dir);
    if (pngPaths.empty()) {
        std::cerr << "ERROR: no PNG files found in \"" << cfg.dataset_dir
                  << "\"\n";
        return 1;
    }

    // ---- Augmentation loop -------------------------------------------------
    std::vector<LabelRow> outLabels;
    int  generated   = 0;
    bool aborted     = false;

    for (const auto& pngPath : pngPaths) {
        if (aborted) break;

        std::string fname = fs::path(pngPath).filename().string();
        std::string stem  = fs::path(pngPath).stem().string();

        if (labelMap.find(fname) == labelMap.end()) {
            std::cout << "[SKIP]  No label for " << fname << "\n";
            continue;
        }
        const LabelRow& srcLabel = labelMap[fname];

        cv::Mat src = cv::imread(pngPath);
        if (src.empty()) {
            std::cerr << "[WARN]  Could not read " << pngPath << "\n";
            continue;
        }

        std::cout << "[PROC]  " << fname
                  << "  (" << src.cols << "x" << src.rows << ")"
                  << "  N=" << cfg.num_augmentations << "\n";

        // ---- Determine lowest and highest labelled row fractions ------------
        double lowestFrac  = 0.75;
        double highestFrac = 0.25;

        auto extractFrac = [](const std::string& colName) -> double {
            auto rowPos = colName.find("row");
            auto xPos   = colName.find("_x");
            if (rowPos == std::string::npos || xPos == std::string::npos) return -1.0;
            try { return std::stod(colName.substr(rowPos + 3, xPos - rowPos - 3)); }
            catch (...) { return -1.0; }
        };

        if (!headerCols.empty()) {
            double f = extractFrac(headerCols.front());
            if (f >= 0.0) lowestFrac = f;
        }
        if (headerCols.size() > 1) {
            double f = extractFrac(headerCols.back());
            if (f >= 0.0) highestFrac = f;
        } else {
            highestFrac = lowestFrac;
        }
        if (lowestFrac < highestFrac) std::swap(lowestFrac, highestFrac);

        // ---- Generate N augmented versions ----------------------------------
        for (int v = 0; v < cfg.num_augmentations && !aborted; ++v) {
            // Every version applies a random shift followed by a randomly
            // chosen warp type (exponential or linear), all with independent
            // random parameters and sign.
            double shift     = shiftDist(rng);
            double warpSign  = (signDist(rng) == 0) ? 1.0 : -1.0;
            bool   doLinear  = (typeDist(rng) == 1);

            // Parameters declared here so the visualisation block can use them
            double linMax = 0.0, A = 0.0, k = 0.0;

            // Step 1 — horizontal shift
            cv::Mat shifted       = applyShift(src, shift, cfg.inpaint_radius);
            LabelRow shiftedLabel = transformLabel(srcLabel, "", headerCols, shift);

            // Step 2 — warp (exp or linear) on the shifted image
            cv::Mat augImg;
            LabelRow augLabel;
            std::string augFname;
            std::ostringstream d;
            d << "SHIFT" << (shift >= 0 ? "+" : "")
              << std::fixed << std::setprecision(2) << shift * 100.0 << "%";

            if (doLinear) {
                linMax   = linearWarpDist(rng);
                augImg   = applyLinearWarp(shifted, lowestFrac, highestFrac,
                                            linMax, warpSign, cfg.inpaint_radius);
                augFname = stem + "_sl" + std::to_string(v) + ".png";
                shiftedLabel.filename = augFname;
                augLabel = transformLabel_linearWarp(
                    shiftedLabel, augFname, headerCols,
                    lowestFrac, highestFrac, linMax, warpSign);
                d << "  LINWARP" << (warpSign > 0 ? "+" : "-")
                  << " M=" << std::setprecision(3) << linMax;
            } else {
                A        = warpADist(rng);
                k        = warpKDist(rng);
                augImg   = applyPerspectiveWarp(shifted, lowestFrac, highestFrac,
                                                 A, k, cfg.warp_n, warpSign,
                                                 cfg.inpaint_radius);
                augFname = stem + "_sw" + std::to_string(v) + ".png";
                shiftedLabel.filename = augFname;
                augLabel = transformLabel_warp(
                    shiftedLabel, augFname, headerCols,
                    lowestFrac, highestFrac, A, k, cfg.warp_n, warpSign);
                d << "  EXPWARP" << (warpSign > 0 ? "+" : "-")
                  << " A=" << std::setprecision(3) << A
                  << " k=" << std::setprecision(2) << k;
            }

            std::string descStr = d.str();

            std::string augPath = (outDir / augFname).string();
            if (!cv::imwrite(augPath, augImg)) {
                std::cerr << "[WARN]  Failed to write " << augPath << "\n";
                continue;
            }

            outLabels.push_back(augLabel);
            ++generated;
            std::cout << "         [" << v << "] " << descStr
                      << "  → " << augFname << "\n";

            // ---- Visualisation ---------------------------------------------
            if (cfg.visualize) {
                cv::Mat leftPanel = drawAnnotated(src, headerCols, srcLabel, fname);
                if (doLinear)
                    leftPanel = drawLinearWarpDebug(leftPanel, lowestFrac, highestFrac,
                                                    linMax, warpSign);
                else
                    leftPanel = drawWarpDebug(leftPanel, lowestFrac, highestFrac,
                                              A, k, cfg.warp_n, warpSign);

                cv::Mat rightPanel = drawAnnotated(augImg, headerCols, augLabel,
                                                   augFname + "  [" + descStr + "]");
                if (leftPanel.rows != rightPanel.rows) {
                    double scale = static_cast<double>(leftPanel.rows) / rightPanel.rows;
                    cv::resize(rightPanel, rightPanel, cv::Size(), scale, scale);
                }
                cv::Mat sep(leftPanel.rows, 4, CV_8UC3, cv::Scalar(80, 80, 80));
                cv::Mat canvas;
                cv::hconcat(std::vector<cv::Mat>{leftPanel, sep, rightPanel}, canvas);
                const std::string WIN =
                    "Augmentation Preview  |  any key = next  |  q = quit";
                cv::namedWindow(WIN, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
                cv::imshow(WIN, canvas);
                int key = cv::waitKeyEx(0) & 0xFF;
                if (key == 'q' || key == 27) { aborted = true; continue; }
            }

            // ---- Horizontal flip of the augmented image --------------------
            if (!aborted) {
                cv::Mat flipped;
                cv::flip(augImg, flipped, 1);   // 1 = horizontal

                std::string flipFname = fs::path(augFname).stem().string()
                                        + "_f.png";
                std::string flipPath  = (outDir / flipFname).string();

                if (cv::imwrite(flipPath, flipped)) {
                    // Reflect all x-fractions: new_x = 1.0 - old_x
                    LabelRow flipLabel = transformLabel_flip(
                        augLabel, flipFname, headerCols);
                    outLabels.push_back(flipLabel);
                    ++generated;
                    std::cout << "         [" << v << "] FLIP"
                              << "  → " << flipFname << "\n";

                    if (cfg.visualize) {
                        cv::Mat leftPanel  = drawAnnotated(augImg,  headerCols,
                                                           augLabel,  augFname);
                        cv::Mat rightPanel = drawAnnotated(flipped, headerCols,
                                                           flipLabel,
                                                           flipFname + "  [FLIP]");
                        if (leftPanel.rows != rightPanel.rows) {
                            double scale = static_cast<double>(leftPanel.rows)
                                           / rightPanel.rows;
                            cv::resize(rightPanel, rightPanel, cv::Size(), scale, scale);
                        }
                        cv::Mat sep(leftPanel.rows, 4, CV_8UC3, cv::Scalar(80,80,80));
                        cv::Mat canvas;
                        cv::hconcat(std::vector<cv::Mat>{leftPanel, sep, rightPanel},
                                    canvas);
                        const std::string WIN =
                            "Augmentation Preview  |  any key = next  |  q = quit";
                        cv::namedWindow(WIN, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
                        cv::imshow(WIN, canvas);
                        int key = cv::waitKeyEx(0) & 0xFF;
                        if (key == 'q' || key == 27) aborted = true;
                    }
                } else {
                    std::cerr << "[WARN]  Failed to write " << flipPath << "\n";
                }
            }
        }
    }

    // ---- Merge new labels into the existing labels.csv ----------------------
    // Re-read the original labels, append new rows, write back as one file.
    std::string labelsOutPath = labelsPath;   // same file we read from
    std::vector<LabelRow> allLabels = sourceLabels;
    for (const auto& r : outLabels)
        allLabels.push_back(r);
    writeLabels(labelsOutPath, header, allLabels);

    std::cout << "\n" << (aborted ? "[ABORTED]" : "[DONE]")
              << "  Generated " << generated << " augmented images"
              << " (" << cfg.num_augmentations
              << " shift+warp versions × 2 with flips per source image).\n"
              << "[DONE]  Labels appended to " << labelsOutPath << "\n";
    return 0;
}
