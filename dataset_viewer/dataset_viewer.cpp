#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <map>
#include <iomanip>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Colour scheme:
//   r0 = furthest from camera  → RED
//   r1 = middle row            → YELLOW
//   r2 = closest to camera     → GREEN
// For N rows the mapping is:  rN-1-col gets the colour at index col
// (col 0 in CSV = closest = GREEN, last col = furthest = RED)
// ---------------------------------------------------------------------------
const std::vector<cv::Scalar> ROW_COLOURS = {
    cv::Scalar(0,   0,   255),   // RED    — furthest  (r0)
    cv::Scalar(0,   255, 255),   // YELLOW — middle    (r1)
    cv::Scalar(0,   255, 0  ),   // GREEN  — closest   (r2)
};

// Return the colour for a given row index (r0, r1, r2, ...)
// r0 → ROW_COLOURS[0] (RED), r1 → YELLOW, r2 → GREEN
cv::Scalar colourForRow(int rIndex, int numRows)
{
    // rIndex 0 = furthest = RED = ROW_COLOURS[0]
    int ci = std::min(rIndex, static_cast<int>(ROW_COLOURS.size()) - 1);
    return ROW_COLOURS[ci];
}

// ---------------------------------------------------------------------------
// Label row
// ---------------------------------------------------------------------------
struct LabelRow {
    std::string filename;
    int         road = 0;
    // cols[0] = closest row (loc_r(N-1)), cols[N-1] = furthest (loc_r0)
    // This is the order they appear in the CSV after filename,road_present
    std::vector<std::string> cols;
};

// ---------------------------------------------------------------------------
// Parse semicolon-separated x-fractions from a CSV cell
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
// Read labels.csv → header + rows
// ---------------------------------------------------------------------------
bool readLabels(const std::string& path,
                std::string& header,
                std::vector<LabelRow>& rows)
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
        if (firstLine) { header = line; firstLine = false; continue; }
        std::istringstream ss(line);
        std::string tok;
        LabelRow row;
        std::getline(ss, row.filename, ',');
        std::string roadStr;
        std::getline(ss, roadStr, ',');
        try { row.road = std::stoi(roadStr); } catch (...) {}
        while (std::getline(ss, tok, ','))
            row.cols.push_back(tok);
        rows.push_back(row);
    }
    return true;
}

// ---------------------------------------------------------------------------
// Read dataset.info → map of loc_rN → fraction
// ---------------------------------------------------------------------------
std::map<std::string, double> readDatasetInfo(const std::string& dsDir)
{
    std::map<std::string, double> info;
    std::string infoPath = (fs::path(dsDir) / "dataset.info").string();
    std::ifstream f(infoPath);
    if (!f.is_open()) {
        std::cerr << "[INFO]  dataset.info not found at \"" << infoPath << "\"\n";
        return info;
    }

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
        try { info[key] = std::stod(val); } catch (...) {}
    }
    return info;
}

// ---------------------------------------------------------------------------
// Collect column names from CSV header (after filename, road_present)
// Returns e.g. {"loc_r2", "loc_r1", "loc_r0"} — closest first as in CSV
// ---------------------------------------------------------------------------
std::vector<std::string> parseHeaderCols(const std::string& header)
{
    std::vector<std::string> cols;
    std::istringstream ss(header);
    std::string tok;
    int idx = 0;
    while (std::getline(ss, tok, ',')) {
        // trim whitespace
        size_t a = tok.find_first_not_of(" \t\r\n");
        size_t b = tok.find_last_not_of(" \t\r\n");
        if (a != std::string::npos) tok = tok.substr(a, b - a + 1);
        if (idx >= 2) cols.push_back(tok);
        ++idx;
    }
    return cols;
}

// ---------------------------------------------------------------------------
// Draw the annotated image:
//   - Horizontal line in the row's colour
//   - Vertical bar for each point, bottom anchored on the horizontal line,
//     bar extends upward by barHeight pixels
//   - Filename + index overlay in top-left corner
// ---------------------------------------------------------------------------
cv::Mat drawAnnotated(const cv::Mat&                   src,
                      const LabelRow&                  label,
                      const std::vector<std::string>&  headerCols,
                      const std::map<std::string, double>& info,
                      const std::string&               overlayText,
                      int                              barHeight = 20,
                      int                              barWidth  = 4)
{
    cv::Mat out = src.clone();
    int W = out.cols, H = out.rows;
    int N = static_cast<int>(headerCols.size());

    // headerCols[0] = closest row (loc_r(N-1)), headerCols[N-1] = furthest (loc_r0)
    // We want furthest = r0 = RED, so rIndex for col ci is (N-1-ci)
    for (int ci = 0; ci < N; ++ci) {
        const std::string& colName = headerCols[ci];

        // Look up the row fraction from dataset.info
        double rowFrac = 0.5;
        auto it = info.find(colName);
        if (it != info.end()) rowFrac = it->second;

        int rowPx = static_cast<int>(rowFrac * H);

        // rIndex: col 0 is closest = r(N-1), col N-1 is furthest = r0
        int rIndex = N - 1 - ci;
        cv::Scalar colour = colourForRow(rIndex, N);

        // Draw horizontal line
        cv::line(out, cv::Point(0, rowPx), cv::Point(W - 1, rowPx), colour, 2);

        // Draw vertical bars for each labelled point on this row
        if (ci < static_cast<int>(label.cols.size())) {
            auto xs = parseCellValues(label.cols[ci]);
            for (double xFrac : xs) {
                int px = static_cast<int>(xFrac * W);
                px = std::max(barWidth / 2, std::min(W - barWidth / 2 - 1, px));

                // Bar: bottom at rowPx, top at rowPx - barHeight
                int top    = std::max(0, rowPx - barHeight);
                int bottom = rowPx;
                int left   = px - barWidth / 2;
                int right  = px + barWidth / 2;

                cv::rectangle(out,
                              cv::Point(left, top),
                              cv::Point(right, bottom),
                              colour, cv::FILLED);
            }
        }
    }

    // Draw violet crop lines from dataset.info
    const cv::Scalar VIOLET(211, 0, 148);
    if (info.count("crop_top") && info.at("crop_top") > 0.0) {
        int row = static_cast<int>(info.at("crop_top") * H);
        cv::line(out, cv::Point(0, row), cv::Point(W - 1, row), VIOLET, 2);
    }
    if (info.count("crop_bottom") && info.at("crop_bottom") > 0.0) {
        int row = static_cast<int>((1.0 - info.at("crop_bottom")) * H);
        cv::line(out, cv::Point(0, row), cv::Point(W - 1, row), VIOLET, 2);
    }

    // Filename overlay — white text with dark shadow, top-left corner
    if (!overlayText.empty()) {
        const int    margin    = 8;
        const double fontScale = 1.2;
        const int    thickness = 1;
        const int    font      = cv::FONT_HERSHEY_SIMPLEX;
        cv::putText(out, overlayText,
                    cv::Point(margin + 1, margin + 30),
                    font, fontScale, cv::Scalar(0, 0, 0), thickness + 1);
        cv::putText(out, overlayText,
                    cv::Point(margin, margin + 29),
                    font, fontScale, cv::Scalar(255, 255, 255), thickness);
    }

    return out;
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
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    std::string dsPath;

    for (int i = 1; i < argc; ++i) {
        std::string flag = argv[i];
        if ((flag == "-ds" || flag == "--ds") && i + 1 < argc)
            dsPath = argv[++i];
        else if (flag == "-h" || flag == "--help") {
            std::cout << "Usage: " << argv[0] << " -ds <dataset_path>\n\n"
                      << "Displays each labelled image with horizontal lines and\n"
                      << "vertical bar markers at the labelled point positions.\n\n"
                      << "Colour coding:\n"
                      << "  RED    — furthest row (r0, top of road)\n"
                      << "  YELLOW — middle row  (r1)\n"
                      << "  GREEN  — closest row (r2, bottom of road)\n\n"
                      << "Keys:\n"
                      << "  n / Right arrow  — next image\n"
                      << "  f                — skip forward 50 images\n"
                      << "  p / Left arrow   — previous image\n"
                      << "  b                — skip back 50 images\n"
                      << "  q / ESC          — quit\n";
            return 0;
        }
    }

    if (dsPath.empty()) {
        std::cerr << "ERROR: -ds <dataset_path> is required.\n";
        return 1;
    }

    if (!fs::exists(dsPath) || !fs::is_directory(dsPath)) {
        std::cerr << "ERROR: dataset path \"" << dsPath << "\" not found.\n";
        return 1;
    }

    // ---- Load dataset.info -------------------------------------------------
    auto info = readDatasetInfo(dsPath);
    if (info.empty()) {
        std::cerr << "ERROR: could not read dataset.info from \"" << dsPath << "\".\n";
        return 1;
    }
    std::cout << "[INFO]  Row fractions loaded:\n";
    for (const auto& kv : info) {
        if (kv.first.substr(0, 5) == "loc_r")
            std::cout << "         " << kv.first << " = " << kv.second << "\n";
    }
    if (info.count("crop_top"))
        std::cout << "[INFO]  crop_top    = " << info.at("crop_top")    << "\n";
    if (info.count("crop_bottom"))
        std::cout << "[INFO]  crop_bottom = " << info.at("crop_bottom") << "\n";

    // ---- Load labels.csv ---------------------------------------------------
    std::string labelsPath = (fs::path(dsPath) / "labels.csv").string();
    std::string header;
    std::vector<LabelRow> labels;
    if (!readLabels(labelsPath, header, labels)) return 1;

    auto headerCols = parseHeaderCols(header);
    std::cout << "[INFO]  Loaded " << labels.size() << " label rows.\n";
    std::cout << "[INFO]  Columns: ";
    for (size_t i = 0; i < headerCols.size(); ++i)
        std::cout << (i ? ", " : "") << headerCols[i];
    std::cout << "\n";

    // Build filename → label map
    std::map<std::string, LabelRow> labelMap;
    for (const auto& r : labels)
        labelMap[r.filename] = r;

    // ---- Collect PNGs and filter to labelled images only -------------------
    auto allPngs = collectPNGs(dsPath);
    std::vector<std::string> pngPaths;
    for (const auto& p : allPngs) {
        std::string fname = fs::path(p).filename().string();
        if (labelMap.count(fname))
            pngPaths.push_back(p);
    }

    std::cout << "[INFO]  PNG files on disk  : " << allPngs.size()  << "\n";
    std::cout << "[INFO]  Labelled images    : " << pngPaths.size() << "\n";

    if (pngPaths.empty()) {
        std::cerr << "ERROR: no labelled PNG files found in \"" << dsPath << "\".\n";
        return 1;
    }
    std::cout << "Controls:\n"
              << "  n / Right arrow  — next image\n"
              << "  f                — skip forward 50 images\n"
              << "  p / Left arrow   — previous image\n"
              << "  b                — skip back 50 images\n"
              << "  q / ESC          — quit\n\n";

    // ---- Display loop ------------------------------------------------------
    const std::string WIN = "Dataset Viewer  |  n=next  p=prev  q=quit";
    cv::namedWindow(WIN, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);

    int index = 0;

    auto showImage = [&]() {
        const std::string& pngPath = pngPaths[index];
        std::string fname = fs::path(pngPath).filename().string();

        cv::Mat src = cv::imread(pngPath);
        if (src.empty()) {
            std::cerr << "[WARN]  Could not read " << pngPath << "\n";
            return;
        }

        // Look up label — use empty row if not found
        LabelRow label;
        auto it = labelMap.find(fname);
        if (it != labelMap.end())
            label = it->second;
        else
            std::cout << "[WARN]  No label for " << fname << "\n";

        std::string overlayText = fname
            + "  (" + std::to_string(index + 1)
            + " / " + std::to_string(pngPaths.size()) + ")"
            + (label.road == 0 ? "  [no road]" : "");

        cv::Mat annotated = drawAnnotated(src, label, headerCols,
                                          info, overlayText);

        cv::setWindowTitle(WIN, fname + "  ("
                           + std::to_string(index + 1) + "/"
                           + std::to_string(pngPaths.size()) + ")");
        cv::imshow(WIN, annotated);

        std::cout << "[VIEW]  " << (index + 1) << "/" << pngPaths.size()
                  << "  " << fname
                  << (label.road == 0 ? "  [no road]" : "") << "\n";
    };

    showImage();

    while (true) {
        int key = cv::waitKeyEx(0);
        int lkey = key & 0xFF;
        int n    = static_cast<int>(pngPaths.size());

        // Next 1: n, right arrow
        if (lkey == 'n' || key == 65363 || key == 0xFF53) {
            index = (index + 1) % n;
            showImage();
        }
        // Next 10: f (fast forward)
        else if (lkey == 'f') {
            index = (index + 50) % n;
            showImage();
        }
        // Previous 1: p, left arrow
        else if (lkey == 'p' || key == 65361 || key == 0xFF51) {
            index = (index - 1 + n) % n;
            showImage();
        }
        // Previous 10: b (fast back)
        else if (lkey == 'b') {
            index = (index - 50 + n) % n;
            showImage();
        }
        // Quit: q, ESC
        else if (lkey == 'q' || lkey == 27) {
            std::cout << "Exiting.\n";
            cv::destroyAllWindows();
            return 0;
        }
    }
}
