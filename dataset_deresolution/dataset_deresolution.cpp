#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
struct DerConfig {
    // Full path to the source dataset directory
    std::string source_dir;

    // Full path to the output (low-resolution) dataset directory.
    // Will be created if it does not exist.
    std::string output_dir;

    // Target image dimensions in pixels
    int target_width  = 160;
    int target_height = 120;

    // Config file path
    std::string config_file = "dataset_deresolution.ini";
};

// ---------------------------------------------------------------------------
// Minimal INI parser
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

// ---------------------------------------------------------------------------
// Load config
// ---------------------------------------------------------------------------
bool loadConfig(DerConfig& cfg, const std::string& path)
{
    auto kv = parseIni(path);
    if (kv.empty()) {
        std::cerr << "[CONFIG] Could not open \"" << path
                  << "\" — using defaults.\n";
        return false;
    }
    try {
        if (kv.count("source_dir"))    cfg.source_dir    = kv.at("source_dir");
        if (kv.count("output_dir"))    cfg.output_dir    = kv.at("output_dir");
        if (kv.count("target_width"))  cfg.target_width  = std::stoi(kv.at("target_width"));
        if (kv.count("target_height")) cfg.target_height = std::stoi(kv.at("target_height"));
    } catch (const std::exception& e) {
        std::cerr << "[CONFIG] Parse error: " << e.what() << "\n";
        return false;
    }

    std::cout << "[CONFIG] Loaded \"" << path << "\"\n"
              << "         source_dir    = " << cfg.source_dir    << "\n"
              << "         output_dir    = " << cfg.output_dir    << "\n"
              << "         target_width  = " << cfg.target_width  << "\n"
              << "         target_height = " << cfg.target_height << "\n";
    return true;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    DerConfig cfg;

    for (int i = 1; i < argc; ++i) {
        std::string flag = argv[i];
        if ((flag == "-cfg" || flag == "--cfg") && i + 1 < argc)
            cfg.config_file = argv[++i];
        else if (flag == "-h" || flag == "--help") {
            std::cout << "Usage: " << argv[0] << " [-cfg <config>]\n\n"
                      << "Resizes all PNG images in a source dataset to a target\n"
                      << "resolution and saves them to a new output directory.\n"
                      << "The labels.csv and dataset.info files are copied as-is.\n\n"
                      << "Config file parameters (dataset_deresolution.ini):\n"
                      << "  source_dir    Full path to source dataset directory\n"
                      << "  output_dir    Full path to output dataset directory\n"
                      << "  target_width  Output image width  in pixels\n"
                      << "  target_height Output image height in pixels\n";
            return 0;
        }
    }

    if (!loadConfig(cfg, cfg.config_file)) return 1;

    // ---- Validate ----------------------------------------------------------
    if (cfg.source_dir.empty()) {
        std::cerr << "ERROR: source_dir not set in config.\n"; return 1;
    }
    if (cfg.output_dir.empty()) {
        std::cerr << "ERROR: output_dir not set in config.\n"; return 1;
    }
    if (cfg.target_width <= 0 || cfg.target_height <= 0) {
        std::cerr << "ERROR: target_width and target_height must be > 0.\n"; return 1;
    }
    if (!fs::exists(cfg.source_dir) || !fs::is_directory(cfg.source_dir)) {
        std::cerr << "ERROR: source_dir \"" << cfg.source_dir << "\" not found.\n";
        return 1;
    }
    // Check source and output are not the same directory
    // Use weakly_canonical for both so the check works even if output_dir
    // doesn't exist yet (weakly_canonical doesn't require the path to exist)
    if (fs::weakly_canonical(fs::path(cfg.source_dir)) ==
        fs::weakly_canonical(fs::path(cfg.output_dir))) {
        std::cerr << "ERROR: source_dir and output_dir must be different.\n"
                  << "       source_dir : " << cfg.source_dir << "\n"
                  << "       output_dir : " << cfg.output_dir << "\n"
                  << "       Please specify a different output_dir in the config.\n";
        return 1;
    }

    // ---- Create output directory -------------------------------------------
    std::error_code ec;
    fs::create_directories(cfg.output_dir, ec);
    if (ec) {
        std::cerr << "ERROR: Cannot create output_dir \""
                  << cfg.output_dir << "\": " << ec.message() << "\n";
        return 1;
    }
    std::cout << "\n[INFO]  Output dir created : " << cfg.output_dir << "\n\n";

    // ---- Collect PNG files -------------------------------------------------
    std::vector<fs::path> pngPaths;
    for (const auto& entry : fs::directory_iterator(cfg.source_dir)) {
        if (!entry.is_regular_file()) continue;
        auto ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".png") pngPaths.push_back(entry.path());
    }
    std::sort(pngPaths.begin(), pngPaths.end());
    std::cout << "[INFO]  PNG files found : " << pngPaths.size() << "\n\n";

    // ---- Resize and save each image ----------------------------------------
    int processed = 0, failed = 0;
    const cv::Size targetSize(cfg.target_width, cfg.target_height);

    for (const auto& srcPath : pngPaths) {
        cv::Mat img = cv::imread(srcPath.string());
        if (img.empty()) {
            std::cerr << "[WARN]  Cannot read " << srcPath.filename() << "\n";
            ++failed;
            continue;
        }

        cv::Mat resized;
        cv::resize(img, resized, targetSize, 0, 0, cv::INTER_AREA);

        fs::path destPath = fs::path(cfg.output_dir) / srcPath.filename();
        if (!cv::imwrite(destPath.string(), resized)) {
            std::cerr << "[WARN]  Cannot write " << destPath << "\n";
            ++failed;
            continue;
        }

        ++processed;
        if (processed % 100 == 0)
            std::cout << "[INFO]  Processed " << processed << " / "
                      << pngPaths.size() << "\n" << std::flush;
    }

    std::cout << "\n[INFO]  Images processed : " << processed << "\n"
              << "[INFO]  Images failed    : " << failed    << "\n";

    // ---- Copy labels.csv ---------------------------------------------------
    fs::path srcLabels = fs::path(cfg.source_dir) / "labels.csv";
    if (fs::exists(srcLabels)) {
        fs::path dstLabels = fs::path(cfg.output_dir) / "labels.csv";
        fs::copy_file(srcLabels, dstLabels,
                      fs::copy_options::overwrite_existing, ec);
        if (ec)
            std::cerr << "[WARN]  Failed to copy labels.csv: " << ec.message() << "\n";
        else
            std::cout << "[INFO]  Copied labels.csv\n";
    } else {
        std::cerr << "[WARN]  labels.csv not found in source_dir\n";
    }

    // ---- Copy dataset.info -------------------------------------------------
    fs::path srcInfo = fs::path(cfg.source_dir) / "dataset.info";
    if (fs::exists(srcInfo)) {
        fs::path dstInfo = fs::path(cfg.output_dir) / "dataset.info";
        fs::copy_file(srcInfo, dstInfo,
                      fs::copy_options::overwrite_existing, ec);
        if (ec)
            std::cerr << "[WARN]  Failed to copy dataset.info: " << ec.message() << "\n";
        else
            std::cout << "[INFO]  Copied dataset.info\n";
    } else {
        std::cerr << "[WARN]  dataset.info not found in source_dir\n";
    }

    std::cout << "\n[DONE]  Low-resolution dataset written to \""
              << cfg.output_dir << "\"\n"
              << "[DONE]  Resolution: " << cfg.target_width
              << " x " << cfg.target_height << "\n";
    return 0;
}
