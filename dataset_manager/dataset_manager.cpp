#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <algorithm>
#include <random>
#include <iomanip>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Configuration — loaded from dataset_manager.ini
// ---------------------------------------------------------------------------
struct DmConfig {
    // Root directory that contains all source dataset subdirectories
    std::string root_dir;

    // Source dataset subdirectory names to include
    std::vector<std::string> datasets;

    // Name of the composite output dataset (created under root_dir)
    std::string composite_name;

    // Target image size for resizing on copy.
    // Set both to 0 to skip resizing (just copy the file as-is).
    int target_width  = 0;
    int target_height = 0;

    // Fraction of all collected images to retain before splitting [0.0, 1.0].
    // Images are randomly selected (via the shuffle).  1.0 = keep all.
    double retention_fraction = 1.0;

    // Fraction of images to place in the training set [0.0, 1.0]
    // Remainder goes to test.  Default 80/20 split.
    double train_fraction = 0.80;

    // Shuffle images before splitting (recommended)
    bool shuffle = true;

    // Random seed (0 = use random_device)
    unsigned int seed = 0;

    // Config file path
    std::string config_file = "dataset_manager.ini";
};

// ---------------------------------------------------------------------------
// Label row — one entry in labels.csv
// ---------------------------------------------------------------------------
struct LabelRow {
    std::string              filename;
    int                      road = 0;
    std::vector<std::string> cols;   // one entry per header column after "road"
};

// ---------------------------------------------------------------------------
// Minimal INI parser — same pattern as augment.ini
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
// Split a comma-separated string into trimmed tokens
// ---------------------------------------------------------------------------
std::vector<std::string> splitComma(const std::string& s)
{
    std::vector<std::string> out;
    std::istringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        // trim
        size_t a = tok.find_first_not_of(" \t\r\n");
        size_t b = tok.find_last_not_of(" \t\r\n");
        if (a != std::string::npos)
            out.push_back(tok.substr(a, b - a + 1));
    }
    return out;
}

// ---------------------------------------------------------------------------
// Read dataset.info → ordered list of "key = value" lines (comments stripped)
// Returns an empty map if the file cannot be opened.
// ---------------------------------------------------------------------------
std::map<std::string, std::string> readDatasetInfo(const std::string& dsDir)
{
    std::map<std::string, std::string> info;
    std::string infoPath = (fs::path(dsDir) / "dataset.info").string();
    std::ifstream f(infoPath);
    if (!f.is_open()) return info;

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
        if (!key.empty()) info[key] = val;
    }
    return info;
}

// ---------------------------------------------------------------------------
// Load config
// ---------------------------------------------------------------------------
bool loadDmConfig(DmConfig& cfg, const std::string& path)
{
    auto kv = parseIni(path);
    if (kv.empty()) {
        std::cerr << "[CONFIG] Could not open \"" << path
                  << "\" — using defaults.\n";
        return false;
    }
    try {
        if (kv.count("root_dir"))        cfg.root_dir        = kv.at("root_dir");
        if (kv.count("composite_name"))  cfg.composite_name  = kv.at("composite_name");
        if (kv.count("target_width"))      cfg.target_width      = std::stoi(kv.at("target_width"));
        if (kv.count("target_height"))     cfg.target_height     = std::stoi(kv.at("target_height"));
        if (kv.count("retention_fraction")) cfg.retention_fraction = std::stod(kv.at("retention_fraction"));
        if (kv.count("train_fraction"))     cfg.train_fraction     = std::stod(kv.at("train_fraction"));
        if (kv.count("shuffle"))         cfg.shuffle         = (kv.at("shuffle") == "true"
                                                                 || kv.at("shuffle") == "1");
        if (kv.count("seed"))            cfg.seed            = static_cast<unsigned>(
                                                                 std::stoul(kv.at("seed")));
        if (kv.count("datasets"))        cfg.datasets        = splitComma(kv.at("datasets"));
    } catch (const std::exception& e) {
        std::cerr << "[CONFIG] Parse error: " << e.what() << "\n";
        return false;
    }

    std::cout << "[CONFIG] Loaded \"" << path << "\"\n"
              << "         root_dir            = " << cfg.root_dir            << "\n"
              << "         composite_name      = " << cfg.composite_name      << "\n"
              << "         target_size         = " << cfg.target_width
              << " x " << cfg.target_height
              << (cfg.target_width == 0 ? "  (no resize)" : "") << "\n"
              << "         retention_fraction  = " << cfg.retention_fraction  << "\n"
              << "         train_fraction      = " << cfg.train_fraction      << "\n"
              << "         shuffle             = " << (cfg.shuffle ? "true" : "false") << "\n"
              << "         seed                = " << cfg.seed                << "\n"
              << "         datasets            = ";
    for (size_t i = 0; i < cfg.datasets.size(); ++i)
        std::cout << (i ? ", " : "") << cfg.datasets[i];
    std::cout << "\n";
    return true;
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
// Write labels.csv
// ---------------------------------------------------------------------------
void writeLabels(const std::string& path,
                 const std::string& header,
                 const std::vector<LabelRow>& rows)
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
// Resolve a filename collision in a destination directory.
// If destDir/fname doesn't exist, returns fname unchanged.
// Otherwise appends _v1, _v2, ... until a free name is found.
// ---------------------------------------------------------------------------
std::string resolveCollision(const fs::path& destDir,
                             const std::string& fname,
                             const std::set<std::string>& usedNames)
{
    if (usedNames.find(fname) == usedNames.end()) return fname;

    fs::path p(fname);
    std::string stem = p.stem().string();
    std::string ext  = p.extension().string();

    for (int v = 1; ; ++v) {
        std::string candidate = stem + "_v" + std::to_string(v) + ext;
        if (usedNames.find(candidate) == usedNames.end())
            return candidate;
    }
}

// ---------------------------------------------------------------------------
// Entry: one image to be copied, with its source path and label row
// ---------------------------------------------------------------------------
struct Entry {
    fs::path   srcPath;
    LabelRow   label;
};

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    DmConfig cfg;

    for (int i = 1; i < argc; ++i) {
        std::string flag = argv[i];
        if ((flag == "-cfg" || flag == "--cfg") && i + 1 < argc)
            cfg.config_file = argv[++i];
        else if (flag == "--help" || flag == "-h") {
            std::cout << "Usage: " << argv[0] << " [-cfg <config>]\n\n"
                      << "Builds a composite train/test dataset from multiple source datasets.\n\n"
                      << "Config file parameters (dataset_manager.ini):\n"
                      << "  root_dir        Root directory containing source dataset folders\n"
                      << "  datasets        Comma-separated list of source dataset subfolder names\n"
                      << "  composite_name  Name for the composite output dataset folder\n"
                      << "  train_fraction  Fraction of images for training (default: 0.80)\n"
                      << "  shuffle         Shuffle before splitting: true/false (default: true)\n"
                      << "  seed            RNG seed; 0 = random (default: 0)\n";
            return 0;
        }
    }

    if (!loadDmConfig(cfg, cfg.config_file)) return 1;

    // ---- Validate config ---------------------------------------------------
    if (cfg.root_dir.empty()) {
        std::cerr << "ERROR: root_dir not set in config.\n"; return 1;
    }
    if (cfg.composite_name.empty()) {
        std::cerr << "ERROR: composite_name not set in config.\n"; return 1;
    }
    if (cfg.datasets.empty()) {
        std::cerr << "ERROR: no datasets listed in config.\n"; return 1;
    }
    if (cfg.train_fraction <= 0.0 || cfg.train_fraction >= 1.0) {
        std::cerr << "ERROR: train_fraction must be in (0, 1).\n"; return 1;
    }

    fs::path rootDir = cfg.root_dir;
    if (!fs::exists(rootDir) || !fs::is_directory(rootDir)) {
        std::cerr << "ERROR: root_dir \"" << rootDir << "\" not found.\n"; return 1;
    }

    // ---- Validate dataset.info files match across all datasets --------------
    std::map<std::string, std::string> referenceInfo;
    std::string referenceDs;

    for (const auto& dsName : cfg.datasets) {
        fs::path dsDir = rootDir / dsName;
        if (!fs::exists(dsDir) || !fs::is_directory(dsDir)) continue;

        auto info = readDatasetInfo(dsDir.string());
        if (info.empty()) {
            std::cerr << "\nERROR: dataset.info missing or empty for dataset \""
                      << dsName << "\"\n"
                      << "       Expected at: " << (dsDir / "dataset.info") << "\n"
                      << "       All datasets must have a dataset.info file.\n"
                      << "       Run image_viewer on \"" << dsName
                      << "\" first to generate it.\n";
            return 1;
        }

        if (referenceInfo.empty()) {
            referenceInfo = info;
            referenceDs   = dsName;
            std::cout << "[INFO]  Reference dataset.info from \"" << dsName << "\":\n";
            for (const auto& kv : referenceInfo)
                std::cout << "         " << kv.first << " = " << kv.second << "\n";
        } else {
            // Compare this dataset's info against the reference
            bool mismatch = false;
            std::ostringstream diffMsg;

            // Check all keys in reference are present and equal in this dataset
            for (const auto& kv : referenceInfo) {
                auto it = info.find(kv.first);
                if (it == info.end()) {
                    diffMsg << "       Key \"" << kv.first << "\" present in \""
                            << referenceDs << "\" but missing in \"" << dsName << "\"\n";
                    mismatch = true;
                } else if (it->second != kv.second) {
                    diffMsg << "       Key \"" << kv.first << "\": \""
                            << referenceDs << "\" = " << kv.second
                            << "  vs  \"" << dsName << "\" = " << it->second << "\n";
                    mismatch = true;
                }
            }
            // Check for keys present in this dataset but not in reference
            for (const auto& kv : info) {
                if (referenceInfo.find(kv.first) == referenceInfo.end()) {
                    diffMsg << "       Key \"" << kv.first << "\" present in \""
                            << dsName << "\" but missing in \"" << referenceDs << "\"\n";
                    mismatch = true;
                }
            }

            if (mismatch) {
                std::cerr << "\nERROR: dataset.info mismatch detected.\n"
                          << "       Cannot build composite dataset because the source\n"
                          << "       datasets have different geometry configurations.\n"
                          << "       All datasets must share identical line_rows,\n"
                          << "       crop_top, and crop_bottom values.\n\n"
                          << "       Differences found between \""
                          << referenceDs << "\" and \"" << dsName << "\":\n"
                          << diffMsg.str()
                          << "\n       Fix: re-run image_viewer on each dataset with\n"
                          << "       the same config.ini to regenerate matching dataset.info files.\n";
                return 1;
            }
            std::cout << "[INFO]  dataset.info OK for \"" << dsName << "\"\n";
        }
    }
    std::cout << "\n";

    // ---- Collect all entries from each source dataset ----------------------
    std::string header;
    std::vector<Entry> allEntries;

    for (const auto& dsName : cfg.datasets) {
        fs::path dsDir = rootDir / dsName;
        if (!fs::exists(dsDir) || !fs::is_directory(dsDir)) {
            std::cerr << "[WARN]  Dataset dir not found: " << dsDir << "\n";
            continue;
        }

        fs::path labelsPath = dsDir / "labels.csv";
        std::string dsHeader;
        std::vector<LabelRow> rows;
        if (!readLabels(labelsPath.string(), dsHeader, rows)) continue;

        // Use the first successfully read header as the composite header
        if (header.empty()) header = dsHeader;

        int loaded = 0;
        for (const auto& row : rows) {
            fs::path imgPath = dsDir / row.filename;
            if (!fs::exists(imgPath)) {
                std::cerr << "[WARN]  Image not found: " << imgPath << "\n";
                continue;
            }
            allEntries.push_back({imgPath, row});
            ++loaded;
        }
        std::cout << "[DS]    " << dsName << "  →  " << loaded << " images\n";
    }

    if (allEntries.empty()) {
        std::cerr << "ERROR: no images found across all datasets.\n"; return 1;
    }
    std::cout << "\n[INFO]  Total images collected : " << allEntries.size() << "\n";

    // ---- Shuffle and apply retention ----------------------------------------
    std::mt19937 rng(cfg.seed == 0
                     ? std::random_device{}()
                     : static_cast<std::mt19937::result_type>(cfg.seed));

    if (cfg.shuffle)
        std::shuffle(allEntries.begin(), allEntries.end(), rng);

    // Apply retention: keep only the first retention_fraction of the
    // shuffled pool (random because the list was just shuffled)
    if (cfg.retention_fraction < 1.0) {
        size_t keepCount = static_cast<size_t>(
            std::round(cfg.retention_fraction * allEntries.size()));
        keepCount = std::max(size_t(2), std::min(keepCount, allEntries.size()));
        allEntries.resize(keepCount);
        std::cout << "[INFO]  After retention (" << std::fixed
                  << std::setprecision(2) << cfg.retention_fraction * 100.0
                  << "%)  : " << allEntries.size() << " images kept\n";
    }

    // ---- Train / test split -------------------------------------------------

    size_t trainCount = static_cast<size_t>(
        std::round(cfg.train_fraction * allEntries.size()));
    trainCount = std::max(size_t(1),
                 std::min(trainCount, allEntries.size() - 1));

    std::cout << "[INFO]  Train : " << trainCount
              << "  Test : " << (allEntries.size() - trainCount) << "\n\n";

    // ---- Create output directories -----------------------------------------
    fs::path compDir  = rootDir / cfg.composite_name;
    fs::path trainDir = compDir / "train";
    fs::path testDir  = compDir / "test";
    fs::create_directories(trainDir);
    fs::create_directories(testDir);

    // ---- Copy images and build label lists ---------------------------------
    auto copyEntries = [&](const std::vector<Entry>& entries,
                           const fs::path& destDir,
                           const std::string& splitName) -> std::vector<LabelRow>
    {
        std::vector<LabelRow> outRows;
        std::set<std::string> usedNames;
        int copied = 0, skipped = 0;

        for (const auto& e : entries) {
            // Resolve filename collision
            std::string destFname = resolveCollision(
                destDir, e.label.filename, usedNames);
            usedNames.insert(destFname);

            fs::path destPath = destDir / destFname;

            bool writeOk = false;
            if (cfg.target_width > 0 && cfg.target_height > 0) {
                // Read, resize, write
                cv::Mat img = cv::imread(e.srcPath.string());
                if (img.empty()) {
                    std::cerr << "[WARN]  Cannot read " << e.srcPath << "\n";
                    ++skipped;
                    continue;
                }
                cv::Mat resized;
                cv::resize(img, resized,
                           cv::Size(cfg.target_width, cfg.target_height),
                           0, 0, cv::INTER_AREA);
                writeOk = cv::imwrite(destPath.string(), resized);
                if (!writeOk)
                    std::cerr << "[WARN]  Failed to write " << destPath << "\n";
            } else {
                // No resize — plain file copy
                std::error_code ec;
                fs::copy_file(e.srcPath, destPath,
                              fs::copy_options::overwrite_existing, ec);
                writeOk = !ec;
                if (!writeOk)
                    std::cerr << "[WARN]  Failed to copy "
                              << e.srcPath << " → " << destPath
                              << " : " << ec.message() << "\n";
            }

            if (!writeOk) { ++skipped; continue; }

            LabelRow outRow  = e.label;
            outRow.filename  = destFname;
            outRows.push_back(outRow);
            ++copied;

            if (destFname != e.label.filename)
                std::cout << "         [RENAME] " << e.label.filename
                          << " → " << destFname << "\n";
        }

        std::cout << "[" << splitName << "] "
                  << (cfg.target_width > 0 ? "Resized & copied " : "Copied ")
                  << copied << " images, skipped " << skipped << "\n";
        return outRows;
    };

    std::vector<Entry> trainEntries(allEntries.begin(),
                                    allEntries.begin() + trainCount);
    std::vector<Entry> testEntries (allEntries.begin() + trainCount,
                                    allEntries.end());

    auto trainLabels = copyEntries(trainEntries, trainDir, "TRAIN");
    auto testLabels  = copyEntries(testEntries,  testDir,  "TEST ");

    // ---- Write composite labels.csv files ----------------------------------
    writeLabels((trainDir / "labels.csv").string(), header, trainLabels);
    writeLabels((testDir  / "labels.csv").string(), header, testLabels);

    // ---- Copy dataset.info from the first source dataset -------------------
    // All source datasets must share the same geometry (same line_rows and
    // crop values), so we copy the first one found into each output split.
    bool infoCopied = false;
    for (const auto& dsName : cfg.datasets) {
        fs::path srcInfo = rootDir / dsName / "dataset.info";
        if (fs::exists(srcInfo)) {
            std::error_code ec;
            fs::copy_file(srcInfo, trainDir / "dataset.info",
                          fs::copy_options::overwrite_existing, ec);
            if (!ec) {
                fs::copy_file(srcInfo, testDir / "dataset.info",
                              fs::copy_options::overwrite_existing, ec);
            }
            if (!ec) {
                std::cout << "[INFO]  Copied dataset.info from \"" << dsName << "\"\n";
                infoCopied = true;
            } else {
                std::cerr << "[WARN]  Failed to copy dataset.info: " << ec.message() << "\n";
            }
            break;
        }
    }
    if (!infoCopied)
        std::cerr << "[WARN]  No dataset.info found in any source dataset.\n"
                  << "        Downstream tools may not know crop/row geometry.\n";

    std::cout << "\n[DONE]  Composite dataset \"" << cfg.composite_name
              << "\" built in " << compDir << "\n"
              << "[DONE]  Train: " << trainLabels.size()
              << " images  |  Test: " << testLabels.size() << " images\n";
    return 0;
}
