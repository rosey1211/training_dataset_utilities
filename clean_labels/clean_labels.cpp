#include <filesystem>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cstring>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Returns true if the filename stem contains an augmentation suffix:
//   _sw<N>   shift + exponential warp
//   _sl<N>   shift + linear warp
//   _c<N>    non-road copy
//   _f       horizontal flip (ends in _f)
// ---------------------------------------------------------------------------
bool isAugmented(const std::string& filename)
{
    std::string stem = fs::path(filename).stem().string();

    // Ends in _f
    if (stem.size() >= 2 && stem.back() == 'f' && stem[stem.size()-2] == '_')
        return true;

    // Contains _sw, _sl, or _c followed by a digit
    for (const char* tag : {"_sw", "_sl", "_c"}) {
        auto pos = stem.rfind(tag);
        if (pos == std::string::npos) continue;
        size_t after = pos + std::strlen(tag);
        if (after < stem.size() && std::isdigit((unsigned char)stem[after]))
            return true;
    }
    return false;
}

int main(int argc, char* argv[])
{
    std::string dsPath;
    bool dryRun = false;

    for (int i = 1; i < argc; ++i) {
        std::string flag = argv[i];
        if ((flag == "-ds" || flag == "--ds") && i + 1 < argc)
            dsPath = argv[++i];
        else if (flag == "--dry-run")
            dryRun = true;
        else if (flag == "-h" || flag == "--help") {
            std::cout << "Usage: " << argv[0] << " -ds <dataset_path> [--dry-run]\n\n"
                      << "Removes augmented entries from labels.csv.\n"
                      << "Augmented files are identified by stem suffixes:\n"
                      << "  _sw<N>   shift + exponential warp\n"
                      << "  _sl<N>   shift + linear warp\n"
                      << "  _c<N>    non-road copy\n"
                      << "  _f       horizontal flip\n\n"
                      << "Options:\n"
                      << "  --dry-run   Show what would be removed without writing anything\n";
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

    std::string labelsPath = (fs::path(dsPath) / "labels.csv").string();
    std::ifstream in(labelsPath);
    if (!in.is_open()) {
        std::cerr << "ERROR: Cannot open \"" << labelsPath << "\"\n";
        return 1;
    }

    // Read all lines
    std::string header;
    std::vector<std::string> kept;
    std::vector<std::string> removed;
    bool firstLine = true;

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (firstLine) {
            header = line;
            firstLine = false;
            continue;
        }
        // Extract filename (first comma-separated field)
        std::string fname = line.substr(0, line.find(','));
        if (isAugmented(fname)) {
            removed.push_back(fname);
        } else {
            kept.push_back(line);
        }
    }
    in.close();

    std::cout << "[INFO]  Dataset     : " << dsPath      << "\n"
              << "[INFO]  Total rows  : " << (kept.size() + removed.size()) << "\n"
              << "[INFO]  Kept        : " << kept.size()    << "\n"
              << "[INFO]  Removed     : " << removed.size() << "\n";

    if (!removed.empty()) {
        std::cout << "\nAugmented entries to remove:\n";
        for (const auto& f : removed)
            std::cout << "  " << f << "\n";
    } else {
        std::cout << "\n[INFO]  No augmented entries found — nothing to do.\n";
        return 0;
    }

    if (dryRun) {
        std::cout << "\n[DRY-RUN]  No changes written.\n";
        return 0;
    }

    // Write back only the kept rows
    std::ofstream out(labelsPath, std::ios::trunc);
    if (!out.is_open()) {
        std::cerr << "ERROR: Cannot write \"" << labelsPath << "\"\n";
        return 1;
    }
    out << header << '\n';
    for (const auto& l : kept)
        out << l << '\n';

    std::cout << "\n[DONE]  labels.csv updated — "
              << removed.size() << " augmented entries removed, "
              << kept.size() << " original entries kept.\n";
    return 0;
}
