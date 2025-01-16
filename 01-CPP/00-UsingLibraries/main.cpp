#include <iostream>
#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

int main() {

    // Use Boost Filesystem to create a directory
    fs::path dir("example_dir");
    if (fs::create_directory(dir)) {
        std::cout << "Directory created: " << dir << "\n";
    } else {
        std::cout << "Directory already exists or could not be created: " << dir << "\n";
    }

    return 0;
}