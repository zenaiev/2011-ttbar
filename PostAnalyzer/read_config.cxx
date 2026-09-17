#include "read_config.h"
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

int read_int(const std::string& fname, const std::string& pattern, int default_value) {
  bool flag_found = false;  // Default value if not found in file
  int value = default_value;

  // If input file name is empty, return the default value
  if (fname.size() == 0) {
    return value;
  }

  // Open the input file
  std::ifstream inputFile(fname);
  if (!inputFile.is_open()) {
    std::cerr << "Error: Could not open file!" << std::endl;
    throw 1;
  }

  std::string line;
  while (std::getline(inputFile, line)) {
    // Skip lines starting with # (comments)
    if (line[0] == '#') continue;
    // Check if the line contains pattern
    if (line.find(pattern) != std::string::npos) {
      // Extract the value after '='
      std::istringstream iss(line);
      std::string token;
      
      iss >> token;  // Reads pattern
      iss >> value;  // Reads the integer value (0 or 1)
      
      flag_found = true;  // Convert to bool
      break;  // Exit loop once found
    }
  }

  inputFile.close();

  if(flag_found) {
    printf("Found in config %s %d\n", pattern.c_str(), value);
  }
  else {
    printf("Not found in config %s -> using default %d\n", pattern.c_str(), value);
  }
  return value;
}

std::string read_string(const std::string& fname, const std::string& pattern, const std::string& default_value) {
  std::string value = default_value;
  bool flag_found = false;
  if (fname.size() == 0) {
    return value;
  }
  std::ifstream inputFile(fname);
  if (!inputFile.is_open()) {
    std::cerr << "Error: Could not open file!" << std::endl;
    throw 1;
  }
  std::string line;
  while (std::getline(inputFile, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream iss(line);
    std::string token, v;
    // точний збіг назви (на відміну від read_int), щоб krNNModel не спрацьовував на krNNModelGen
    if (!(iss >> token) || token != pattern) continue;
    if (iss >> v) {
      value = v;
      flag_found = true;
    }
    break;
  }
  inputFile.close();
  if (flag_found) printf("Found in config %s %s\n", pattern.c_str(), value.c_str());
  else printf("Not found in config %s -> using default %s\n", pattern.c_str(), value.c_str());
  return value;
}
