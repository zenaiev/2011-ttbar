#pragma once
#include <string>
int read_int(const std::string& fname, const std::string& pattern, int default_value = 0);
// рядкове значення параметра (перше слово після назви); назва параметра має збігатися точно
std::string read_string(const std::string& fname, const std::string& pattern, const std::string& default_value = "");
