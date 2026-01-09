#include "read_csv.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

std::vector<std::vector<float>> read_csv(const std::string& filename) {
    std::vector<std::vector<float>> data;
    std::ifstream file(filename);
    std::string line;

    while (std::getline(file, line)) {  // getline is from <string>
        std::stringstream ss(line);
        std::string value;
        std::vector<float> row;

        while (std::getline(ss, value, ',')) {
            row.push_back(std::stof(value));  // stof is from <string>
        }

        data.push_back(row);
    }

    return data;
}



void read_csv(float* inp, std::string name) {
	std::ifstream file(name);
	std::string line;

	while (std::getline(file, line, '\n')) {
		*inp = std::stof(line);
		inp++;
	}
}
