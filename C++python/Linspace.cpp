// Created by Junho HA on 2024-07-18.
// Ref: https://gist.github.com/pmav99/d124872c879f3e9fa51e

#include <cmath>
#include <iostream>
#include <vector>

class Linspace {
public:
    // Static method to generate linspace values
    static std::vector<double> generate(double min, double max, int n) {
        std::vector<double> result;
        result.reserve(n);

        double step = (max - min) / (n - 1);

        for (int i = 0; i < n; ++i) {
            result.push_back(min + i * step);
        }

        // Ensure the last element is exactly max
        result[n-1] = max;

        return result;
    }
};

int main() {
    // Linspace class를 이용하여 -10에서 10까지 400개의 점 생성
    std::vector<double> result = Linspace::generate(-10, 10, 400);

    // 결과 확인을 위해 적절한 위치마다 줄바꿈
    for (int i = 0; i < result.size(); ++i) {
        std::cout << result[i] << "[" << i << "]" << " ";
        if ((i + 1) % 10 == 0) {  // 10개의 값마다 줄바꿈
            std::cout << "\n";
        }
    }
    std::cout << std::endl;

    return 0;
}