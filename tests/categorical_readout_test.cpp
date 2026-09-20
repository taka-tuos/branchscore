#include "branchscore/categorical_readout.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

int main() {
    try {
        const auto summary = branchscore::summarize_categorical(
            {-1000.0F, -998.0F, -998.0F});
        const auto probability_sum =
            summary.relative_probabilities[0] +
            summary.relative_probabilities[1] +
            summary.relative_probabilities[2];
        if (summary.selected_index != 1 || !summary.exact_tie ||
            std::abs(probability_sum - 1.0) > 1e-12 ||
            !(summary.relative_probabilities[1] >
              summary.relative_probabilities[0])) {
            throw std::runtime_error("categorical softmax/tie contract is incorrect");
        }

        bool rejected_nonfinite = false;
        try {
            (void) branchscore::summarize_categorical(
                {0.0F, std::numeric_limits<float>::quiet_NaN()});
        } catch (const std::runtime_error &) {
            rejected_nonfinite = true;
        }
        if (!rejected_nonfinite) {
            throw std::runtime_error("non-finite categorical logits were accepted");
        }

        std::cout << "selected=" << summary.selected_index
                  << " tie=" << (summary.exact_tie ? "true" : "false") << '\n';
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
