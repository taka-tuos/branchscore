#include "branchscore/option_scorer.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

int main() {
    try {
        std::vector<branchscore::OptionScore> scores(3);
        scores[0].input_index = 9;
        scores[0].option_id = "low";
        scores[0].sum_logprob = -1000.0;
        scores[1].input_index = 4;
        scores[1].option_id = "high";
        scores[1].sum_logprob = -998.0;
        scores[2].input_index = 7;
        scores[2].option_id = "tie";
        scores[2].sum_logprob = -998.0;

        const auto summary = branchscore::summarize_options(std::move(scores));
        const auto probability_sum =
            summary.scores[0].relative_probability +
            summary.scores[1].relative_probability +
            summary.scores[2].relative_probability;
        if (summary.selected_position != 1 || !summary.tie ||
            std::abs(probability_sum - 1.0) > 1e-12 ||
            !(summary.scores[1].relative_probability >
              summary.scores[0].relative_probability)) {
            throw std::runtime_error("option softmax summary is incorrect");
        }
        std::cout << "selected=" << summary.scores[summary.selected_position].option_id
                  << " probability="
                  << summary.scores[summary.selected_position].relative_probability << '\n';
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
