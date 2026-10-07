#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace branchscore {

inline constexpr std::size_t min_decision_options = 2;
inline constexpr std::size_t legacy_decision_options = 16;
inline constexpr std::size_t max_decision_options = 512;
// Count actual Prefill positions, replacing the image placeholder with visual tokens.
inline constexpr std::size_t max_prefill_positions = 16384;
inline constexpr std::size_t max_request_prefill_positions = 32768;

class DecisionBudgetError : public std::runtime_error {
public:
    DecisionBudgetError(std::string code, std::string message)
        : std::runtime_error(std::move(message)), code_(std::move(code)) {}
    const std::string & code() const noexcept { return code_; }
private:
    std::string code_;
};

} // namespace branchscore
