#include "branchscore/json.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

int main() {
    try {
        const auto value = branchscore::json::parse(
            R"({"text":"日本語\n\uD83D\uDE80","ok":true,"values":[1,-2.5,null]})");
        if (value.find("text") == nullptr ||
            value.find("text")->string() != "日本語\n🚀" ||
            !value.find("ok")->boolean() ||
            std::fabs(value.find("values")->array().at(1).number() + 2.5) > 1e-12) {
            throw std::runtime_error("JSON parse result mismatch");
        }

        const auto encoded = branchscore::json::stringify(value);
        const auto round_trip = branchscore::json::parse(encoded);
        if (round_trip.find("text")->string() != "日本語\n🚀") {
            throw std::runtime_error("JSON round trip mismatch");
        }

        bool duplicate_rejected = false;
        try {
            static_cast<void>(branchscore::json::parse(R"({"duplicate":1,"duplicate":2})"));
        } catch (const std::exception &) {
            duplicate_rejected = true;
        }
        if (!duplicate_rejected) throw std::runtime_error("duplicate keys were accepted");

        std::cout << "json checks passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "json test: " << error.what() << '\n';
        return 1;
    }
}
