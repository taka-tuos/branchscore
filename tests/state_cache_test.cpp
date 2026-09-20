#include "branchscore/state_cache.hpp"

#include <iostream>

int main() {
    branchscore::TextModelConfig config;
    config.context_length = 32;
    config.block_count = 6;
    config.shared_kv_layers = 2;
    config.head_count_kv = 1;
    config.key_length = 8;
    config.value_length = 8;
    config.key_length_swa = 4;
    config.value_length_swa = 4;
    config.sliding_window_pattern = {true, false, true, false, true, false};

    branchscore::BackendContext backend("cpu");
    branchscore::StateCache cache(config, 12, backend);
    bool valid = true;
    valid &= cache.capacity() == 12;
    valid &= cache.source_layer(0) == 0;
    valid &= cache.source_layer(4) == 2;
    valid &= cache.source_layer(5) == 3;
    valid &= cache.key(4) == cache.key(2);
    valid &= cache.value(5) == cache.value(3);
    valid &= cache.key(0)->ne[0] == 4;
    valid &= cache.key(1)->ne[0] == 8;

    cache.freeze_prefix(7);
    valid &= cache.cursor() == 7;

    if (!valid) std::cerr << "state-cache invariant failed\n";
    return valid ? 0 : 1;
}
