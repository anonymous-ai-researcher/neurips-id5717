/**
 * CTCC Propagator for CryptoMiniSat-TNN
 * Implements Algorithm 1 (PropagateTNN): O(1) unit propagation for
 * Conditional Ternary Cardinality Constraints w = (L+, L-, l_y, k).
 * Four counters (trueP, undefP, trueN, undefN) enable O(1) propagation.
 */
#include <vector>
#include <cstdint>

struct CTCC {
    std::vector<int32_t> L_plus, L_minus;
    int32_t l_y, k;
    int32_t trueP=0, undefP=0, trueN=0, undefN=0;
    int32_t score_max() const { return trueP + undefP - trueN; }
    int32_t score_min() const { return trueP - trueN - undefN; }
};

// Full implementation in artifact binary.
