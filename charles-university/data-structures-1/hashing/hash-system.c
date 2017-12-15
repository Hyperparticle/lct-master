/**
 *
 *
 * @date 12/15/17
 * @author Dan Kondratyuk
 */

#include <stdlib.h>
#include "hash-system.h"
#include "random-gen.h"

#define UNIV_BITS (sizeof(uint32_t) * 8)
#define MASK(bits) (~(~0 << (bits)))

struct tabulation_state tabulation_init(uint32_t blocks, uint32_t hash_size) {
    struct tabulation_state state = { blocks, NULL };
    uint32_t tabulation_table_size = 1u << (UNIV_BITS / state.num_blocks);

    state.table = malloc(state.num_blocks * tabulation_table_size * sizeof(uint32_t));

    for (uint32_t i = 0; i < state.num_blocks; i++) {
        for (uint32_t j = 0; j < tabulation_table_size; j++) {
            state.table[i*state.num_blocks + j] =
                    (uint32_t) rng_next() >> (UNIV_BITS - hash_size);
        }
    }

    return state;
}

uint32_t tabulate(struct tabulation_state state, uint32_t x) {
    uint32_t xor = 0;
    uint32_t x_bits = UNIV_BITS / state.num_blocks;

    for (uint32_t i = 0; i < state.num_blocks; i++) {
        uint32_t split = (x >> (UNIV_BITS - i*x_bits)) & MASK(x_bits);
        xor ^= state.table[i*state.num_blocks + split];
    }

    return xor;
}

struct multiply_shift_state multiply_shift_init(uint32_t hash_size) {
    struct multiply_shift_state state = { hash_size, (uint32_t) rng_next(), (uint32_t) rng_next() };
    return state;
}

uint32_t multiply_shift(struct multiply_shift_state state, uint32_t x) {
    return (state.a*x + state.b) >> (UNIV_BITS - state.hash_size);
}

struct naive_modulo_state naive_modulo_init(uint32_t hash_size) {
    struct naive_modulo_state state = { hash_size };
    return state;
}

uint32_t naive_modulo(struct naive_modulo_state state, uint32_t x) {
    return x >> (UNIV_BITS - state.hash_size);
}
