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

struct hash_system tabulation_system(uint32_t hash_size, uint32_t num_blocks) {
    struct hash_system system = {
            .hash_size = hash_size,
            .type = tab,
            .state.tabulation = {.num_blocks = num_blocks, .table = NULL}
    };
    return system;
}

struct hash_system muliply_shift_system(uint32_t hash_size) {
    struct hash_system system = {
            .hash_size = hash_size,
            .type = mul_shift,
            .state.multiply_shift = {.a = 0, .b = 0}
    };
    return system;
}

struct hash_system naive_modulo_system(uint32_t hash_size) {
    struct hash_system system = {
            .hash_size = hash_size,
            .type = naive_mod,
            .state = {0}
    };
    return system;
}

void tabulation_init(struct hash_system *system) {
    struct tabulation_state *state = &system->state.tabulation;

    uint32_t tabulation_table_size = 1u << (UNIV_BITS / state->num_blocks);

    state->table = malloc(state->num_blocks * tabulation_table_size * sizeof(uint32_t));

    for (uint32_t i = 0; i < state->num_blocks; i++) {
        for (uint32_t j = 0; j < tabulation_table_size; j++) {
            uint32_t next_random = (uint32_t) rng_next() >> (UNIV_BITS - system->hash_size);
            state->table[i * state->num_blocks + j] = next_random;
        }
    }
}

uint32_t tabulate(struct hash_system *system, uint32_t x) {
    struct tabulation_state *state = &system->state.tabulation;

    uint32_t xor = 0;
    uint32_t x_bits = UNIV_BITS / state->num_blocks;

    for (uint32_t i = 0; i < state->num_blocks; i++) {
        uint32_t split = (x >> (UNIV_BITS - (i - 1) * x_bits)) & MASK(x_bits);
        xor ^= state->table[i * state->num_blocks + split];
    }

    return xor;
}

void multiply_shift_init(struct hash_system *system) {
    struct multiply_shift_state *state = &system->state.multiply_shift;
    state->a = (uint32_t) rng_next();
    state->b = (uint32_t) rng_next();
}

uint32_t multiply_shift(struct hash_system *system, uint32_t x) {
    struct multiply_shift_state *state = &system->state.multiply_shift;
    return (state->a * x + state->b) >> (UNIV_BITS - system->hash_size);
}

void naive_modulo_init(struct hash_system *system) {}

uint32_t naive_modulo(struct hash_system *system, uint32_t x) {
    return x >> (UNIV_BITS - system->hash_size);
}

uint32_t random_element(uint32_t hash_size) {
    return (uint32_t) rng_next() >> (UNIV_BITS - hash_size);
}
