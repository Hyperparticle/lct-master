/**
 *
 *
 * @date 12/15/17
 * @author Dan Kondratyuk
 */

#ifndef HASHING_HASH_SYSTEM_H
#define HASHING_HASH_SYSTEM_H

#include <stdint.h>

struct tabulation_state {
    uint32_t num_blocks; // Split x into this many blocks
    uint32_t *table;     // Lookup table for tabulation hashing
};

struct multiply_shift_state {
    uint32_t hash_size; // Hash table size = 2 ^ hash_bits
    uint32_t a, b;      // Random integers for multiply shift hashing
};

struct naive_modulo_state {
    uint32_t hash_size; // Hash table size = 2 ^ hash_bits
};

struct tabulation_state tabulation_init(uint32_t blocks, uint32_t hash_size);
uint32_t tabulate(struct tabulation_state state, uint32_t x);

struct multiply_shift_state multiply_shift_init(uint32_t hash_size);
uint32_t multiply_shift(struct multiply_shift_state state, uint32_t x);

struct naive_modulo_state naive_modulo_init(uint32_t hash_size);
uint32_t naive_modulo(struct naive_modulo_state state, uint32_t x);

#endif //HASHING_HASH_SYSTEM_H
