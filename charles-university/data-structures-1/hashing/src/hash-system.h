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
    uint32_t a, b;      // Random integers for multiply shift hashing
};

enum state_type {
    tab = 0, mul_shift = 1, naive_mod = 2
};

struct hash_system {
    uint32_t hash_size; // Hash table size = 2 ^ hash_bits
    enum state_type type;
    union {
        struct tabulation_state tabulation;
        struct multiply_shift_state multiply_shift;
    } state;
};

typedef uint32_t (*hash_func)(struct hash_system *, uint32_t);

typedef void (*rebuild_func)(struct hash_system *);

struct hash_system tabulation_system(uint32_t hash_size, uint32_t num_blocks);

struct hash_system multiply_shift_system(uint32_t hash_size);

struct hash_system naive_modulo_system(uint32_t hash_size);

void tabulation_init(struct hash_system *system);

uint32_t tabulate(struct hash_system *system, uint32_t x);

void multiply_shift_init(struct hash_system *system);

uint32_t multiply_shift(struct hash_system *system, uint32_t x);

void naive_modulo_init(struct hash_system *system);

uint32_t naive_modulo(struct hash_system *system, uint32_t x);

uint32_t random_element();

#endif //HASHING_HASH_SYSTEM_H
