/**
 * Defines tabulation, multiply-shift, and naive modulo hashing systems
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
    uint32_t a; // Random integers for multiply shift hashing
};

enum state_type {
    tab = 0, mul_shift = 1, naive_mod = 2
};

/** All necessary information to construct a generic hash system */
struct hash_system {
    uint32_t hash_size_bits; // Hash table size = 2 ^ hash_bits
    enum state_type type;
    union {
        struct tabulation_state tabulation;
        struct multiply_shift_state multiply_shift;
    } state;
};

/** A generic hash function */
typedef uint32_t (*hash_func)(struct hash_system *, uint32_t);

/** A generic function for rebuilding a hash system (i.e., generating new random parameters) */
typedef void (*rebuild_func)(struct hash_system *);

/** Creates a tabulation-based hash system */
struct hash_system tabulation_system(uint32_t hash_size, uint32_t num_blocks);

/** Creates a multiply-shift-based hash system */
struct hash_system multiply_shift_system(uint32_t hash_size);

/** Creates a naive modulo hash system */
struct hash_system naive_modulo_system(uint32_t hash_size);

/** Initializes the tabulation system by constructing the tabulation tables */
void tabulation_init(struct hash_system *system);

/** The tabulation hashing hash function */
uint32_t tabulate(struct hash_system *system, uint32_t x);

/** Initializes the multiply-shift hash system */
void multiply_shift_init(struct hash_system *system);

/** The multiply-shift hash function */
uint32_t multiply_shift(struct hash_system *system, uint32_t x);

/** Initializes the naive modulo hash system */
void naive_modulo_init(struct hash_system *system);

/** The naive modulo hash function */
uint32_t naive_modulo(struct hash_system *system, uint32_t x);

/** Generates a random 32-bit element */
uint32_t random_element();

#endif //HASHING_HASH_SYSTEM_H
