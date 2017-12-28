/**
 * Defines cuckoo and linear probing hashing schemes
 *
 * @date 12/15/17
 * @author Dan Kondratyuk
 */

#ifndef HASHING_HASH_SCHEME_H
#define HASHING_HASH_SCHEME_H

#include <stdint.h>
#include "hash-system.h"

/** 
 * A hash table contains all necessary data/functions to 
 * be able to insert elements
 */
struct hash_table {
    uint32_t hash_size;
    uint32_t capacity;
    uint32_t element_count;
    uint32_t *elements;   // The array of elements
    struct hash_system system[2]; // All hash functions need at most two separate systems
    hash_func hash;
    rebuild_func rebuild; // Function for regenerating a hash system
};

/** Generic hash insert function */
typedef long (*insert_func)(struct hash_table *, uint32_t x);

/** Initializes a hash table with the given system and functions */
struct hash_table hash_table_init(struct hash_system system, hash_func hash, rebuild_func rebuild);

/** Hash insert for cuckoo hashing */
long insert_cuckoo(struct hash_table *table, uint32_t x);

/** Hash insert for linear probing */
long insert_linear_probe(struct hash_table *table, uint32_t x);

/** Returns the load factor of the hash table */
double load_factor(struct hash_table table);

#endif //HASHING_HASH_SCHEME_H
