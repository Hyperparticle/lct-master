/**
 *
 *
 * @date 12/15/17
 * @author Dan Kondratyuk
 */

#ifndef HASHING_HASH_SCHEME_H
#define HASHING_HASH_SCHEME_H

#include <stdint.h>
#include "hash-system.h"

struct hash_table {
    uint32_t hash_size;
    uint32_t capacity;
    uint32_t element_count;
    uint32_t *elements;
    struct hash_system system[2];
    hash_func hash;
    rebuild_func rebuild;
};

typedef long (*insert_func)(struct hash_table *, uint32_t x);

struct hash_table hash_table_init(struct hash_system system, hash_func hash, rebuild_func rebuild);

long insert_cuckoo(struct hash_table *table, uint32_t x);

long insert_linear_probe(struct hash_table *table, uint32_t x);

double load_factor(struct hash_table table);

#endif //HASHING_HASH_SCHEME_H
