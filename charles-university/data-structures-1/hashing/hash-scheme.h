/**
 *
 *
 * @date 12/15/17
 * @author Dan Kondratyuk
 */

#ifndef HASHING_HASH_SCHEME_H
#define HASHING_HASH_SCHEME_H

#include <stdint.h>

struct hash_table {
    uint32_t hash_size;
    uint32_t num_blocks;
    uint32_t capacity;
    uint32_t element_count;
    uint32_t *elements;
//    uint32_t state[8];
};

struct hash_table hash_table_init(uint32_t hash_size, uint32_t num_blocks);

void insert_cuckoo_tabulation(struct hash_table table, uint32_t x);

#endif //HASHING_HASH_SCHEME_H
