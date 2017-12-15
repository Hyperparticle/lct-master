/**
 *
 *
 * @date 12/15/17
 * @author Dan Kondratyuk
 */

#include <stdio.h>
#include "random-gen.h"
#include "hash-system.h"
#include "hash-scheme.h"

int main(int argc, char **argv) {
    rng_init(42);

    uint32_t hash_size = 20;
    uint32_t num_blocks = 4;

    struct hash_table table = hash_table_init(hash_size, num_blocks);
    insert_cuckoo_tabulation(table, 1234);

    return 0;
}