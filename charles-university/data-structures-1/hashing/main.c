/**
 *
 *
 * @date 12/15/17
 * @author Dan Kondratyuk
 */

#include "random-gen.h"
#include "hash-system.h"
#include "hash-scheme.h"

int main(int argc, char **argv) {
    rng_init(42);

    struct hash_system system = tabulation_system(20, 4);

    struct hash_table table = hash_table_init(system, tabulate, tabulation_init);

    for (uint32_t i = 0; i < table.capacity / 2; i++) {
        uint32_t x = random_element(table.hash_size);
        insert_cuckoo(&table, x);
    }

    return 0;
}