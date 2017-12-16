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

    struct hash_system system = tabulation_system(20, 2);
    struct hash_table table = hash_table_init(system, tabulate, tabulation_init);

//    struct hash_system system = muliply_shift_system(20);
//    struct hash_table table = hash_table_init(system, multiply_shift, multiply_shift_init);

    while (table.element_count < table.capacity / 2) {
        uint32_t x = random_element(table.hash_size);
        insert_cuckoo(&table, x);
    }

    return 0;
}