/**
 *
 *
 * @date 12/15/17
 * @author Dan Kondratyuk
 */

#include <stdio.h>
#include <stdbool.h>
#include "random-gen.h"
#include "hash-system.h"
#include "hash-scheme.h"

#define RANDOM_SEED 42

static void benchmark_random(struct hash_table table);

int main(int argc, char **argv) {
    rng_init(RANDOM_SEED);

    struct hash_system system = tabulation_system(20, 4);
    struct hash_table table = hash_table_init(system, tabulate, tabulation_init);

//    struct hash_system system = multiply_shift_system(20);
//    struct hash_table table = hash_table_init(system, multiply_shift, multiply_shift_init);

    benchmark_random(table);

    return 0;
}

static void benchmark_random(struct hash_table table) {
    while (table.element_count < table.capacity * 99 / 100) {
        uint32_t x = random_element(table.hash_size);
        long result = insert_cuckoo(&table, x);

        if (result < 0) {
            break;
        } else if (result != 0) { // Don't print duplicates
            double alpha = load_factor(table);
            printf("%lu\t%f\n", result, alpha);
        }
    }
}
