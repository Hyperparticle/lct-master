/**
 *
 *
 * @date 12/15/17
 * @author Dan Kondratyuk
 */

#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include "random-gen.h"
#include "hash-system.h"
#include "hash-scheme.h"

#define RANDOM_SEED 42
#define HASH_SIZE 20
#define NUM_BLOCKS 4

static void benchmark_random(struct hash_table table);
static void print_usage();

int main(int argc, char **argv) {
    rng_init(RANDOM_SEED);

//    if (argc != 1) {
//        print_usage();
//    }
//
//    if (strcmp(argv[0], "-n") == 0) {
//
//    }


    printf("Tabulation-Cuckoo\n");
    struct hash_system system = tabulation_system(HASH_SIZE, NUM_BLOCKS);
    struct hash_table table = hash_table_init(system, tabulate, tabulation_init);

//    struct hash_system system = multiply_shift_system(20);
//    struct hash_table table = hash_table_init(system, multiply_shift, multiply_shift_init);

    benchmark_random(table);

    printf("\n");

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

static void print_usage() {
    printf("Usage:\n");
    printf("hashing [-r|-s] [-c|-l] [-t|-m|-n]\n");
}
