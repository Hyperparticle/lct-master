/**
 *
 *
 * @date 12/15/17
 * @author Dan Kondratyuk
 */

#include <stdio.h>
#include <stdlib.h>
#include "random-gen.h"
#include "hash-system.h"
#include "hash-scheme.h"

#define RANDOM_SEED 42
#define HASH_SIZE 20
#define NUM_BLOCKS 4

static void benchmark_random(struct hash_table table, insert_func insert);

static void benchmark_sequential(enum state_type type);

int main(int argc, char **argv) {
    rng_init(RANDOM_SEED);

    struct hash_system system;
    struct hash_table table;

    printf("RANDOM_TEST\n");
    printf("cuckoo_tabulation\n");
    system = tabulation_system(HASH_SIZE, NUM_BLOCKS);
    table = hash_table_init(system, tabulate, tabulation_init);
    benchmark_random(table, insert_cuckoo);
    printf("\n");

    printf("cuckoo_multiply_shift\n");
    system = multiply_shift_system(HASH_SIZE);
    table = hash_table_init(system, multiply_shift, multiply_shift_init);
    benchmark_random(table, insert_cuckoo);
    printf("\n");

    printf("linear_probing_tabulation\n");
    system = tabulation_system(HASH_SIZE, NUM_BLOCKS);
    table = hash_table_init(system, tabulate, tabulation_init);
    benchmark_random(table, insert_linear_probe);
    printf("\n");

    printf("linear_probing_multiply_shift\n");
    system = multiply_shift_system(HASH_SIZE);
    table = hash_table_init(system, multiply_shift, multiply_shift_init);
    benchmark_random(table, insert_linear_probe);
    printf("\n");

    printf("linear_probing_naive_modulo\n");
    system = naive_modulo_system(HASH_SIZE);
    table = hash_table_init(system, naive_modulo, naive_modulo_init);
    benchmark_random(table, insert_linear_probe);
    printf("\n");

    printf("SEQUENTIAL_TEST\n");
    printf("linear_probing_tabulation\n");
    benchmark_sequential(tab);
    printf("\n");

    printf("linear_probing_multiply_shift\n");
    benchmark_sequential(mul_shift);
    printf("\n");

    return 0;
}

static void benchmark_random(struct hash_table table, insert_func insert) {
    while (table.element_count < table.capacity * 9 / 10) {
        uint32_t x = random_element(table.hash_size);
        long result = insert(&table, x);

        if (result < 0) {
            break;
        } else if (result != 0) { // Don't print duplicates
            double alpha = load_factor(table);
            printf("%lu\t%f\n", result, alpha);
        }
    }
}

static void benchmark_sequential(enum state_type type) {
    struct hash_system system;
    struct hash_table table;

    for (uint32_t hash_size = 15; hash_size < 30; hash_size++) {
        system = type == tab ? tabulation_system(hash_size, NUM_BLOCKS) : multiply_shift_system(hash_size);
        table = hash_table_init(system, multiply_shift, multiply_shift_init);
        uint32_t element = 1;

        while (table.element_count < table.capacity * 0.89) {
            insert_linear_probe(&table, element);
            element++;
        }

        while (table.element_count < table.capacity * 0.91) {
            long result = insert_linear_probe(&table, element);
            if (result > 0) {
                printf("%d\t%lu\t\n", table.capacity, result);
            }
            element++;
        }

        free(table.elements);
        if (type == tab) {
            free(table.system->state.tabulation.table);
        }
    }
}
