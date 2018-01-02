/**
 * Main program execution for benchmarking
 *
 * @date 12/15/17
 * @author Dan Kondratyuk
 */

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <stdbool.h>
#include "random-gen.h"
#include "hash-system.h"
#include "hash-scheme.h"
#include "clock.h"

#define RANDOM_SEED 42
#define HASH_SIZE 20
#define NUM_BLOCKS 4

/** Runs the random test */
static void benchmark_random(struct hash_table table, insert_func insert);

/** Runs the sequential test */
static void benchmark_sequential(enum state_type type);

int main(int argc, char **argv) {
    // Defaults
    bool random = true;
    bool cuckoo = true;
    insert_func insert = insert_cuckoo;
    enum state_type type = tab;
    hash_func hash = tabulate;
    rebuild_func rebuild = tabulation_init;

    // Read command-line switches
    if (argc > 1) {
        for (int i = 0; i < strlen(argv[1]); i++) {
            switch (argv[1][i]) {
                case 'r':
                    random = true;
                    break;
                case 's':
                    random = false;
                    break;
                case 'c':
                    cuckoo = true;
                    insert = insert_cuckoo;
                    break;
                case 'l':
                    cuckoo = false;
                    insert = insert_linear_probe;
                    break;
                case 't':
                    type = tab;
                    hash = tabulate;
                    rebuild = tabulation_init;
                    break;
                case 'm':
                    type = mul_shift;
                    hash = multiply_shift;
                    rebuild = multiply_shift_init;
                    break;
                case 'n':
                    type = naive_mod;
                    hash = naive_modulo;
                    rebuild = naive_modulo_init;
                    break;
                default:
                    break;
            }
        }
    }

    rng_init(RANDOM_SEED);

    if (random) {
        // Random test
        char *insert_type = cuckoo ? "cuckoo" : "linear_probing";
        char *system_type = type == tab ? "tabulation" : type == mul_shift ? "multiply_shift" : "naive_modulo";
        printf("random,%s,%s\n", insert_type, system_type);

        struct hash_system system;

        if (type == tab) {
            system = tabulation_system(HASH_SIZE, NUM_BLOCKS);
        } else if (type == mul_shift) {
            system = multiply_shift_system(HASH_SIZE);
        } else {
            system = naive_modulo_system(HASH_SIZE);
        }

        struct hash_table table = hash_table_init(system, hash, rebuild);
        benchmark_random(table, insert);
    } else {
        // Sequential test
        printf("sequential,linear_probing,%s\n", type == tab ? "tabulation" : "multiply_shift");
        benchmark_sequential(type);
    }

    return 0;
}

static void benchmark_random(struct hash_table table, insert_func insert) {
    uint32_t max_capacity = table.capacity * 99 / 100;
    uint32_t runs = 2048;
    bool done = false;

    while (table.element_count < max_capacity) {
        uint32_t result_sum = 0;

        double begin = clock() * 1e9 / CLOCKS_PER_SEC;
        for (int i = 0; i < runs; i++) {
            uint32_t x = random_element();
            long result = insert(&table, x);
            if (result < 0) {
                done = true;
                break;
            }
            result_sum += result;
        }
        double end = clock() * 1e9 / CLOCKS_PER_SEC;

        if (done) {
            break;
        }

        // Average results over runs
        double alpha = load_factor(table) - ((double) runs / (2 * table.capacity));
        double result = (double) result_sum / (double) runs;
        double time_ns = (double) (end - begin) / (double) runs;
        printf("%f\t%f\t%f\n", alpha, result, time_ns);
    }
}

static void benchmark_sequential(enum state_type type) {
    uint32_t max_hash_size = 29;
    for (uint32_t hash_size = 10; hash_size < max_hash_size; hash_size++) {
        fprintf(stderr, "m = %d", hash_size);

        uint32_t max_inserts = 1u << max_hash_size;
        uint32_t num_inserts = 0;
        uint32_t num_tests = 0;

        while (num_inserts < max_inserts) {
            struct hash_system system;
            struct hash_table table;

            if (type == tab) {
                system = tabulation_system(hash_size, NUM_BLOCKS);
                table = hash_table_init(system, tabulate, tabulation_init);
            } else {
                system = multiply_shift_system(hash_size);
                table = hash_table_init(system, multiply_shift, multiply_shift_init);
            }

            uint32_t element = 1;
            while (table.element_count < table.capacity * 0.89) {
                insert_linear_probe(&table, element);
                element++;
                num_inserts++;
            }

            while (table.element_count < table.capacity * 0.91) {
                long result = insert_linear_probe(&table, element);
                if (result > 0) {
                    printf("%d\t%d\t%lu\n", hash_size, num_tests, result);
                }
                element++;
                num_inserts++;
            }

            free(table.elements);
            if (type == tab) {
                free(table.system->state.tabulation.table);
            }

            num_tests++;
        }

        fprintf(stderr, "\tt = %d\n", num_tests);
    }
}
