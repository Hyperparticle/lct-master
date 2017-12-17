/**
 *
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

#define RANDOM_SEED 42
#define HASH_SIZE 20
#define NUM_BLOCKS 4

static void benchmark_random(struct hash_table table, insert_func insert);

static void benchmark_sequential(enum state_type type);

int main(int argc, char **argv) {
    // Defaults
    bool random = true;
    bool cuckoo = true;
    insert_func insert = insert_cuckoo;
    enum state_type type = tab;
    hash_func hash = tabulate;
    rebuild_func rebuild = tabulation_init;

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
        printf("sequential,linear_probing,%s\n", type == tab ? "tabulation" : "multiply_shift");
        benchmark_sequential(type);
    }

    return 0;
}

static void benchmark_random(struct hash_table table, insert_func insert) {
    while (table.element_count < table.capacity * 99 / 100) {
        uint32_t x = random_element();
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

    for (int run = 0; run < 5; run++) {
        for (uint32_t hash_size = 15; hash_size < 30; hash_size++) {
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
            }

            while (table.element_count < table.capacity * 0.91) {
                long result = insert_linear_probe(&table, element);
                if (result > 0) {
                    printf("%d\t%lu\n", hash_size, result);
                }
                element++;
            }

            free(table.elements);
            if (type == tab) {
                free(table.system->state.tabulation.table);
            }
        }
    }
}
