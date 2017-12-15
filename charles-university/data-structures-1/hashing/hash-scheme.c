/**
 *
 *
 * @date 12/15/17
 * @author Dan Kondratyuk
 */

#include <stdlib.h>
#include <stdbool.h>
#include "hash-scheme.h"
#include "hash-system.h"

static bool first_build = true;
static bool rebuild_table = false;

static struct tabulation_state tabulation_state[2];
static struct multiply_shift_state multiply_shift_state[2];
static struct naive_modulo_state naive_modulo_state;

struct hash_table hash_table_init(uint32_t hash_size, uint32_t num_blocks) {
    first_build = true;
    rebuild_table = false;

    struct hash_table table = { hash_size, num_blocks, 1u << hash_size, 0, NULL };
    table.elements = calloc(table.capacity, sizeof(uint32_t));
    return table;
}

void insert_cuckoo_tabulation(struct hash_table table, uint32_t x) {
    if (first_build) {
        tabulation_state[0] = tabulation_init(table.num_blocks, table.hash_size);
        tabulation_state[1] = tabulation_init(table.num_blocks, table.hash_size);
        first_build = rebuild_table = false;
    }
    if (rebuild_table) {
        // TODO: make generic, maybe use function pointer arrays


        rebuild_table = false;
    }

    uint32_t pos = tabulate(tabulation_state[0], x);
    for (uint32_t i = 0; i < table.element_count + 1; i++) {
        if (table.elements[pos] == 0) {
            table.elements[pos] = x;
            table.element_count++;
            return;
        }

        uint32_t tmp = table.elements[pos];
        table.elements[pos] = x;
        x = tmp;

        bool h0 = pos == tabulate(tabulation_state[0], x);
        pos = tabulate(tabulation_state[h0 ? 1 : 0], x);
    }

    rebuild_table = true;
    insert_cuckoo_tabulation(table, x);
}

void insert_cuckoo_multiply_shift(struct hash_table table, uint32_t x) {
    if (rebuild_table) {
        multiply_shift_state[0] = multiply_shift_init(table.hash_size);
        multiply_shift_state[1] = multiply_shift_init(table.hash_size);
        rebuild_table = false;
    }

    uint32_t index = multiply_shift(multiply_shift_state[0], x);
}

void insert_linear_probe_tabulation(struct hash_table table, uint32_t x) {
    if (rebuild_table) {
        tabulation_state[0] = tabulation_init(table.num_blocks, table.hash_size);
        rebuild_table = false;
    }

    uint32_t index = tabulate(tabulation_state[0], x);
}

void insert_linear_probe_multiply_shift(struct hash_table table, uint32_t x) {
    if (rebuild_table) {
        multiply_shift_state[0] = multiply_shift_init(table.hash_size);
        rebuild_table = false;
    }

    uint32_t index = multiply_shift(multiply_shift_state[0], x);
}

void insert_linear_probe_naive_modulo(struct hash_table table, uint32_t x) {
    if (rebuild_table) {
        naive_modulo_state = naive_modulo_init(table.hash_size);
        rebuild_table = false;
    }

    uint32_t index = multiply_shift(multiply_shift_state[0], x);
}

double load_factor(struct hash_table table) {
    return (double) table.element_count / (double) table.capacity;
}
