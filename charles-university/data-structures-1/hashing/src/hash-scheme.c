/**
 *
 *
 * @date 12/15/17
 * @author Dan Kondratyuk
 */

#include <stdlib.h>
#include <stdbool.h>
#include "hash-scheme.h"

#define MAX_REHASHES 4

static long rebuild_cuckoo(struct hash_table *table, uint32_t *rehashes);

static long insert_cuckoo_rec(struct hash_table *table, uint32_t x, uint32_t *rehashes);

struct hash_table hash_table_init(struct hash_system system, hash_func hash, rebuild_func rebuild) {
    uint32_t capacity = 1u << system.hash_size;

    struct hash_table table = {
            .hash_size = system.hash_size,
            .capacity = capacity,
            .element_count = 0,
            .elements = calloc(capacity, sizeof(uint32_t)),
            .system = {system, system},
            .hash = hash,
            .rebuild = rebuild
    };

    table.rebuild(&table.system[0]);
    table.rebuild(&table.system[1]);

    return table;
}

long insert_cuckoo(struct hash_table *table, uint32_t x) {
    uint32_t rehashes = 0;

    // Check for duplicates
    uint32_t pos0 = table->hash(&table->system[0], x);
    uint32_t pos1 = table->hash(&table->system[1], x);
    if (table->elements[pos0] == x || table->elements[pos1] == x) {
        return 0;
    }

    return insert_cuckoo_rec(table, x, &rehashes);
}

static long insert_cuckoo_rec(struct hash_table *table, uint32_t x, uint32_t *rehashes) {
    if (*rehashes > MAX_REHASHES) { return -1; }

    long accesses = 0;

    uint32_t pos = table->hash(&table->system[0], x);
    uint32_t pos2 = table->hash(&table->system[1], x);

    if (table->elements[pos] == x || table->elements[pos2] == x) {
        return accesses;
    }

    for (uint32_t i = 0; i < table->element_count + 1; i++) {
        accesses++;

        if (table->elements[pos] == 0) {
            table->elements[pos] = x;
            table->element_count++;
            return accesses;
        }

        uint32_t tmp = table->elements[pos];
        table->elements[pos] = x;
        x = tmp;

        bool h0 = pos == table->hash(&table->system[0], x);
        pos = table->hash(&table->system[h0 ? 1 : 0], x);
    }

    // Rebuild table
    *rehashes += 1;
    accesses += rebuild_cuckoo(table, rehashes);
    accesses += insert_cuckoo_rec(table, x, rehashes);

    if (*rehashes > MAX_REHASHES) { return -1; }
    return accesses;
}

static long rebuild_cuckoo(struct hash_table *table, uint32_t *rehashes) {
    if (*rehashes > MAX_REHASHES) { return -1; }

    long accesses = 0;

    struct hash_system system = table->system[0];
    uint32_t *old_tabulation_table = system.type == tab ? system.state.tabulation.table : NULL;

    struct hash_table new_table = hash_table_init(system, table->hash, table->rebuild);
    new_table.rebuild(&new_table.system[0]);
    new_table.rebuild(&new_table.system[1]);

    for (uint32_t i = 0; i < table->capacity; i++) {
        uint32_t element = table->elements[i];
        if (element != 0) {
            accesses += insert_cuckoo_rec(&new_table, element, rehashes);
        }
    }

    free(table->elements);
    free(old_tabulation_table);
    *table = new_table;

    if (*rehashes > MAX_REHASHES) { return -1; }
    return accesses;
}

long insert_linear_probe(struct hash_table *table, uint32_t x) {
    long accesses = 1;

    uint32_t pos = table->hash(&table->system[0], x);
    while (table->elements[pos] != 0 && table->elements[pos] != x) {
        pos = (pos + 1) % table->capacity;
        accesses++;
    }

    table->elements[pos] = x;
    table->element_count++;

    return accesses;
}

double load_factor(struct hash_table table) {
    return (double) table.element_count / (double) table.capacity;
}
