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

//static bool first_build = true;
//static bool rebuild_table = false;

static void rebuild_cuckoo(struct hash_table *table) {
    struct hash_system system = table->system[0];
    uint32_t *old_tabulation_table = system.type == tab ? system.state.tabulation.table : NULL;

    struct hash_table new_table = hash_table_init(system, table->hash, table->rebuild);
    new_table.rebuild(&new_table.system[0]);
    new_table.rebuild(&new_table.system[1]);

    for (uint32_t i = 0; i < table->capacity; i++) {
        uint32_t element = table->elements[i];
        if (element != 0) {
            insert_cuckoo(&new_table, element);
        }
    }

    free(table->elements);
    free(old_tabulation_table);
    *table = new_table;
}

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

void insert_cuckoo(struct hash_table *table, uint32_t x) {
    uint32_t pos = table->hash(&table->system[0], x);
    for (uint32_t i = 0; i < table->element_count + 1; i++) {
        if (table->elements[pos] == 0) {
            table->elements[pos] = x;
            table->element_count++;
            return;
        } else if (table->elements[pos] == x) {
            return;
        }

        uint32_t tmp = table->elements[pos];
        table->elements[pos] = x;
        x = tmp;

        bool h0 = pos == table->hash(&table->system[0], x);
        pos = table->hash(&table->system[h0 ? 1 : 0], x);
    }

    rebuild_cuckoo(table);
    insert_cuckoo(table, x);
}

void insert_linear_probe(struct hash_table *table, uint32_t x) {

}

double load_factor(struct hash_table table) {
    return (double) table.element_count / (double) table.capacity;
}
