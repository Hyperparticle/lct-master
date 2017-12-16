/**
 *
 *
 * @date 12/15/17
 * @author Dan Kondratyuk
 */

#include <stdlib.h>
#include "hash-scheme.h"

//static bool first_build = true;
//static bool rebuild_table = false;

struct hash_table hash_table_init(struct hash_system system, hash_func hash, rebuild_func rebuild) {
    uint32_t capacity = 1u << system.hash_size;

    struct hash_table table = {
            .capacity = capacity,
            .element_count = 0,
            .elements = calloc(capacity, sizeof(uint32_t)),
            .system = system,
            .hash = hash,
            .rebuild = rebuild
    };

    return table;
}

void insert_cuckoo(struct hash_table *table, uint32_t x) {
    table->rebuild(&table->system);
    uint32_t i = table->hash(&table->system, x);

//    if (first_build || rebuild_table) {
//        first_build = false;
//        tabulation_state[0] = tabulation_init(table->num_blocks, table->hash_size);
//        tabulation_state[1] = tabulation_init(table->num_blocks, table->hash_size);
//    }
//
//    if (rebuild_table) {
//        rebuild_table = false;
//        // TODO: make generic, maybe use function pointer arrays
//        // TODO: recursion needs work
//        struct hash_table new_table = hash_table_init(table->hash_size, table->num_blocks);
//        for (uint32_t i = 0; i < table->capacity; i++) {
//            uint32_t element = table->elements[i];
//            if (element != 0) {
//                insert_cuckoo_tabulation(&new_table, element);
//            }
//        }
//
//        free(table->elements);
//        *table = new_table;
//    }
//
//    uint32_t pos = tabulate(tabulation_state[0], x);
//    for (uint32_t i = 0; i < table->element_count + 2; i++) {
//        if (table->elements[pos] == 0) {
//            table->elements[pos] = x;
//            table->element_count++;
//            return;
//        } else if (table->elements[pos] == x) {
//            return;
//        }
//
//        uint32_t tmp = table->elements[pos];
//        table->elements[pos] = x;
//        x = tmp;
//
//        bool h0 = pos == tabulate(tabulation_state[0], x);
//        pos = tabulate(tabulation_state[h0 ? 1 : 0], x);
//    }
//
//    rebuild_table = true;
//    insert_cuckoo_tabulation(table, x);
}

void insert_linear_probe(struct hash_table *table, uint32_t x) {

}

double load_factor(struct hash_table table) {
    return (double) table.element_count / (double) table.capacity;
}
