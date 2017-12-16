/**
 *
 *
 * @date 12/15/17
 * @author Dan Kondratyuk
 */

#ifndef HASHING_RANDOM_GEN_H
#define HASHING_RANDOM_GEN_H

#include <stdint.h>

void rng_init(uint32_t seed);

uint64_t rng_next();

#endif //HASHING_RANDOM_GEN_H
