/**
 * This is the xoroshiro128+ random generator, designed in 2016 by David Blackman
 * and Sebastiano Vigna, distributed under the CC-0 license. For more details,
 * see http://vigna.di.unimi.it/xorshift/.
 *
 * @date 12/15/17
 * @author Dan Kondratyuk
 */

#ifndef HASHING_RANDOM_GEN_H
#define HASHING_RANDOM_GEN_H

#include <stdint.h>

/** Initializes the global RNG with a seed */
void rng_init(uint32_t seed);

/** Generates a random 64-bit integer */
uint64_t rng_next();

#endif //HASHING_RANDOM_GEN_H
