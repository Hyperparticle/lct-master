/**
 * This is the xoroshiro128+ random generator, designed in 2016 by David Blackman
 * and Sebastiano Vigna, distributed under the CC-0 license. For more details,
 * see http://vigna.di.unimi.it/xorshift/.
 *
 * @date 12/15/17
 * @author Dan Kondratyuk
 */

#include "random-gen.h"


static uint64_t rng_state[2];

static uint64_t rng_rotl(uint64_t x, uint32_t k) {
    return (x << k) | (x >> (64 - k));
}

void rng_init(uint32_t seed) {
    rng_state[0] = seed * 0xdeadbeef;
    rng_state[1] = seed ^ 0xc0de1234;
    for (int i = 0; i < 100; i++)
        rng_next();
}

uint64_t rng_next() {
    uint64_t s0 = rng_state[0], s1 = rng_state[1];
    uint64_t result = s0 + s1;
    s1 ^= s0;
    rng_state[0] = rng_rotl(s0, 55) ^ s1 ^ (s1 << 14);
    rng_state[1] = rng_rotl(s1, 36);
    return result;
}
