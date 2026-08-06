// Copyright 2026 Chips-IT
//
// ABI compatibility shims.
//
// This cluster has RVD = 0, so FLEN = 32 (see the FLEN formula in
// hw/snitch_cluster/src/snitch_cc.sv) and the software must be built with
// -mabi=ilp32f: with ilp32d the compiler believes the callee-saved FP
// registers are 64 bit wide and spills them with fsd, which is an illegal
// instruction here and traps in the prologue of any function that keeps a
// float alive across a call.
//
// The LLVM install ships a single newlib and a single compiler-rt, both built
// for ilp32d, so linking against them fails with "cannot link object files
// with different floating-point ABI". Only two symbols are actually pulled in
// from those archives, and both are pure integer code, so they are provided
// here and compiled with the project flags.

#include <stddef.h>
#include <stdint.h>

extern "C" {

// newlib's memset. Written through a volatile pointer so that the compiler
// does not recognise the loop and turn it into a call to memset itself.
void *memset(void *s, int c, size_t n) {
    volatile unsigned char *p = (volatile unsigned char *)s;
    unsigned char v = (unsigned char)c;
    while (n--) *p++ = v;
    return s;
}

void *memcpy(void *d, const void *s, size_t n) {
    volatile unsigned char *dp = (volatile unsigned char *)d;
    const unsigned char *sp = (const unsigned char *)s;
    while (n--) *dp++ = *sp++;
    return d;
}

void *memmove(void *d, const void *s, size_t n) {
    volatile unsigned char *dp = (volatile unsigned char *)d;
    const unsigned char *sp = (const unsigned char *)s;
    if (dp < sp) { while (n--) *dp++ = *sp++; }
    else { dp += n; sp += n; while (n--) *--dp = *--sp; }
    return d;
}

// compiler-rt's 64 bit unsigned division helper, shift-and-subtract.
unsigned long long __udivdi3(unsigned long long a, unsigned long long b) {
    if (b == 0) return 0;  // same as compiler-rt: undefined, do not trap

    unsigned long long q = 0;
    unsigned long long r = 0;

    for (int i = 63; i >= 0; i--) {
        r = (r << 1) | ((a >> i) & 1ULL);
        if (r >= b) {
            r -= b;
            q |= (1ULL << i);
        }
    }
    return q;
}

unsigned long long __umoddi3(unsigned long long a, unsigned long long b) {
    if (b == 0) return a;
    return a - __udivdi3(a, b) * b;
}

}  // extern "C"
