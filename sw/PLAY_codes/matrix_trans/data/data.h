#ifndef DATA_H
#define DATA_H


// Matrix rows and cols, only square (for now only a multiple of ncores, default 8)
#ifndef elems
#define elems 64
#endif
/* MAX value 80 * 80 * 8 B = 51200 B = 50 KiB,   src + dst = 100 KiB
             88 * 88 * 8 B = 61952 B = 60.5 KiB, src + dst = 121 KiB so you overwrite results 
             and get 0 as the result, avoid using more than 80 elements*/

// TCDM pointers to our data
double *src,*dst;

// Cycle and performance metrics
uint64_t start_cycle[16], end_cycle[16], total_cycles[16]; // Increase the num if ncores>16
double flop_cycle[16];

#endif