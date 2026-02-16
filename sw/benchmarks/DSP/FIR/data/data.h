#ifndef DATA_H
#define DATA_H
// TCDM pointers to our data
float *x,*y,*h;

// Vector length (for now only a multiple of ncores, default 8)
#ifndef LEN
#define LEN 1024
#endif
// If you exceed TCDM size you get 0 as result
// Max size is 128KB/8bytes (64bits for doubles) = 16k theoretically -> 8K theoretical for each vector!
// but in reality only 4120

// Vector length (for now only a multiple of ncores, default 8)
#ifndef FILTER_LEN
#define FILTER_LEN 64
#endif


#endif