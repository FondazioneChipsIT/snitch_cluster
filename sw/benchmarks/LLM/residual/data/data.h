#ifndef DATA_H
#define DATA_H
// TCDM pointers to our data
double *x,*y,*out,*h;

// Matrix length (for now only a multiple of ncores, default 8)
#ifndef LEN
#define LEN 8
#endif
// If you exceed TCDM size you get 0 as result
// Max size is 128KB/8bytes (64bits for doubles) = 16k theoretically -> 8K theoretical for each vector!
// but in reality only 4120
#ifndef CONV3x3_LEN
#define CONV3x3_LEN 3
#endif
#endif