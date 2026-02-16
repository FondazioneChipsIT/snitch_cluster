#ifndef DATA_H
#define DATA_H
// TCDM pointers to our data
double *x,*y,*out;

// Matrix length (for now only a multiple of ncores, default 8)
#ifndef LEN
#define LEN 64
#endif 
#endif