#ifndef DATA_H
#define DATA_H
// TCDM pointers to our data
float *x,*y,*out;

/* Square matrix dimension (LEN x LEN, must be a multiple of ncores) */
#ifndef LEN
#define LEN 64
#endif 
#endif