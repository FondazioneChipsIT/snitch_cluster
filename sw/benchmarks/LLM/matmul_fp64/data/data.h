#ifndef DATA_H
#define DATA_H

/* Square matrix dimension */
/* Max size: 24. 24*24*3(num of matrices) * 64 bits/element --> 110,592 bits, so 110KiB*/
#ifndef elems
#define elems 128
#endif

/* TCDM pointers  */
double *mat_a; 
double *mat_b; 
double *dst; 

#endif 