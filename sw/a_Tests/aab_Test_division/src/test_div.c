// Luca Colombo Chips-IT 2025
// Simple division test by core 0

/* DOES NOT WORK! illegal instruction fdiv, but if you don't use ASM but write a division in C
   it will work, the compiler will write it using other instructions instead of fdiv!
*/

#include "snrt.h"

__attribute__((noinline)) void division(double a_val, double b_val, double *div_val){
    // Associa registri FP fisici
    register double ft0_val asm("ft0") = a_val;  // ft0 = a
    register double ft1_val asm("ft1") = b_val;  // ft1 = b
    register double ft2_val asm("ft2");          // ft2 = risultato

    asm volatile(
        "fdiv.d ft2, ft1, ft0\n"  // ft2 = ft1 / ft0
        : "=f"(ft2_val)           // output: copia ft2_val da ft2
        : "f"(ft0_val), "f"(ft1_val)  // input
        : "memory"
    );

    *div_val = ft2_val;  // copia in memoria

}

int main(){
    // Core ID
    uint32_t core_idx = snrt_cluster_core_idx();

    if(core_idx==0){

        double a,b,div;      
        a = 1.35;
        b = 2.67;

        division(a,b,&div);
        printf("a: %f and b: %f\n", a,b);
        printf("Division result: %f\n", div);
    }

    return 0;
}