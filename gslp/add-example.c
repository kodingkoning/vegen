#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int32_t A [8];
int32_t B [8];
int32_t C [4];

void vector_add()
{
    C[0] = A[0] + B[0];
    C[1] = A[1] + B[1];
    C[2] = A[2] + 9;
    C[3] = A[3] + 11;
}

