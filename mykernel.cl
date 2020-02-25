/* D = AxB+C */

__kernel void matrix_calc(
        __global float *outputD,
        const int widthA,
        const int heightA,
        const int widthB,
        const int heightB,
        __global float *A,
        __global float *B,
        __global float *C)
{
    // get index of the work item
    int globalRow = get_global_id(1);
    int globalCol = get_global_id(0);

    float sum = 0.0f;

    for (int i=0; i<widthA; i++) {
        sum += A[globalRow*widthA + i] * B[i*widthB + globalCol];
    }

    sum += C[globalRow*widthB + globalCol]; //add C
 
    // Store the result
    outputD[globalRow*widthB + globalCol] = sum;
}

