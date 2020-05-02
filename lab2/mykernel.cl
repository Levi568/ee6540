__kernel void Pi(int numIterations, __global float *outputPi,__local float* local_result, int numWorkers){
    __private const uint global_id = get_global_id(0);
    __private const uint local_id = get_local_id(0);
    __private const uint offset = numIterations*global_id*2; 
    __private float sum = 0.0f;
    
    if (global_id == 0){
        for (int i = 0; i < numWorkers; i++){
            local_result[i] = 0.0f;
        }
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    for (int i=0; i<numIterations; i++) {
        if (i % 2 == 0){
            sum += 1. / (1 + 2*i + offset);
        }
        else{
            sum -= 1. / (1 + 2*i + offset);
        }
    }

    local_result[global_id] = sum;    

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id == 0){
        outputPi[0] = 0;
        for (int i = 0; i < numWorkers; i++){
            outputPi[0] += local_result[i]; 
        }

        outputPi[0] *= 4;
    }    
}
