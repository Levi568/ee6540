/***** GCL Generated File *********************/
/* Automatically generated file, do not edit! */
/**********************************************/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <dispatch/dispatch.h>
#include <OpenCL/opencl.h>
#include <OpenCL/gcl_priv.h>
#include "mykernel.cl.h"

static void initBlocks(void);

// Initialize static data structures
static block_kernel_pair pair_map[1] = {
      { NULL, NULL }
};

static block_kernel_map bmap = { 0, 1, initBlocks, pair_map };

// Block function
void (^matrix_calc_kernel)(const cl_ndrange *ndrange, cl_float* outputD, cl_int widthA, cl_int heightA, cl_int widthB, cl_int heightB, cl_float* A, cl_float* B, cl_float* C) =
^(const cl_ndrange *ndrange, cl_float* outputD, cl_int widthA, cl_int heightA, cl_int widthB, cl_int heightB, cl_float* A, cl_float* B, cl_float* C) {
  int err = 0;
  cl_kernel k = bmap.map[0].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[0].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel matrix_calc does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, outputD, &kargs);
  err |= gclSetKernelArgAPPLE(k, 1, sizeof(widthA), &widthA, &kargs);
  err |= gclSetKernelArgAPPLE(k, 2, sizeof(heightA), &heightA, &kargs);
  err |= gclSetKernelArgAPPLE(k, 3, sizeof(widthB), &widthB, &kargs);
  err |= gclSetKernelArgAPPLE(k, 4, sizeof(heightB), &heightB, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 5, A, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 6, B, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 7, C, &kargs);
  gcl_log_cl_fatal(err, "setting argument for matrix_calc failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing matrix_calc failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

// Initialization functions
static void initBlocks(void) {
  const char* build_opts = "";
  static dispatch_once_t once;
  dispatch_once(&once,
    ^{ int err = gclBuildProgramBinaryAPPLE("mykernel.cl", "", &bmap, build_opts);
       if (!err) {
          assert(bmap.map[0].block_ptr == matrix_calc_kernel && "mismatch block");
          bmap.map[0].kernel = clCreateKernel(bmap.program, "matrix_calc", &err);
       }
     });
}

__attribute__((constructor))
static void RegisterMap(void) {
  gclRegisterBlockKernelMap(&bmap);
  bmap.map[0].block_ptr = matrix_calc_kernel;
}
