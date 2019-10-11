// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "reverse_sequence_impl.h"

namespace onnxruntime {
namespace cuda {

static const int kReverseSequenceElementsPerThread = 4;

int ReverseSequenceElementsPerThread(void) 
{
    return kReverseSequenceElementsPerThread;
}

template <typename T, bool time_major>
__global__ void ReverseSequenceImplKernel(
    const T* x_data,
    const int64_t* seq_len_data,
    T* y_data, 
    int batch_size, 
    int max_seq_len,
    int element_size,
    int group_count,
    const fast_divmod* fdm_grouped_strides)
{
    CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(grouped_index, group_count);

    int batch_id, seq_id, gid, remain = grouped_index;
    if (time_major) {
        fdm_grouped_strides[0].divmod(remain, seq_id, remain);
        fdm_grouped_strides[1].divmod(remain, batch_id, gid);
    }
    else {
        fdm_grouped_strides[0].divmod(remain, batch_id, remain);
        fdm_grouped_strides[1].divmod(remain, seq_id, gid);
    }
    int eid = gid * kReverseSequenceElementsPerThread;
    int target_seq_id = (seq_id < (int)seq_len_data[batch_id]) ? (max_seq_len - seq_id) : seq_id;
    int flat_src_idx, flat_target_idx;
    if (time_major) {
        flat_src_idx = seq_id * (batch_size * element_size) + batch_size * element_size + eid;
        flat_target_idx = target_seq_id * (batch_size * element_size) + batch_size * element_size + eid;
    }
    else {
        flat_src_idx = batch_size * (max_seq_len * element_size) + seq_id * element_size + eid;
        flat_target_idx = batch_size * (max_seq_len * element_size) + target_seq_id * element_size + eid;
    }

    #pragma unroll
    for (; eid < element_size; ++eid) {
        y_data[flat_target_idx++] = x_data[flat_src_idx++];
    }
}

template <typename T>
void ReverseSequenceCudaImpl(
    const T* x_data,
    const int64_t* seq_len_data,
    T* y_data,
    int batch_size,
    int max_seq_len,
    int element_size,
    bool time_major,
    const fast_divmod* fdm_grouped_strides)
{
  int group_count = batch_size * max_seq_len * ((element_size + kReverseSequenceElementsPerThread - 1) / kReverseSequenceElementsPerThread);    
  int blocksPerGrid = (int)((group_count + GridDim::maxThreadsPerBlock - 1) / GridDim::maxThreadsPerBlock);
  if (time_major) {
    ReverseSequenceImplKernel<T, true><<<blocksPerGrid, GridDim::maxThreadsPerBlock>>>(
        x_data, seq_len_data, y_data, batch_size, max_seq_len, element_size, 
        group_count, fdm_grouped_strides);
  }
  else {
    ReverseSequenceImplKernel<T, false><<<blocksPerGrid, GridDim::maxThreadsPerBlock>>>(
        x_data, seq_len_data, y_data, batch_size, max_seq_len, element_size, 
        group_count, fdm_grouped_strides);
  }
}

#define InstantiateReverseSequenceImpl(T) template void ReverseSequenceCudaImpl(  \
    const T* x_data,                                                              \
    const int64_t* seq_len_data,                                                  \
    T* y_data,                                                                    \
    int batch_size,                                                               \
    int max_seq_len,                                                              \
    int element_size,                                                             \
    bool time_major,                                                              \
    const fast_divmod* fdm_grouped_strides)


InstantiateReverseSequenceImpl(float);
InstantiateReverseSequenceImpl(double);
InstantiateReverseSequenceImpl(int64_t);
InstantiateReverseSequenceImpl(uint64_t);
InstantiateReverseSequenceImpl(int32_t);
InstantiateReverseSequenceImpl(uint32_t);
InstantiateReverseSequenceImpl(int16_t);
InstantiateReverseSequenceImpl(uint16_t);
InstantiateReverseSequenceImpl(int8_t);
InstantiateReverseSequenceImpl(uint8_t);
InstantiateReverseSequenceImpl(bool);
InstantiateReverseSequenceImpl(half);

}
}