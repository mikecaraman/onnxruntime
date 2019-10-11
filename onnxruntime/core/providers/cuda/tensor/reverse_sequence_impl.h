// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {

int ReverseSequenceElementsPerThread(void);

template <typename T>
void ReverseSequenceCudaImpl(
    const T* x_data,
    const int64_t* seq_len_data,
    T* y_data, 
    int batch_size, 
    int max_seq_len,
    int element_size,
    bool time_major,
    const fast_divmod* fdm_grouped_strides);

}  // namespace cuda
}  // namespace onnxruntime

