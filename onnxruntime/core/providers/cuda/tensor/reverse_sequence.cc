// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "reverse_sequence.h"
#include "reverse_sequence_impl.h"

#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/common.h"
#include "core/framework/utils.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    ReverseSequence,
    kOnnxDomain,
    10,
    kCudaExecutionProvider,
    // No string type implmeneted in cuda
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    ReverseSequenceOp);

template <typename T>
static void ReverseSequenceImpl(
    const Tensor& X,
    const Tensor& sequence_lengths,
    Tensor& Y,
    const int64_t batch_size,
    const int64_t max_seq_len,
    const int64_t element_size,
    bool time_major,
    const fast_divmod* fdm_grouped_strides) {
  const T* x_data = X.Data<T>();
  const int64_t* seq_len_data = sequence_lengths.Data<int64_t>();
  T* y_data = Y.MutableData<T>();

  ReverseSequenceCudaImpl(
      reinterpret_cast<const typename ToCudaType<T>::MappedType *>(x_data),
      seq_len_data,
      reinterpret_cast<typename ToCudaType<T>::MappedType *>(y_data),
      gsl::narrow<int>(batch_size), gsl::narrow<int>(max_seq_len), gsl::narrow<int>(element_size),
      time_major, fdm_grouped_strides);
}

Status ReverseSequenceOp::ComputeInternal(OpKernelContext* context) const {
  const auto& X = *context->Input<Tensor>(0);
  const auto data_type = X.DataType();
  const auto& dims = X.Shape();

  const auto batch_size = time_major_ ? dims[1] : dims[0];
  const auto max_seq_len = time_major_ ? dims[0] : dims[1];
  const auto element_size = dims.SizeFromDimension(2);

  const auto& seq_lengths = *context->Input<Tensor>(1);
  const auto& seq_len_shape = seq_lengths.Shape();

  if (seq_len_shape.NumDimensions() != 1 || seq_len_shape[0] != batch_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "sequence_lens shape must be {batch_size}. Got:",
                           seq_len_shape, ". batch_size=", batch_size);
  }

  // elements per threads
  int ept = ReverseSequenceElementsPerThread();
  int64_t element_group_size = (element_size + ept - 1) / ept;
  std::vector<int64_t> grouped_strides = {element_group_size, element_group_size, 1};
  grouped_strides[0] *= (time_major_) ? batch_size : max_seq_len;
  CudaAsyncBuffer<fast_divmod> fdm_grouped_strides(this, 3);
  ORT_ENFORCE(CalculateFdmStrides(fdm_grouped_strides.CpuSpan(), grouped_strides));
  ORT_RETURN_IF_ERROR(fdm_grouped_strides.CopyToGpu());

  auto& Y = *context->Output(0, dims);

  #define CheckAndCallTypedImpl(T)                              \
    if (data_type == DataTypeImpl::GetType<T>()) {              \
      ReverseSequenceImpl<T>(                                   \
          X, seq_lengths, Y,                                    \
          batch_size, max_seq_len, element_size,                \
          time_major_, fdm_grouped_strides.GpuPtr());           \
      return Status::OK();                                      \
    }

  CheckAndCallTypedImpl(float)
  CheckAndCallTypedImpl(MLFloat16)
  CheckAndCallTypedImpl(int32_t)
  CheckAndCallTypedImpl(uint32_t)
  CheckAndCallTypedImpl(int16_t)
  CheckAndCallTypedImpl(uint16_t)
  CheckAndCallTypedImpl(int8_t)
  CheckAndCallTypedImpl(uint8_t)
  CheckAndCallTypedImpl(double)
  CheckAndCallTypedImpl(bool)
  CheckAndCallTypedImpl(int64_t)
  CheckAndCallTypedImpl(uint64_t)

  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, 
      "Type for ", data_type, " is not supported yet in ReverseSequence.");

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
