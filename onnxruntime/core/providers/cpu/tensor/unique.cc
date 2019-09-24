// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// there's no way to use a raw pointer as the copy destination with std::copy_n
// (which gsl::copy uses with span::data() which returns a raw pointer) with the 14.11 toolset
// without generating a 4996 warning. going through an iterator is way too much overhead so turn off the warning.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
#include "gsl/gsl_algorithm"
#include "gsl/span"
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include "core/providers/cpu/tensor/unique.h"

#include "core/framework/utils.h"
#include "core/providers/common.h"

#include <map>

namespace onnxruntime {

/*
ONNX_OPERATOR_SET_SCHEMA(
    Unique,
    11,
    OpSchema()
        .SetDoc(Unique_ver11_doc)
        .Attr(
            "sorted",
            "(Optional) Whether to sort the unique elements in ascending order before returning as output. "
            "Must be one of 0, or 1 (default).",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .Attr(
            "axis",
            "(Optional) The dimension to apply unique. If not specified, the unique elements of the "
            "flattened input are returned. Negative value means counting dimensions "
            "from the back. Accepted range is [-r, r-1] where r = rank(input).",
            AttributeProto::INT,
            OPTIONAL)
        .Input(0, "X", "A N-D input tensor that is to be processed.", "T")
        .Output(
            0,
            "Y",
            "A tensor of the same type as 'X' "
            "containing all the unique values or subtensors sliced along a provided 'axis' in 'X', either sorted "
            "or maintained in the same order they occur in input 'X'",
            "T")
        .Output(
            1,
            "indices",
            "A 1-D INT64 tensor "
            "containing indices of 'Y' elements' first occurance in 'X'. "
            "When 'axis' is provided, it contains indices to subtensors in input 'X' on the 'axis'. "
            "When 'axis' is not provided, it contains indices to values in the flattened input tensor. ",
            "tensor(int64)",
            OpSchema::Optional)
        .Output(
            2,
            "inverse_indices",
            "A 1-D INT64 tensor "
            "containing, for elements of 'X', its corresponding indices in 'Y'. "
            "When 'axis' is provided, it contains indices to subtensors in output 'Y' on the 'axis'. "
            "When 'axis' is not provided, it contains indices to values in output 'Y'. ",
            "tensor(int64)",
            OpSchema::Optional)
        .Output(
            3,
            "counts",
            "A 1-D INT64 tensor containing "
            "the count of each element "
            "of 'Y' in input 'X'",
            "tensor(int64)",
            OpSchema::Optional)
        .TypeConstraint(
            "T",
            OpSchema::all_tensor_types(),
            "Input can be of any tensor type.")    
*/
ONNX_CPU_OPERATOR_KERNEL(
    Unique,
    11,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    Unique);

Status Unique::Compute(OpKernelContext* context) const {
  const Tensor& input = *context->Input<Tensor>(0);

  Status status;
  auto data_type = input.DataType();

  DispatchOnTensorTypeWithReturn(data_type, status, ComputeImpl, *context);

  return status;
}

// class to represent a subtensor along a given axis for a single entry on that axis
template <typename T>
class Subtensor {
 public:
  // Create Subtensor for entry 'idx' on axis 'axis'
  // n_axis is the number of entries for 'axis' is the original data.
  //   e.g. if original shape was [4, 2] and axis is 1, n_axis == 2.
  // subtensor_shape is the shape for the subtensor. the dimension value for the 'axis' dimension in it will be 1.
  Subtensor(const gsl::span<const T>& data, const TensorShape& subtensor_shape,
            int64_t axis, int64_t n_axis, int64_t idx) {
    // rows and columns for the slice along axis, flattened to 2D by merging the dimensions before and after the axis
    int64_t columns = subtensor_shape.SizeFromDimension(axis);
    int64_t rows = subtensor_shape.SizeToDimension(axis);
    items_.reserve(rows * columns);
    size_t cur_data = idx * columns;  // offset into data for first row of slice

    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < columns; ++c) {
        // TODO: could copy blocks instead of individual items for non std::string types
        items_.push_back(data[cur_data + c]);
      }

      cur_data += columns * n_axis;
    }
  }

  bool operator<(const Subtensor& rhs) const {
    // we only expect to be comparing entries with the same shape
    assert(items_.size() == rhs.items_.size());

    return items_ < rhs.items_;
    //bool less_than = false;
    //for (size_t i = 0, end = items_.size(); i < end; ++i) {
    //  if (items_[i] == rhs.items_[i])
    //    continue;
    //  else {
    //    // return items_[i] < rhs.items_[i];
    //    less_than = items_[i] < rhs.items_[i];
    //    break;
    //  }
    //}

    //ORT_ENFORCE(less_than == c, "temporary test");

    //return less_than;  // equal if we get here
  }

  const std::vector<T>& GetItems() const { return items_; }

 private:
  // TODO: Copy for now. std::string would be better as std::reference_wrapper<std::string>
  std::vector<T> items_;
};

template <typename T>
static void CreateFlattenedOutput(OpKernelContext& context,
                                  const std::map<const T, int64_t>& offsets,         // sorted:unsorted idx
                                  const std::vector<std::vector<int64_t>>& indices,  // unsorted
                                  const std::vector<int64_t>& inverse_index,         // unsorted
                                  bool sorted) {
  int64_t num_unique = static_cast<int64_t>(indices.size());
  Tensor& Y = *context.Output(0, TensorShape({num_unique /*, <Subtensor shape> if not flattened */}));
  Tensor* indices_out = context.Output(1, TensorShape({num_unique}));
  Tensor* inverse_indices = context.Output(2, TensorShape({static_cast<int64_t>(inverse_index.size())}));
  Tensor* counts = context.Output(3, TensorShape({num_unique}));

  auto Y_data = Y.MutableDataAsSpan<T>();
  gsl::span<int64_t> indices_data = indices_out != nullptr ? indices_out->MutableDataAsSpan<int64_t>()
                                                           : gsl::span<int64_t>();
  gsl::span<int64_t> inverse_indices_data = inverse_indices != nullptr ? inverse_indices->MutableDataAsSpan<int64_t>()
                                                                       : gsl::span<int64_t>();
  gsl::span<int64_t> counts_data = counts != nullptr ? counts->MutableDataAsSpan<int64_t>()
                                                     : gsl::span<int64_t>();

  // iterate using 'offsets' which is sorted, but contains the offset of the unsorted entry
  auto offsets_iter = offsets.begin();
  for (int64_t i = 0, end = num_unique; i < end; ++i, ++offsets_iter) {
    auto unsorted_idx = offsets_iter->second;
    // write sequentially if we want sorted output
    auto output_idx = sorted ? i : unsorted_idx;

    Y_data[output_idx] = offsets_iter->first;

    if (indices_out) {
      indices_data[output_idx] = indices[unsorted_idx].front();
    }

    if (counts) {
      counts_data[output_idx] = indices[unsorted_idx].size();
    }
  }

  if (inverse_indices) {
    if (sorted) {
      // need to convert unsorted entries in the inverse index to their sorted values
      std::vector<int64_t> unsorted_to_sorted;
      unsorted_to_sorted.resize(num_unique);
      int64_t sorted_idx = 0;
      for (const auto& offset : offsets) {
        unsorted_to_sorted[offset.second] = sorted_idx++;
      }

      for (size_t i = 0, end = inverse_index.size(); i < end; ++i) {
        inverse_indices_data[i] = unsorted_to_sorted[inverse_index[i]];
      }
    } else {
      // memcpy or gsl::copy
      for (size_t i = 0, end = inverse_index.size(); i < end; ++i) {
        inverse_indices_data[i] = inverse_index[i];
      }
    }
  }
}

template <typename T>
static void CreateOutput(OpKernelContext& context,
                         const TensorShape& subtensor_shape,
                         int64_t axis,
                         const std::map<const Subtensor<T>, int64_t>& offsets,  // sorted:unsorted idx
                         const std::vector<std::vector<int64_t>>& indices,      // unsorted
                         const std::vector<int64_t>& inverse_index,             // unsorted
                         bool sorted) {
  int64_t num_unique = static_cast<int64_t>(indices.size());
  int64_t num_cols = subtensor_shape.SizeFromDimension(axis);
  int64_t num_rows = subtensor_shape.SizeToDimension(axis);

  const std::vector<int64_t> subtensor_dims = subtensor_shape.GetDims();
  std::vector<int64_t> Y_dims;
  Y_dims.reserve(subtensor_dims.size());
  for (int64_t i = 0, end = subtensor_dims.size(); i < end; ++i) {
    if (i == axis)
      Y_dims.push_back(num_unique);
    else
      Y_dims.push_back(subtensor_dims[i]);
  }

  Tensor& Y = *context.Output(0, TensorShape(std::move(Y_dims)));
  Tensor* indices_out = context.Output(1, TensorShape({num_unique}));
  Tensor* inverse_indices = context.Output(2, TensorShape({static_cast<int64_t>(inverse_index.size())}));
  Tensor* counts = context.Output(3, TensorShape({num_unique}));

  auto Y_data = Y.MutableDataAsSpan<T>();
  gsl::span<int64_t> indices_data = indices_out != nullptr ? indices_out->MutableDataAsSpan<int64_t>()
                                                           : gsl::span<int64_t>();
  gsl::span<int64_t> inverse_indices_data = inverse_indices != nullptr ? inverse_indices->MutableDataAsSpan<int64_t>()
                                                                       : gsl::span<int64_t>();
  gsl::span<int64_t> counts_data = counts != nullptr ? counts->MutableDataAsSpan<int64_t>()
                                                     : gsl::span<int64_t>();

  // iterate using 'offsets' which is sorted, but contains the offset of the unsorted entry
  auto offsets_iter = offsets.begin();
  //size_t items_per_entry = subtensor_shape.Size();

  for (int64_t i = 0, end = num_unique; i < end; ++i, ++offsets_iter) {
    auto unsorted_idx = offsets_iter->second;
    // write sequentially if we want sorted output
    auto output_idx = (sorted ? i : unsorted_idx);

    const auto& items = offsets_iter->first.GetItems();
    auto item = items.cbegin();
    assert(static_cast<int64_t>(items.size()) == num_rows * num_cols);

    int64_t out_offset = output_idx * num_cols;

    for (int64_t row = 0; row < num_rows; ++row) {
      // copy num_cols items from entries to output
      if (std::is_same<T, std::string>::value) {
        std::copy(item, item + num_cols, &Y_data[out_offset]);
      } else {
        std::copy_n(item, num_cols, &Y_data[out_offset]);
      }

      item += num_cols;
      out_offset += num_unique * num_cols;
    }

    assert(item == items.cend());

    if (indices_out) {
      indices_data[output_idx] = indices[unsorted_idx].front();
    }

    if (counts) {
      counts_data[output_idx] = indices[unsorted_idx].size();
    }
  }

  if (inverse_indices) {
    if (sorted) {
      // need to convert unsorted entries in the inverse index to their sorted values
      std::vector<int64_t> unsorted_to_sorted;
      unsorted_to_sorted.resize(num_unique);
      int64_t sorted_idx = 0;
      for (const auto& offset : offsets) {
        unsorted_to_sorted[offset.second] = sorted_idx++;
      }

      for (size_t i = 0, end = inverse_index.size(); i < end; ++i) {
        inverse_indices_data[i] = unsorted_to_sorted[inverse_index[i]];
      }
    } else {
      // memcpy or gsl::copy
      for (size_t i = 0, end = inverse_index.size(); i < end; ++i) {
        inverse_indices_data[i] = inverse_index[i];
      }
    }
  }
}

//// struct to allow us to use T as a key in a map without copying
//template <typename T>
//struct TRef {
//  TRef(const T& value) : ref{value} {}
//  operator<(const TRef& rhs) {
//    return ref.get() < rhs.ref.get();
//  }
//
//  std::reference_wrapper<T> ref;
//};

template <typename T>
Status Unique::ComputeImpl(OpKernelContext& context) const {
  const Tensor& input = *context.Input<Tensor>(0);
  auto data = input.DataAsSpan<T>();

  if (flatten_) {
    // offset of entry in indices
    // TODO: Could handle T=std::string better and avoid copying to use in the key of offsets but that req
    std::map<const T, int64_t> offsets;
    std::vector<std::vector<int64_t>> indices;
    std::vector<int64_t> inverse_index;

    indices.reserve(data.size() / 2);  // arbitrary value. at worst 1 realloc but could be too large
    inverse_index.reserve(data.size());

    int64_t num_unique = 0;

    for (int64_t i = 0, end = input.Shape().Size(); i < end; ++i) {
      auto entry = offsets.find(data[i]);

      if (entry == offsets.end()) {
        // new value
        offsets[data[i]] = num_unique;
        inverse_index.push_back({num_unique});
        indices.push_back({i});
        ++num_unique;
      } else {
        size_t indices_idx = entry->second;
        indices[indices_idx].push_back(i);
        inverse_index.push_back(indices_idx);
      }
    }

    CreateFlattenedOutput(context, offsets, indices, inverse_index, sort_);
  } else {
    const auto& input_shape = input.Shape();
    const int64_t input_dims = static_cast<int64_t>(input_shape.NumDimensions());

    int64_t axis = HandleNegativeAxis(axis_, input_dims);

    std::vector<int64_t> subtensor_dims;
    subtensor_dims.reserve(input_dims);
    for (int64_t i = 0; i < input_dims; ++i) {
      if (i == axis)
        subtensor_dims.push_back(1);
      else
        subtensor_dims.push_back(input_shape[i]);
    }

    TensorShape subtensor_shape(std::move(subtensor_dims));

    std::map<const Subtensor<T>, int64_t> offsets;
    std::vector<std::vector<int64_t>> indices;
    std::vector<int64_t> inverse_index;

    indices.reserve(data.size() / 2);  // arbitrary value. at worst 1 realloc but could be too large
    inverse_index.reserve(data.size());

    int64_t num_unique = 0;
    int64_t n_axis = input_shape[axis];

    for (int64_t i = 0; i < n_axis; ++i) {
      Subtensor<T> s(data, subtensor_shape, axis, n_axis, i);

      auto entry = offsets.find(s);
      if (entry == offsets.end()) {
        // new value
        offsets[std::move(s)] = num_unique;
        inverse_index.push_back({num_unique});
        indices.push_back({i});
        ++num_unique;
      } else {
        size_t indices_idx = entry->second;
        indices[indices_idx].push_back(i);
        inverse_index.push_back(indices_idx);
      }
    }

    CreateOutput(context, subtensor_shape, axis, offsets, indices, inverse_index, sort_);
  }

  return Status::OK();
}

}  // namespace onnxruntime
