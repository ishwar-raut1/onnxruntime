// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cpu/llm/rotary_embedding_helper.h"
#include "core/providers/cuda/llm/rotary_embedding.h"
#include "core/providers/cuda/llm/rotary_embedding_impl.h"

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;
using namespace onnxruntime::rotary_embedding_helper;

namespace onnxruntime {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                        \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                        \
      RotaryEmbedding,                                                  \
      kOnnxDomain,                                                      \
      23,                                                               \
      T,                                                                \
      kCudaExecutionProvider,                                           \
      (*KernelDefBuilder::Create())                                     \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())        \
          .TypeConstraint("M", DataTypeImpl::GetTensorType<int64_t>()), \
      RotaryEmbedding<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

template <typename T>
RotaryEmbedding<T>::RotaryEmbedding(const OpKernelInfo& info) : CudaKernel(info) {
  rotary_embedding_dim = static_cast<int>(info.GetAttrOrDefault<int64_t>("rotary_embedding_dim", 0));
  num_heads = static_cast<int>(info.GetAttrOrDefault<int64_t>("num_heads", 0));
  interleaved = (info.GetAttrOrDefault<int64_t>("interleaved", 0) == 1);
}

template <typename T>
Status RotaryEmbedding<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* cos_cache = context->Input<Tensor>(1);
  const Tensor* sin_cache = context->Input<Tensor>(2);
  const Tensor* position_ids = context->Input<Tensor>(3);  // Optional, can be nullptr

  RotaryParameters parameters = {};
  ORT_RETURN_IF_ERROR(rotary_embedding_helper::CheckInputs<Tensor>(input,
                                                                   position_ids,
                                                                   cos_cache,
                                                                   sin_cache,
                                                                   num_heads,
                                                                   rotary_embedding_dim,
                                                                   &parameters));

  Tensor* output = context->Output(0, input->Shape());

  // Launch rotary embedding kernel
  typedef typename ToCudaType<T>::MappedType CudaT;
  auto& device_prop = GetDeviceProp();

  // Handle optional position_ids - pass nullptr if position_ids is null
  const int64_t* position_ids_data = (position_ids != nullptr) ? position_ids->Data<int64_t>() : nullptr;

  return LaunchRotaryEmbeddingKernel<CudaT>(
      Stream(context),
      reinterpret_cast<CudaT*>(output->template MutableData<T>()),
      reinterpret_cast<const CudaT*>(input->template Data<T>()),
      position_ids_data,
      reinterpret_cast<const CudaT*>(cos_cache->template Data<T>()),
      reinterpret_cast<const CudaT*>(sin_cache->template Data<T>()),
      parameters.batch_size,
      parameters.sequence_length,
      parameters.num_heads,
      parameters.head_size,
      parameters.rotary_embedding_dim,
      parameters.max_sequence_length,
      parameters.position_ids_format,
      interleaved,
      device_prop.maxThreadsPerBlock,
      parameters.transposed);
}

}  // namespace cuda
}  // namespace onnxruntime
