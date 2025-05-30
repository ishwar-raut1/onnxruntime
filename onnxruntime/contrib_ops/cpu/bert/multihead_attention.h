// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
class MultiHeadAttention final : public OpKernel, public AttentionCPUBase {
 public:
  MultiHeadAttention(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;

 protected:
  int num_heads_;  // number of attention heads
  float mask_filter_value_;
  bool is_unidirectional_;
  bool disable_flash_;
  bool disable_decoder_attention_;
  int l2_cache_size_;
};

}  // namespace contrib
}  // namespace onnxruntime
