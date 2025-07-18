// Copyright (c) Microsoft Corporation. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "nv_includes.h"
#include "core/framework/data_transfer.h"
#include "core/framework/ort_value.h"
#include "core/session/abi_devices.h"

namespace onnxruntime {

class GPUDataTransfer : public IDataTransfer {
 public:
  GPUDataTransfer() = default;
  ~GPUDataTransfer() = default;

  bool CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const override;

  // Dumpen MSVC warning about not fully overriding
  using IDataTransfer::CopyTensor;
  common::Status CopyTensor(const Tensor& src, Tensor& dst) const override;
  common::Status CopyTensorAsync(const Tensor& src, Tensor& dst, Stream& stream) const override;
};

struct NvTensorRtRtxDataTransfer : OrtDataTransferImpl, OrtApi {
  NvTensorRtRtxDataTransfer(OrtApi ort_api,
                            const OrtMemoryDevice* device_mem_info_,
                            const OrtMemoryDevice* shared_mem_info_ )
      : OrtApi(ort_api), device_mem_info{device_mem_info_}, shared_mem_info{shared_mem_info_} {
    CanCopy = CanCopyImpl;
    CopyTensors = CopyTensorsImpl;
    Release = ReleaseImpl;

  }
  static bool ORT_API_CALL CanCopyImpl(void* this_ptr,
                                       const OrtMemoryDevice* src_memory_device,
                                       const OrtMemoryDevice* dst_memory_device) noexcept;

  // function to copy one or more tensors.
  // implementation can optionally use async copy if a stream is available for the input.
  static OrtStatus* ORT_API_CALL CopyTensorsImpl(void* this_ptr,
                                                 const OrtValue** src_tensors_ptr,
                                                 OrtValue** dst_tensors_ptr,
                                                 OrtSyncStream** streams_ptr,
                                                 size_t num_tensors) noexcept;
  static void ORT_API_CALL ReleaseImpl(void* this_ptr) noexcept;

 private:
  const OrtMemoryDevice* device_mem_info;
  const OrtMemoryDevice* shared_mem_info;
  GPUDataTransfer gpu_data_transfer;
};



}  // namespace onnxruntime
