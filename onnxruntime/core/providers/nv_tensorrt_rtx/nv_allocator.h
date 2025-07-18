// Copyright (c) Microsoft Corporation. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/inlined_containers.h"
#include "core/framework/allocator.h"
#include <mutex>

namespace onnxruntime {

class CUDAAllocator : public IAllocator {
 public:
  CUDAAllocator(OrtDevice::DeviceId device_id, const char* name)
      : IAllocator(OrtMemoryInfo(name, OrtAllocatorType::OrtDeviceAllocator,
                                 OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, OrtDevice::VendorIds::NVIDIA,
                                           device_id),
                                 OrtMemTypeDefault)) {}
  void* Alloc(size_t size) override;
  void Free(void* p) override;

 private:
  void CheckDevice(bool throw_when_fail) const;
  void SetDevice(bool throw_when_fail) const;
};

class CUDAExternalAllocator : public CUDAAllocator {
  typedef void* (*ExternalAlloc)(size_t size);
  typedef void (*ExternalFree)(void* p);
  typedef void (*ExternalEmptyCache)();

 public:
  CUDAExternalAllocator(OrtDevice::DeviceId device_id, const char* name, void* alloc, void* free, void* empty_cache)
      : CUDAAllocator(device_id, name) {
    alloc_ = reinterpret_cast<ExternalAlloc>(alloc);
    free_ = reinterpret_cast<ExternalFree>(free);
    empty_cache_ = reinterpret_cast<ExternalEmptyCache>(empty_cache);
  }

  void* Alloc(size_t size) override;
  void Free(void* p) override;
  void* Reserve(size_t size) override;

 private:
  mutable std::mutex lock_;
  ExternalAlloc alloc_;
  ExternalFree free_;
  ExternalEmptyCache empty_cache_;
  InlinedHashSet<void*> reserved_;
};

// TODO: add a default constructor
class CUDAPinnedAllocator : public IAllocator {
 public:
  CUDAPinnedAllocator(OrtDevice::DeviceId device_id, const char* name)
      : IAllocator(
            OrtMemoryInfo(name, OrtAllocatorType::OrtDeviceAllocator,
                          OrtDevice(OrtDevice::GPU, OrtDevice::MemType::HOST_ACCESSIBLE, OrtDevice::VendorIds::NVIDIA,
                                    device_id),
                          OrtMemTypeCPUOutput)) {}

  void* Alloc(size_t size) override;
  void Free(void* p) override;
};

struct CudaOrtAllocator : public OrtAllocator {
  CudaOrtAllocator(const OrtMemoryInfo* memory_info, OrtDevice::DeviceId device_id, const char* name)
      : memory_info(memory_info) {
    cuda_allocator = new CUDAAllocator(device_id, name);
    Alloc = AllocImpl;
    Free = FreeImpl;
    Info = InfoImpl;
    Reserve = ReserveImpl;
  }
  ~CudaOrtAllocator() {
    delete cuda_allocator;
  }
  static void* ORT_API_CALL AllocImpl(struct OrtAllocator* this_, size_t size) {
    const CudaOrtAllocator* impl = static_cast<const CudaOrtAllocator*>(this_);
    return impl->cuda_allocator->Alloc(size);
  }
  static void ORT_API_CALL FreeImpl(struct OrtAllocator* this_, void* p) {
    const CudaOrtAllocator* impl = static_cast<const CudaOrtAllocator*>(this_);
    return impl->cuda_allocator->Free(p);
  }
  static const OrtMemoryInfo* ORT_API_CALL InfoImpl(const struct OrtAllocator* this_) {
    const CudaOrtAllocator* impl = static_cast<const CudaOrtAllocator*>(this_);
    return impl->memory_info;
  }
  static void* ORT_API_CALL ReserveImpl(struct OrtAllocator* this_, size_t size) {
    const CudaOrtAllocator* impl = static_cast<const CudaOrtAllocator*>(this_);
    return impl->cuda_allocator->Reserve(size);
  }
  const OrtMemoryInfo* memory_info;
  CUDAAllocator* cuda_allocator;
};

struct CudaOrtPinnedAllocator : public OrtAllocator {
  CudaOrtPinnedAllocator(const OrtMemoryInfo* memory_info, OrtDevice::DeviceId device_id, const char* name)
      : memory_info(memory_info) {
    cuda_allocator = new CUDAPinnedAllocator(device_id, name);
    Alloc = AllocImpl;
    Free = FreeImpl;
    Info = InfoImpl;
    Reserve = ReserveImpl;
  }
  ~CudaOrtPinnedAllocator() {
    delete cuda_allocator;
  }
  static void* ORT_API_CALL AllocImpl(struct OrtAllocator* this_, size_t size) {
    const CudaOrtPinnedAllocator* impl = static_cast<const CudaOrtPinnedAllocator*>(this_);
    return impl->cuda_allocator->Alloc(size);
  }
  static void ORT_API_CALL FreeImpl(struct OrtAllocator* this_, void* p) {
    const CudaOrtPinnedAllocator* impl = static_cast<const CudaOrtPinnedAllocator*>(this_);
    return impl->cuda_allocator->Free(p);
  }
  static const OrtMemoryInfo* ORT_API_CALL InfoImpl(const struct OrtAllocator* this_) {
    const CudaOrtPinnedAllocator* impl = static_cast<const CudaOrtPinnedAllocator*>(this_);
    return impl->memory_info;
  }
  static void* ORT_API_CALL ReserveImpl(struct OrtAllocator* this_, size_t size) {
    const CudaOrtPinnedAllocator* impl = static_cast<const CudaOrtPinnedAllocator*>(this_);
    return impl->cuda_allocator->Reserve(size);
  }
  const OrtMemoryInfo* memory_info;
  CUDAPinnedAllocator* cuda_allocator;
};
}  // namespace onnxruntime
