
// Copyright (c) 2024 NVIDIA Corporation.
// Licensed under the MIT License.

#pragma once
#include <d3d12.h>
#include <DirectML.h>
#include "core/framework/execution_provider.h"
#include "core/framework/allocator.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/compute_capability.h"


#include <wrl/client.h>
#include <wrl/implements.h>

#include <wil/wrl.h>
using Microsoft::WRL::ComPtr;

#include "core/providers/dml/DmlExecutionProvider/src/External/D3DX12/d3dx12.h"
#include "core/providers/dml/DmlExecutionProvider/src/ErrorHandling.h"
#include "core/providers/dml/DmlExecutionProvider/src/DescriptorPool.h"
#include "core/providers/dml/DmlExecutionProvider/src/DmlCommittedResourceAllocator.h"
#include "core/providers/dml/DmlExecutionProvider/inc/DmlExecutionProvider.h"
#include "core/providers/dml/DmlExecutionProvider/src/AllocationInfo.h"


namespace NvDml {
std::unique_ptr<onnxruntime::IExecutionProvider> CreateExecutionProvider(
    IDMLDevice* dmlDevice,
    ID3D12CommandQueue* execution_context);

class NvDmlExecutionProvider : public onnxruntime::IExecutionProvider {
 public:
  virtual ~NvDmlExecutionProvider();
  NvDmlExecutionProvider() = delete;

  explicit NvDmlExecutionProvider(
      ID3D12Device* dmlDevice,
      ID3D12CommandQueue* executionContext,
      Dml::ExecutionContext* context);

  std::unique_ptr<onnxruntime::IDataTransfer> GetDataTransfer() const final override {
    // returning nullptr will use the DataTrasfer implemented in the DML EP
    return nullptr;
  }


  std::vector<std::unique_ptr<onnxruntime::ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph,
                const onnxruntime::IExecutionProvider::IKernelLookup& kernel_lookup) const final override;

  onnxruntime::common::Status OnSessionInitializationEnd() override {
    return Status::OK();
  }

  onnxruntime::Status Sync() const final override {
    m_executionContext->Flush();
    m_executionContext->GetCurrentCompletionEvent().WaitForSignal(false);
    return Status::OK();
  }

  Status OnRunStart(const onnxruntime::RunOptions& /*run_options*/) final {
    return Status::OK();
  }

  Status OnRunEnd(bool /*sync_stream*/, const onnxruntime::RunOptions& /*run_options*/) final {
    m_executionContext->Flush();
    return Status::OK();
  }

  virtual std::vector<onnxruntime::AllocatorPtr> CreatePreferredAllocators() override {
    // returning empty vector will use the preffered allocators implemented by DML EP.
    return std::vector<onnxruntime::AllocatorPtr>();
  }

  Dml::ExecutionContext* GetExecutionContext() const {
    return m_executionContext.Get();
  }

  ID3D12Device* GetD3D12Device() const {
    return m_d3d12Device.Get();
  }

  std::shared_ptr<onnxruntime::KernelRegistry> GetKernelRegistry() const final override {
    return m_kernelRegistry;
  }

  private:
  ComPtr<ID3D12Device> m_d3d12Device;
  ComPtr<ID3D12CommandQueue> m_commandQueue;
  ComPtr<Dml::ExecutionContext> m_executionContext;
  std::shared_ptr<onnxruntime::KernelRegistry> m_kernelRegistry;

};

}  // namespace NvDml
