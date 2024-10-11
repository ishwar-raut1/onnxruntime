
#pragma once
#include <d3d12.h>
#include <DirectML.h>
#include "core/framework/execution_provider.h"
#include "core/framework/allocator.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/compute_capability.h"
namespace NvDml {
std::unique_ptr<onnxruntime::IExecutionProvider> CreateExecutionProvider(
    IDMLDevice* dmlDevice,
    ID3D12CommandQueue* execution_context);

class NvDmlExecutionProvider : public onnxruntime::IExecutionProvider {
 public:
  virtual ~NvDmlExecutionProvider() {}
  NvDmlExecutionProvider() = delete;

  explicit NvDmlExecutionProvider(
      IDMLDevice* dmlDevice,
      ID3D12CommandQueue* executionContext);

  std::unique_ptr<onnxruntime::IDataTransfer> GetDataTransfer() const final override {
    return nullptr;
  }

  std::shared_ptr<onnxruntime::KernelRegistry> GetKernelRegistry() const final override {
    return std::make_shared<onnxruntime::KernelRegistry>();
  }

  std::vector<std::unique_ptr<onnxruntime::ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph,
                const onnxruntime::IExecutionProvider::IKernelLookup& kernel_lookup) const final override;

  onnxruntime::common::Status OnSessionInitializationEnd() override {
    return Status::OK();
  }

  onnxruntime::Status Sync() const final override {
    return Status::OK();
  }

  Status OnRunStart(const onnxruntime::RunOptions& /*run_options*/) final {
    return Status::OK();
  }

  Status OnRunEnd(bool /*sync_stream*/, const onnxruntime::RunOptions& /*run_options*/) final {
    return Status::OK();
  }

  virtual std::vector<onnxruntime::AllocatorPtr> CreatePreferredAllocators() override {
    return std::vector<onnxruntime::AllocatorPtr>();
  }
};

}  // namespace NvDml
