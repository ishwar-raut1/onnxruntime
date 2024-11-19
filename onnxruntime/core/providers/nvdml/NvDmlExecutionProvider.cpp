// Copyright (c) 2024 NVIDIA Corporation.
// Licensed under the MIT License.

#include "NvDmlExecutionProvider.h"
#include "core/graph/constants.h"
#include "core/providers/dml/dml_provider_factory_creator.h"
namespace NvDml {
NvDmlExecutionProvider::NvDmlExecutionProvider(ID3D12Device* d3d12Device, ID3D12CommandQueue* commandQueue, Dml::ExecutionContext* context)
    : IExecutionProvider(onnxruntime::kNvDmlExecutionProvider, OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, 0)), m_d3d12Device(d3d12Device), m_commandQueue(commandQueue), m_executionContext(context) {
  m_kernelRegistry = std::make_shared<onnxruntime::KernelRegistry>();
}

NvDmlExecutionProvider::~NvDmlExecutionProvider() {
  ComPtr<ID3D12Device> d3d12_device;
  ORT_THROW_IF_FAILED(m_commandQueue->GetDevice(IID_PPV_ARGS(&d3d12_device)));
  ORT_THROW_IF_FAILED(d3d12_device->SetPrivateDataInterface(dml_execution_context_guid, nullptr));
}

std::vector<std::unique_ptr<onnxruntime::ComputeCapability>>
NvDmlExecutionProvider::GetCapability(
    const onnxruntime::GraphViewer& graph,
    const onnxruntime::IExecutionProvider::IKernelLookup& kernel_lookup) const {
  return onnxruntime::IExecutionProvider::GetCapability(graph, kernel_lookup);
}

}  // namespace NvDml
