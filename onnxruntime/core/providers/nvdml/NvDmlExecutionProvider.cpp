#include "NvDmlExecutionProvider.h"
#include "core/graph/constants.h"
namespace NvDml {
NvDmlExecutionProvider::NvDmlExecutionProvider(ID3D12Device* d3d12Device, ID3D12CommandQueue* commandQueue, Dml::ExecutionContext* context)
    : IExecutionProvider(onnxruntime::kNvDmlExecutionProvider, OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, 0))
    , m_d3d12Device(d3d12Device), m_commandQueue(commandQueue), m_executionContext(context) {}

std::vector<std::unique_ptr<onnxruntime::ComputeCapability>>
NvDmlExecutionProvider::GetCapability(
    const onnxruntime::GraphViewer& graph,
    const onnxruntime::IExecutionProvider::IKernelLookup& kernel_lookup) const {
  return onnxruntime::IExecutionProvider::GetCapability(graph, kernel_lookup);
}

}  // namespace NvDml
