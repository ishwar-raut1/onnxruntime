#include "NvDmlExecutionProvider.h"
#include "core/graph/constants.h"
namespace NvDml {
NvDmlExecutionProvider::NvDmlExecutionProvider(IDMLDevice* /*dmlDevice*/, ID3D12CommandQueue* /*commandQueue*/) : IExecutionProvider(onnxruntime::kNvDmlExecutionProvider, OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, 0)) {
}

std::vector<std::unique_ptr<onnxruntime::ComputeCapability>>
NvDmlExecutionProvider::GetCapability(
    const onnxruntime::GraphViewer& graph,
    const onnxruntime::IExecutionProvider::IKernelLookup& kernel_lookup) const {
  return onnxruntime::IExecutionProvider::GetCapability(graph, kernel_lookup);
}

}  // namespace NvDml
