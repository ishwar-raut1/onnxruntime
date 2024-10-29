#include "NvDmlExecutionProvider.h"
#include "core/graph/constants.h"
namespace NvDml {

NvDmlExecutionProvider::NvDmlExecutionProvider(ID3D12Device* dmlDevice, ID3D12CommandQueue* executionContext, Dml::ExecutionContext* context): IExecutionProvider(onnxruntime::kNvDmlExecutionProvider, OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, 0)) {
  d3d12_device_ = dmlDevice;
  cmd_queue_ = executionContext;
  context_.reset(context);

}


std::vector<std::unique_ptr<onnxruntime::ComputeCapability>>
NvDmlExecutionProvider::GetCapability(
    const onnxruntime::GraphViewer& graph,
    const onnxruntime::IExecutionProvider::IKernelLookup& kernel_lookup) const {
  return onnxruntime::IExecutionProvider::GetCapability(graph, kernel_lookup);
}

}  // namespace NvDml
