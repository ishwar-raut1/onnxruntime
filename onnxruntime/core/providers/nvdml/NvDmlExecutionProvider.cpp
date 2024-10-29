#include "NvDmlExecutionProvider.h"
#include "core/graph/constants.h"
#include "NvDmlOp.h"
namespace NvDml {
const char* typeLabel = "T";
NvDmlExecutionProvider::NvDmlExecutionProvider(ID3D12Device* dmlDevice, ID3D12CommandQueue* executionContext, Dml::ExecutionContext* context): IExecutionProvider(onnxruntime::kNvDmlExecutionProvider, OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, 0)) {
  d3d12_device_ = dmlDevice;
  cmd_queue_ = executionContext;
  context_.reset(context);

  m_kernelRegistry = std::make_shared<onnxruntime::KernelRegistry>();

  addOperatorToKernelRegistry();

}

void NvDmlExecutionProvider::addOperatorToKernelRegistry() {

  {
    onnxruntime::KernelDefBuilder builder; // this kernelDefBuilder is used to create KernelDef
    builder.SetName("Relu");                // add the name of operator
    builder.SetDomain("")                  //
    .SinceVersion(7)                        // present since version
    .Provider(onnxruntime::kNvDmlExecutionProvider); // add nvdml proveidee

    std::vector<onnxruntime::MLDataType> types;             // this represents the data type supported by the operator

    types.reserve(2); // only 1 type count is used
    types.push_back(onnxruntime::DataTypeImpl::GetTensorType<float>());   // add float
    types.push_back(onnxruntime::DataTypeImpl::GetTensorType<onnxruntime::MLFloat16>());     // add the FP16 data types
    builder.TypeConstraint(typeLabel, types );          // add label denotting the data. here "T" is used denoting the tensor

    auto creationFunction = [](onnxruntime::FuncManager&, const onnxruntime::OpKernelInfo& info, std::unique_ptr<onnxruntime::OpKernel>& out) -> onnxruntime::common::Status
            {
                out = std::make_unique<NvDmlReluOp>(info);
                return Status::OK();
            };
            onnxruntime::KernelCreateInfo create_info(builder.Build(), creationFunction);

            auto status = m_kernelRegistry->Register(std::move(create_info)); // register kernel creation info for conv kenrel
            if(status != Status::OK())
            {
                std::cout<<"nvdml operator cant be added to kernel registry\n";
            }

  }

  {
    onnxruntime::KernelDefBuilder builder; // this kernelDefBuilder is used to create KernelDef
    builder.SetName("Relu");                // add the name of operator
    builder.SetDomain("")                  //
    .SinceVersion(13)                        // present since version
    .Provider(onnxruntime::kNvDmlExecutionProvider); // add nvdml proveidee

    std::vector<onnxruntime::MLDataType> types;             // this represents the data type supported by the operator

    types.reserve(2); // only 1 type count is used
    types.push_back(onnxruntime::DataTypeImpl::GetTensorType<float>());   // add float
    types.push_back(onnxruntime::DataTypeImpl::GetTensorType<onnxruntime::MLFloat16>());     // add the FP16 data types
    builder.TypeConstraint(typeLabel, types );          // add label denotting the data. here "T" is used denoting the tensor

    auto creationFunction = [](onnxruntime::FuncManager&, const onnxruntime::OpKernelInfo& info, std::unique_ptr<onnxruntime::OpKernel>& out) -> onnxruntime::common::Status
            {
                out = std::make_unique<NvDmlReluOp>(info);
                return Status::OK();
            };
            onnxruntime::KernelCreateInfo create_info(builder.Build(), creationFunction);

            auto status = m_kernelRegistry->Register(std::move(create_info)); // register kernel creation info for conv kenrel
            if(status != Status::OK())
            {
                std::cout<<"nvdml operator cant be added to kernel registry\n";
            }

  }
  {
    onnxruntime::KernelDefBuilder builder; // this kernelDefBuilder is used to create KernelDef
    builder.SetName("Relu");                // add the name of operator
    builder.SetDomain("")                  //
    .SinceVersion(14)                        // present since version
    .Provider(onnxruntime::kNvDmlExecutionProvider); // add nvdml proveidee

    std::vector<onnxruntime::MLDataType> types;             // this represents the data type supported by the operator

    types.reserve(2); // only 1 type count is used
    types.push_back(onnxruntime::DataTypeImpl::GetTensorType<float>());   // add float
    types.push_back(onnxruntime::DataTypeImpl::GetTensorType<onnxruntime::MLFloat16>());     // add the FP16 data types
    builder.TypeConstraint(typeLabel, types );          // add label denotting the data. here "T" is used denoting the tensor

    auto creationFunction = [](onnxruntime::FuncManager&, const onnxruntime::OpKernelInfo& info, std::unique_ptr<onnxruntime::OpKernel>& out) -> onnxruntime::common::Status
            {
                out = std::make_unique<NvDmlReluOp>(info);
                return Status::OK();
            };
            onnxruntime::KernelCreateInfo create_info(builder.Build(), creationFunction);

            auto status = m_kernelRegistry->Register(std::move(create_info)); // register kernel creation info for conv kenrel
            if(status != Status::OK())
            {
                std::cout<<"nvdml operator cant be added to kernel registry\n";
            }

  }
}

std::vector<std::unique_ptr<onnxruntime::ComputeCapability>>
NvDmlExecutionProvider::GetCapability(
    const onnxruntime::GraphViewer& graph,
    const onnxruntime::IExecutionProvider::IKernelLookup& kernel_lookup) const {
  return onnxruntime::IExecutionProvider::GetCapability(graph, kernel_lookup);
}

}  // namespace NvDml
