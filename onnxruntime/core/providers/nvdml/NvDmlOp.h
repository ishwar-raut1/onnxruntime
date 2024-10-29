#include "core/framework/op_kernel.h"
#include "NvDmlExecutionProvider.h"

namespace NvDml{

    enum DataType{
        FP16,
        FP32
    };
    struct TensorDesc{
        size_t dim[4];
        DataType type ;
    };
    struct DML_RELU_OPERATOR_DESC {
    TensorDesc InputTensor;
    TensorDesc OutputTensor;
};
class NvDmlReluOp : public onnxruntime::OpKernel
{
 public:
    NvDmlReluOp(
        const onnxruntime::OpKernelInfo& kerneInfo
    );

    onnxruntime::Status Compute(onnxruntime::OpKernelContext* context) const override;
    void ExtractTensorInfoFromDefinitions(const onnxruntime::Node& node) ;

    private :
    struct DML_RELU_OPERATOR_DESC m_desc;
    const NvDmlExecutionProvider* m_nvdmlep;
    D3D12_COMPUTE_PIPELINE_STATE_DESC m_computePSODesc = {};
    ComPtr<ID3D12PipelineState> m_computePSO;
    D3D12_CPU_DESCRIPTOR_HANDLE m_srvHandleCPU ;
    D3D12_GPU_DESCRIPTOR_HANDLE m_srvHandleGPU ;
    D3D12_CPU_DESCRIPTOR_HANDLE m_uavHandleCPU ;
    D3D12_GPU_DESCRIPTOR_HANDLE m_uavHandleGPU ;
    ComPtr<ID3D12RootSignature> m_rootSignature;
    ComPtr<ID3D12DescriptorHeap> descriptorHeap;

};

}
