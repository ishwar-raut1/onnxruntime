#include "NvDmlOp.h"
#include "NvDmlExecutionProvider.h"
#include <d3dcompiler.h>

#pragma comment(lib, "d3dcompiler.lib")
namespace NvDml{

void ExtractTensorInfo(const onnxruntime::NodeArg& node_arg, TensorDesc& tensor) {
    const onnx::TypeProto* type_proto = node_arg.TypeAsProto();

    if (type_proto == nullptr || !type_proto->has_tensor_type()) {
        std::cerr << "NodeArg does not have a tensor type." << std::endl;
        return;
    }

    const onnx::TypeProto_Tensor& tensor_type = type_proto->tensor_type();

    // Extract the data type
    int32_t data_type = tensor_type.elem_type();

    std::cout << "Data Type: " << data_type << std::endl;

    if(onnx::TensorProto_DataType::TensorProto_DataType_FLOAT16 == data_type )
        {
            tensor.type = DataType::FP16;

        }
        else {
             tensor.type = DataType::FP32;

        }
    if (!tensor_type.has_shape()) {
        std::cerr << "Tensor does not have a shape." << std::endl;
        return;
    }

    const onnx::TensorShapeProto& shape_proto = tensor_type.shape();

    // Extract the shape
    std::cout << "Tensor Shape: [";
    for (int i = 0; i < shape_proto.dim_size(); ++i) {
        const onnx::TensorShapeProto_Dimension& dim = shape_proto.dim(i);
        if (dim.has_dim_value()) {
            std::cout << dim.dim_value();
            tensor.dim[i] = dim.dim_value();
        } else if (dim.has_dim_param()) {
            std::cout << dim.dim_param();
        } else {
            std::cout << "Unknown";
        }
        if (i != shape_proto.dim_size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}


void NvDmlReluOp::ExtractTensorInfoFromDefinitions(const onnxruntime::Node& node) {

    auto input = node.InputDefs()[0];
    ExtractTensorInfo( *input, m_desc.InputTensor);

    auto output = node.OutputDefs()[0];
     ExtractTensorInfo( *output, m_desc.OutputTensor);
}


NvDmlReluOp::NvDmlReluOp(const onnxruntime::OpKernelInfo& kernelInfo)
:OpKernel(kernelInfo)
{
    m_nvdmlep = (const NvDml::NvDmlExecutionProvider *)kernelInfo.GetExecutionProvider();

    auto attributes = kernelInfo.node().GetAttributes();


    ExtractTensorInfoFromDefinitions(kernelInfo.node());

    // ID3D12GraphicsCommandList * commandList =  m_nvdmlep->GetImpl()->getExecutionContext()->getDmlCommandRecorder()->GetCommandList().Get();

    // Define Root Signature
    D3D12_ROOT_SIGNATURE_DESC rootSignatureDesc = {};
    rootSignatureDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    // Descriptor ranges for SRV and UAV
    D3D12_DESCRIPTOR_RANGE descriptorRanges[2] = {};

    // Input buffer as SRV (t0)
    descriptorRanges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    descriptorRanges[0].NumDescriptors = 1;
    descriptorRanges[0].BaseShaderRegister = 0;
    descriptorRanges[0].RegisterSpace = 0;
    descriptorRanges[0].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

    // Output buffer as UAV (u0)
    descriptorRanges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    descriptorRanges[1].NumDescriptors = 1;
    descriptorRanges[1].BaseShaderRegister = 0;
    descriptorRanges[1].RegisterSpace = 0;
    descriptorRanges[1].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

    // Root parameters
    D3D12_ROOT_PARAMETER rootParameters[2];

    // SRV Root Parameter
    rootParameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rootParameters[0].DescriptorTable.NumDescriptorRanges = 1;
    rootParameters[0].DescriptorTable.pDescriptorRanges = &descriptorRanges[0];
    rootParameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    // UAV Root Parameter
    rootParameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rootParameters[1].DescriptorTable.NumDescriptorRanges = 1;
    rootParameters[1].DescriptorTable.pDescriptorRanges = &descriptorRanges[1];
    rootParameters[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    // Finalize Root Signature
    rootSignatureDesc.NumParameters = _countof(rootParameters);
    rootSignatureDesc.pParameters = rootParameters;
    rootSignatureDesc.NumStaticSamplers = 0;
    rootSignatureDesc.pStaticSamplers = nullptr;

    // Serialize Root Signature
    ComPtr<ID3DBlob> serializedRootSig = nullptr;
    ComPtr<ID3DBlob> errorBlob = nullptr;
    HRESULT hr = D3D12SerializeRootSignature(
        &rootSignatureDesc,
        D3D_ROOT_SIGNATURE_VERSION_1,
        &serializedRootSig,
        &errorBlob
    );
    if (FAILED(hr))
    {
        if (errorBlob)
        {
            std::cerr << "Error serializing root signature: " << (char*)errorBlob->GetBufferPointer() << "\n";
        }
    }

    // Create Root Signature

    hr = m_nvdmlep->GetD3D12Device()->CreateRootSignature(
        0,
        serializedRootSig->GetBufferPointer(),
        serializedRootSig->GetBufferSize(),
        IID_PPV_ARGS(&m_rootSignature)
    );
    if (FAILED(hr))
    {
        std::cerr << "Failed to create Root Signature.\n";

    }

    // Compile Compute Shader
    const char* shaderCode = R"(
        RWStructuredBuffer<float> OutputBuffer : register(u0);
        StructuredBuffer<float> InputBuffer : register(t0);

        [numthreads(256, 1, 1)]
        void main(uint3 DTid : SV_DispatchThreadID)
        {
            float value = InputBuffer[DTid.x];
            OutputBuffer[DTid.x] = max(0.0, value); // ReLU operation
        }
    )";

    ComPtr<ID3DBlob> computeShader;
    hr = D3DCompile(
        shaderCode,
        strlen(shaderCode),
        nullptr,
        nullptr,
        nullptr,
        "main",
        "cs_5_0",
        0,
        0,
        &computeShader,
        &errorBlob
    );
    if (FAILED(hr))
    {
        if (errorBlob)
        {
            std::cerr << "Shader Compilation Error: " << (char*)errorBlob->GetBufferPointer() << "\n";
        }

    }

    // Create Compute Pipeline State Object (PSO)

    m_computePSODesc.pRootSignature = m_rootSignature.Get();
    m_computePSODesc.CS = { computeShader->GetBufferPointer(), computeShader->GetBufferSize() };

    hr = m_nvdmlep->GetD3D12Device()->CreateComputePipelineState(&m_computePSODesc, IID_PPV_ARGS(&m_computePSO));
    if (FAILED(hr))
    {
        std::cerr << "Failed to create Compute PSO.\n";

    }


    // Create Descriptor Heap for SRV and UAV
    D3D12_DESCRIPTOR_HEAP_DESC descriptorHeapDesc = {};
    descriptorHeapDesc.NumDescriptors = 2; // SRV and UAV
    descriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    descriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;

    hr = m_nvdmlep->GetD3D12Device()->CreateDescriptorHeap(&descriptorHeapDesc, IID_PPV_ARGS(&descriptorHeap));
    if (FAILED(hr))
    {
        std::cerr << "Failed to create Descriptor Heap.\n";

    }

    // Get Descriptor Sizes and Handles
    UINT descriptorSize = m_nvdmlep->GetD3D12Device()->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    m_srvHandleCPU = descriptorHeap->GetCPUDescriptorHandleForHeapStart();
    m_srvHandleGPU = descriptorHeap->GetGPUDescriptorHandleForHeapStart();
    m_uavHandleCPU = { m_srvHandleCPU.ptr + descriptorSize };
    m_uavHandleGPU = { m_srvHandleGPU.ptr + descriptorSize };


}

onnxruntime::Status NvDmlReluOp::Compute(onnxruntime::OpKernelContext* context) const {


   int64_t arr[] = {(int64_t)m_desc.OutputTensor.dim[0],(int64_t)m_desc.OutputTensor.dim[1],(int64_t)m_desc.OutputTensor.dim[2],(int64_t)m_desc.OutputTensor.dim[3]};
   uint32_t elementCount = (uint32_t)(arr[0] * arr[1] * arr[2] * arr[3]);

   gsl::span<int64_t> mySpan(arr);
//  using namespace Dml;

   auto input = context->Input<onnxruntime::Tensor>(gsl::narrow_cast<int>(0));
   onnxruntime::TensorShape shape(mySpan);
   auto out = context->Output(0, shape);



   const void* inpudata = (const void* )input->Data< float>();
   ID3D12Resource* inputResource = ((const Dml::AllocationInfo*)inpudata)->GetResource();


   const void* outdata = (const void* )out->Data<float>();
   ID3D12Resource* outResource = ((const Dml::AllocationInfo*)outdata)->GetResource();


   ID3D12GraphicsCommandList * commandList = nullptr;
   m_nvdmlep->GetExecutionContext()->GetCommandListForRecordingAndInvalidateState(&commandList);



  //       auto filter = context->Input(1);
//  std::unique_ptr<onnxruntime::IDataTransfer> dataTransfer = m_nvdmlep->GetDataTransfer();
//  auto status = m_nvdmlep->GetDataTransfer()->CopyTensor(*input, *out);

// Create SRV for Input Buffer
    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Format = DXGI_FORMAT_UNKNOWN;
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srvDesc.Buffer.FirstElement = 0;
    srvDesc.Buffer.NumElements = elementCount;
    srvDesc.Buffer.StructureByteStride = sizeof(float);
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    m_nvdmlep->GetD3D12Device()->CreateShaderResourceView(inputResource, &srvDesc, m_srvHandleCPU);

    // Create UAV for Output Buffer
    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.Format = DXGI_FORMAT_UNKNOWN;
    uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    uavDesc.Buffer.FirstElement = 0;
    uavDesc.Buffer.NumElements = elementCount;
    uavDesc.Buffer.StructureByteStride = sizeof(float);
    m_nvdmlep->GetD3D12Device()->CreateUnorderedAccessView(outResource, nullptr, &uavDesc, m_uavHandleCPU);

    // Set Pipeline State and Root Signature
    commandList->SetComputeRootSignature(m_rootSignature.Get());
    commandList->SetPipelineState(m_computePSO.Get());

    // Set Descriptor Heaps
    ID3D12DescriptorHeap* descriptorHeaps[] = { descriptorHeap.Get() };
    commandList->SetDescriptorHeaps(_countof(descriptorHeaps), descriptorHeaps);

    // Set Root Descriptor Tables
    commandList->SetComputeRootDescriptorTable(0, m_srvHandleGPU); // SRV at t0
    commandList->SetComputeRootDescriptorTable(1, m_uavHandleGPU); // UAV at u0

    // Dispatch Compute Shader
    UINT dispatchSize = (elementCount + 255) / 256; // Match numthreads in shader
    commandList->Dispatch(dispatchSize, 1, 1);

  return onnxruntime::Status::OK();
}

}  // namespace nvdml
