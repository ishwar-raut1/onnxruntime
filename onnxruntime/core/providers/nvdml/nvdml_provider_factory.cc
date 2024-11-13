// Copyright (c) 2024 NVIDIA Corporation.
// Licensed under the MIT License.

#include <dxcore.h>
#include <vector>

#include <DirectML.h>

#include <wil/wrl.h>
#include <wil/result.h>

#include "core/providers/nvdml/nvdml_provider_factory.h"

#include "core/session/allocator_adapters.h"
#include "core/session/ort_apis.h"
#include "core/framework/error_code_helper.h"
#include "core/platform/env.h"
#include "NvDmlExecutionProvider.h"

#include "core/session/abi_session_options_impl.h"
#include "core/providers/dml/dml_provider_factory_creator.h"

#include "core/providers/dml/dml_provider_factory.h"
const OrtDmlApi* GetOrtDmlApi(_In_ uint32_t version) NO_EXCEPTION;

namespace onnxruntime {

struct NvDmlProviderFactory : IExecutionProviderFactory {
  NvDmlProviderFactory(IDMLDevice* dml_device,
                       ID3D12CommandQueue* cmd_queue) : dml_device_(dml_device),
                                                        cmd_queue_(cmd_queue) {}
  ~NvDmlProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  IDMLDevice* dml_device_;
  ID3D12CommandQueue* cmd_queue_;
};

std::unique_ptr<IExecutionProvider> NvDmlProviderFactory::CreateProvider() {

  ComPtr<ID3D12Device> d3d12_device;
  ORT_THROW_IF_FAILED(cmd_queue_->GetDevice(IID_PPV_ARGS(&d3d12_device)));

  // the execution context shared by DML and NVDML EP. It is created NVDML EP and passed to DML EP as private data in D3D12 Devices.
  auto context = wil::MakeOrThrow<Dml::ExecutionContext>(d3d12_device.Get(), dml_device_, cmd_queue_, false, false);
  ORT_THROW_IF_FAILED(d3d12_device->SetPrivateDataInterface(dml_execution_context_guid, context.Get()));
  auto provider = std::make_unique<NvDml::NvDmlExecutionProvider>(d3d12_device.Get(), cmd_queue_, context.Get());

  return provider;
}

std::shared_ptr<onnxruntime::IExecutionProviderFactory> CreateExecutionProviderFactory_NvDml(IDMLDevice* dml_device,
                                                                                             ID3D12CommandQueue* cmd_queue) {
  return std::make_shared<onnxruntime::NvDmlProviderFactory>(dml_device, cmd_queue);
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_NVDML, _In_ OrtSessionOptions* options,
                    _In_ IDMLDevice* dml_device, _In_ ID3D12CommandQueue* cmd_queue) {
  API_IMPL_BEGIN
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_NvDml(dml_device,
                                                                                          cmd_queue));
  const OrtDmlApi& ortDmlApi = *GetOrtDmlApi(/*version=*/0);
  // Append the DirectML execution provider to the session options as NVDML depends on DML
  ortDmlApi.SessionOptionsAppendExecutionProvider_DML1(options, dml_device, cmd_queue);
  API_IMPL_END
  return nullptr;
}

const OrtNvDmlApi* GetOrtNvDmlApi(uint32_t version) {
    static const OrtNvDmlApi api = [&] {
        OrtNvDmlApi api_instance;
        const OrtDmlApi* base_api = GetOrtDmlApi(version);
        std::memcpy(static_cast<OrtDmlApi*>(&api_instance), base_api, sizeof(OrtDmlApi));
        api_instance.SessionOptionsAppendExecutionProvider_NVDML = OrtSessionOptionsAppendExecutionProvider_NVDML;
        return api_instance;
    }();
    return &api;
}
