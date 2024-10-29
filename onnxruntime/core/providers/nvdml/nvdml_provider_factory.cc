
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

  auto context = wil::MakeOrThrow<Dml::ExecutionContext>(d3d12_device.Get(), dml_device_, cmd_queue_, true, true);

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
  API_IMPL_END
  return nullptr;
}
