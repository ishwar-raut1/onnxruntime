// Copyright (c) 2024 NVIDIA Corporation.
// Licensed under the MIT License.

#include <vector>

#define INITGUID
#include <guiddef.h>
#include <directx/dxcore.h>
#undef INITGUID

#include "directx/d3d12.h"
#include <DirectML.h>

#include <wil/wrl.h>
#include <wil/result.h>

#include "core/providers/nvdml/nvdml_provider_factory.h"

#include "core/session/allocator_adapters.h"
#include "core/session/ort_apis.h"
#include "core/framework/error_code_helper.h"
#include "core/platform/env.h"
#include "NvDmlExecutionProvider.h"
#include "NvDmlFactoryCreator.h"

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
  ComPtr<IDMLDevice> dml_device_;
  ComPtr<ID3D12CommandQueue> cmd_queue_;
};

std::unique_ptr<IExecutionProvider> NvDmlProviderFactory::CreateProvider() {
  ComPtr<ID3D12Device> d3d12_device;
  ORT_THROW_IF_FAILED(cmd_queue_->GetDevice(IID_PPV_ARGS(&d3d12_device)));

  // the execution context shared by DML and NVDML EP. It is created NVDML EP and passed to DML EP as private data in D3D12 Devices.
  auto context = wil::MakeOrThrow<Dml::ExecutionContext>(d3d12_device.Get(), dml_device_.Get(), cmd_queue_.Get(), false, false);
  ORT_THROW_IF_FAILED(d3d12_device->SetPrivateDataInterface(dml_execution_context_guid, context.Get()));
  auto provider = std::make_unique<NvDml::NvDmlExecutionProvider>(d3d12_device.Get(), cmd_queue_.Get(), context.Get());

  return provider;
}

std::shared_ptr<onnxruntime::IExecutionProviderFactory> CreateExecutionProviderFactory_NvDml(IDMLDevice* dml_device,
                                                                                             ID3D12CommandQueue* cmd_queue) {
  return std::make_shared<onnxruntime::NvDmlProviderFactory>(dml_device, cmd_queue);
}
static ComPtr<IDXCoreAdapterList> EnumerateDXCoreAdapters(IDXCoreAdapterFactory* adapter_factory) {
  ComPtr<IDXCoreAdapterList> adapter_list;

  ORT_THROW_IF_FAILED(
      adapter_factory->CreateAdapterList(1,
                                         &DXCORE_ADAPTER_ATTRIBUTE_D3D12_GENERIC_ML,
                                         adapter_list.GetAddressOf()));

  if (adapter_list->GetAdapterCount() == 0) {
    ORT_THROW_IF_FAILED(adapter_factory->CreateAdapterList(1, &DXCORE_ADAPTER_ATTRIBUTE_D3D12_CORE_COMPUTE, adapter_list.GetAddressOf()));
  }

  return adapter_list;
}

static void SortDXCoreAdaptersByPreference(
    IDXCoreAdapterList* adapter_list) {
  if (adapter_list->GetAdapterCount() <= 1) {
    return;
  }

  // DML prefers the HighPerformance adapter by default
  std::array<DXCoreAdapterPreference, 1> adapter_list_preferences = {
      DXCoreAdapterPreference::HighPerformance};

  ORT_THROW_IF_FAILED(adapter_list->Sort(
      static_cast<uint32_t>(adapter_list_preferences.size()),
      adapter_list_preferences.data()));
}

static bool IsHardwareAdapter(IDXCoreAdapter* adapter) {
  bool is_hardware = false;
  THROW_IF_FAILED(adapter->GetProperty(
      DXCoreAdapterProperty::IsHardware,
      &is_hardware));
  return is_hardware;
}

static bool IsGPU(IDXCoreAdapter* compute_adapter) {
  // Only considering hardware adapters
  if (!IsHardwareAdapter(compute_adapter)) {
    return false;
  }
  return compute_adapter->IsAttributeSupported(DXCORE_ADAPTER_ATTRIBUTE_D3D12_GRAPHICS);
}

static std::vector<ComPtr<IDXCoreAdapter>> FilterDXCoreAdapters(
    IDXCoreAdapterList* adapter_list) {
  auto adapters = std::vector<ComPtr<IDXCoreAdapter>>();
  const uint32_t count = adapter_list->GetAdapterCount();
  for (uint32_t i = 0; i < count; ++i) {
    ComPtr<IDXCoreAdapter> candidate_adapter;
    ORT_THROW_IF_FAILED(adapter_list->GetAdapter(i, candidate_adapter.GetAddressOf()));
    if (IsGPU(candidate_adapter.Get())) {
      adapters.push_back(candidate_adapter);
    }
  }

  return adapters;
}
static D3D12_COMMAND_LIST_TYPE CalculateCommandListType(ID3D12Device* d3d12_device) {
  D3D12_FEATURE_DATA_FEATURE_LEVELS feature_levels = {};

  D3D_FEATURE_LEVEL feature_levels_list[] = {
#ifndef _GAMING_XBOX
      D3D_FEATURE_LEVEL_1_0_GENERIC,
#endif
      D3D_FEATURE_LEVEL_1_0_CORE,
      D3D_FEATURE_LEVEL_11_0,
      D3D_FEATURE_LEVEL_11_1,
      D3D_FEATURE_LEVEL_12_0,
      D3D_FEATURE_LEVEL_12_1};

  feature_levels.NumFeatureLevels = ARRAYSIZE(feature_levels_list);
  feature_levels.pFeatureLevelsRequested = feature_levels_list;
  ORT_THROW_IF_FAILED(d3d12_device->CheckFeatureSupport(
      D3D12_FEATURE_FEATURE_LEVELS,
      &feature_levels,
      sizeof(feature_levels)));

  auto use_compute_command_list = (feature_levels.MaxSupportedFeatureLevel <= D3D_FEATURE_LEVEL_1_0_CORE);

  if (use_compute_command_list) {
    return D3D12_COMMAND_LIST_TYPE_COMPUTE;
  }

  return D3D12_COMMAND_LIST_TYPE_DIRECT;
}
std::shared_ptr<IExecutionProviderFactory> NVDMLProviderFactoryCreator::Create(const ConfigOptions&) {
  // Create DXCore Adapter Factory
  ComPtr<IDXCoreAdapterFactory> adapter_factory;
  ORT_THROW_IF_FAILED(::DXCoreCreateAdapterFactory(adapter_factory.GetAddressOf()));

  // Get all DML compatible DXCore adapters
  ComPtr<IDXCoreAdapterList> adapter_list;
  adapter_list = EnumerateDXCoreAdapters(adapter_factory.Get());

  if (adapter_list->GetAdapterCount() == 0) {
    ORT_THROW("No GPUs detected.");
  }

  // Sort the adapter list to honor DXCore hardware ordering
  SortDXCoreAdaptersByPreference(adapter_list.Get());

  std::vector<ComPtr<IDXCoreAdapter>> adapters;
  // Filter all DXCore adapters to hardware type specified by the device filter
  adapters = FilterDXCoreAdapters(adapter_list.Get());
  if (adapters.size() == 0) {
    ORT_THROW("No devices detected that match the filter criteria.");
  }
  // Choose the first device from the list since it's the highest priority
  auto adapter = adapters[0];

  auto feature_level = D3D_FEATURE_LEVEL_11_0;

  // Create D3D12 Device from DXCore Adapter
  ComPtr<ID3D12Device> d3d12_device;

  ORT_THROW_IF_FAILED(D3D12CreateDevice(adapter.Get(), feature_level, IID_PPV_ARGS(d3d12_device.ReleaseAndGetAddressOf())));

  D3D12_COMMAND_QUEUE_DESC cmd_queue_desc = {};
  cmd_queue_desc.Type = CalculateCommandListType(d3d12_device.Get());
  cmd_queue_desc.Flags = D3D12_COMMAND_QUEUE_FLAG_DISABLE_GPU_TIMEOUT;

  ComPtr<ID3D12CommandQueue> cmd_queue;
  ORT_THROW_IF_FAILED(d3d12_device->CreateCommandQueue(&cmd_queue_desc, IID_PPV_ARGS(cmd_queue.ReleaseAndGetAddressOf())));

  ComPtr<IDMLDevice> dml_device;
  dml_device = onnxruntime::DMLProviderFactoryCreator::CreateDMLDevice(d3d12_device.Get());

  return CreateExecutionProviderFactory_NvDml(dml_device.Get(), cmd_queue.Get());
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
