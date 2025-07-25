// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#include "vitisai_execution_provider.h"
#include "vitisai_profiler.h"

// Standard headers/libs.
#include <cassert>
#include <fstream>
#include <istream>
#include <filesystem>

// 1st-party headers/libs.
#include "core/common/exceptions.h"
#include "core/platform/env_var_utils.h"
#include "core/providers/qnn/ort_api.h"

#include "vaip/capability.h"
#include "vaip/global_api.h"

using namespace ONNX_NAMESPACE;

namespace fs = std::filesystem;

namespace onnxruntime {
constexpr const char* VITISAI = "VITISAI";

VitisAIExecutionProvider::VitisAIExecutionProvider(
    const ProviderOptions& info)
    : IExecutionProvider{onnxruntime::kVitisAIExecutionProvider,
                         OrtDevice(OrtDevice::CPU, OrtDevice::MemType::DEFAULT, OrtDevice::VendorIds::NONE,
                                   DEFAULT_CPU_ALLOCATOR_DEVICE_ID)},
      info_(info) {  // Removed 4k alignment for now, need better fix
  auto it = info_.find("ep_context_enable");
  ep_ctx_enabled_ = it != info_.end() && it->second == "1";
  it = info_.find("ep_context_embed_mode");
  ep_ctx_embed_mode_ = it != info_.end() && it->second != "0";
  // ep_ctx_embed_mode_ = it == info_.end() || it->second != "0";
  it = info_.find("ep_context_file_path");
  ep_ctx_model_path_cfg_ = it == info_.end() ? "" : it->second;
  LOGS_DEFAULT(VERBOSE) << "EP Context cache enabled: " << ep_ctx_enabled_;
  LOGS_DEFAULT(VERBOSE) << "EP context cache embed mode: " << ep_ctx_embed_mode_;
  LOGS_DEFAULT(VERBOSE) << "User specified EP context cache path: " << ep_ctx_model_path_cfg_;
}

std::shared_ptr<KernelRegistry> VitisAIExecutionProvider::GetKernelRegistry() const { return get_kernel_registry_vitisaiep(); }

// This method is called after both `GetComputeCapabilityOps()` and `Compile()`.
// This timing is required to work with both compilation-based EPs and non-compilation-based EPs.
const InlinedVector<const Node*> VitisAIExecutionProvider::GetEpContextNodes() const {
  InlinedVector<const Node*> ep_context_node_ptrs;
  auto nodes = create_ep_context_nodes(**execution_providers_);
  if (nodes.has_value()) {
    ep_context_node_ptrs.assign(nodes->begin(), nodes->end());
  }
  return ep_context_node_ptrs;
}
std::vector<std::unique_ptr<ComputeCapability>> VitisAIExecutionProvider::GetCapability(
    const onnxruntime::GraphViewer& graph_viewer, const IKernelLookup& kernel_lookup, const GraphOptimizerRegistry& /* graph_optimizer_registry */, IResourceAccountant* /* resource_accountant */) const {
  if (graph_viewer.IsSubgraph()) {
    // VITIS AI EP not support sungraph. Assigned to CPU.
    return {};
  }
  if (execution_providers_) {
    // Only compiling a model once is currently supported
    return {};
  }
  execution_providers_ = std::make_unique<my_ep_t>(compile_onnx_model(graph_viewer, *GetLogger(), info_));
  auto result = vaip::GetComputeCapabilityOps(graph_viewer, execution_providers_.get(), kernel_lookup);
  size_t index = 0u;
  for (auto& ep : **execution_providers_) {
    result.emplace_back(vaip::XirSubgraphToComputeCapability1(graph_viewer, ep.get(), index));
    index = index + 1;
  }
  return result;
}

common::Status VitisAIExecutionProvider::Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                                                 std::vector<NodeComputeInfo>& node_compute_funcs) {
  for (const auto& fused_node_graph : fused_nodes_and_graphs) {
    NodeComputeInfo compute_info;
    auto& attrs = fused_node_graph.fused_node.get().GetAttributes();
    assert(attrs.count("index"));
    size_t index = attrs.at("index").i();
    auto& ep = (**this->execution_providers_)[index];
    ep->set_fused_node(&fused_node_graph.fused_node.get());
    if (ep->get_meta_def_fallback_CPU()) {
      auto& subgraph = fused_node_graph.filtered_graph.get();
      auto& logger = logging::LoggingManager::DefaultLogger();
      auto model_proto = subgraph.CreateModel(logger)->ToProto();
      subgraph.ToProto(*model_proto->mutable_graph(), true, true);
      auto local_registries = IOnnxRuntimeOpSchemaRegistryList{subgraph.GetSchemaRegistry()};
      auto model = Model::Create(std::move(*model_proto), subgraph.ModelPath(), &local_registries, logger);
      ep->set_model(model.release());
    }
    compute_info.create_state_func = [this, index](ComputeContext* context, FunctionState* state) {
      auto* p = (**this->execution_providers_)[index]->compile().release();
      *state = p;
      return 0;
    };

    compute_info.release_state_func = [](FunctionState state) {
      if (state) {
        delete reinterpret_cast<vaip_core::CustomOp*>(state);
      }
    };
    compute_info.compute_func = [](FunctionState state, const OrtApi* api, OrtKernelContext* context) {
      reinterpret_cast<vaip_core::CustomOp*>(state)->Compute(api, context);
      return Status::OK();
    };
    node_compute_funcs.push_back(compute_info);
  }
  return Status::OK();
}

common::Status VitisAIExecutionProvider::OnRunStart(const onnxruntime::RunOptions& run_options) {
  InlinedVector<const Node*> ep_context_node_ptrs;
  auto get_config_entry = [](const void* state, const char* entry_name) -> vaip_core::DllSafe<std::string> {
    const onnxruntime::RunOptions& run_options = *static_cast<const onnxruntime::RunOptions*>(state);
    auto ret = run_options.GetConfigOptions().GetConfigEntry(std::string(entry_name));
    if (ret) {
      return vaip_core::DllSafe<std::string>(new std::string(ret.value()));
    } else {
      return {};
    };
  };
  auto error_code = vitisai_ep_on_run_start(**execution_providers_, (const void*)&run_options, get_config_entry);
  if (error_code) {
    std::string error_msg = "vitisai_ep_on_run_start ret: " + std::to_string(error_code);
    return Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::StatusCode::FAIL, error_msg);
  }
  return Status::OK();
}

common::Status VitisAIExecutionProvider::SetEpDynamicOptions(gsl::span<const char* const> keys,
                                                             gsl::span<const char* const> values) {
  auto error_code = vitisai_ep_set_ep_dynamic_options(**execution_providers_, keys.data(), values.data(), std::min(keys.size(), values.size()));
  if (error_code) {
    std::string error_msg = "vitisai_ep_set_ep_dynamic_options ret: " + std::to_string(error_code);
    return Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::StatusCode::FAIL, error_msg);
  }
  return Status::OK();
}

std::unique_ptr<profiling::EpProfiler> VitisAIExecutionProvider::GetProfiler() {
  return std::make_unique<profiling::VitisaiProfiler>();
}

std::vector<AllocatorPtr> VitisAIExecutionProvider::CreatePreferredAllocators() {
  std::vector<AllocatorPtr> result;
  // We do not want arena for 4k alignment, as it would not respect alignment.
  // For CPU, use arena
  // Removed 4k alignment for now, need better fix
  constexpr const bool use_arena_true = true;
  AllocatorCreationInfo device_info_cpu{
      [](OrtDevice::DeviceId device_id) {
        return std::make_unique<CPUAllocator>(
            OrtMemoryInfo(
                onnxruntime::CPU, OrtAllocatorType::OrtDeviceAllocator,
                OrtDevice(OrtDevice::CPU, OrtDevice::MemType::DEFAULT, OrtDevice::VendorIds::NONE,
                          device_id)));
      },
      DEFAULT_CPU_ALLOCATOR_DEVICE_ID, use_arena_true};

  result.push_back(CreateAllocator(device_info_cpu));
  return result;
}

}  // namespace onnxruntime
