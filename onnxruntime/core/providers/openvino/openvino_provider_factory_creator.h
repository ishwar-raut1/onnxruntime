// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "core/providers/providers.h"
#include "core/framework/provider_options.h"

struct OrtOpenVINOProviderOptions;

namespace onnxruntime {
struct SessionOptions;
// defined in provider_bridge_ort.cc
struct OpenVINOProviderFactoryCreator {
  static std::shared_ptr<IExecutionProviderFactory> Create(const ProviderOptions* provider_options_map,
                                                           const SessionOptions* session_options);
};
}  // namespace onnxruntime
