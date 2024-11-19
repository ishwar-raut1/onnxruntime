// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once


#include "core/framework/provider_options.h"
#include "core/providers/providers.h"
#include "core/providers/nvdml/nvdml_provider_factory.h"
#include "core/framework/config_options.h"

#include <directx/dxcore.h>
#include <vector>


namespace onnxruntime {

struct NVDMLProviderFactoryCreator {
  static std::shared_ptr<IExecutionProviderFactory> Create(
    const ConfigOptions& config_options);
};
}  // namespace onnxruntime
