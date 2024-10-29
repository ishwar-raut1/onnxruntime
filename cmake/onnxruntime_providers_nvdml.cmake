# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

file(GLOB_RECURSE onnxruntime_providers_nvdml_cc_srcs CONFIGURE_DEPENDS
  "${ONNXRUNTIME_ROOT}/core/providers/nvdml/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/nvdml/*.cpp"
    "${ONNXRUNTIME_ROOT}/core/providers/nvdml/*.cc"
)

source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_nvdml_cc_srcs})
onnxruntime_add_static_library(onnxruntime_providers_nvdml ${onnxruntime_providers_nvdml_cc_srcs} )
onnxruntime_add_include_to_target(onnxruntime_providers_nvdml onnxruntime_common onnx flatbuffers::flatbuffers Boost::mp11 safeint_interface)
add_dependencies(onnxruntime_providers_nvdml onnxruntime_providers_shared ${onnxruntime_EXTERNAL_DEPENDENCIES})

target_include_directories(onnxruntime_providers_nvdml PRIVATE ${ONNXRUNTIME_ROOT} ${CMAKE_CURRENT_BINARY_DIR} ${eigen_INCLUDE_DIRS} )

set_target_properties(onnxruntime_providers_nvdml PROPERTIES FOLDER "ONNXRuntime")

install(FILES ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/providers/nvdml/nvdml_provider_factory.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/
  )

if (NOT onnxruntime_BUILD_SHARED_LIB)
  install(TARGETS onnxruntime_providers_nvdml
          ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
          LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}
          FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
