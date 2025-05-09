// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This contains the utility functions which will be used to build a coreml model

#pragma once

#include <optional>

#include <gsl/gsl>
#include "core/common/status.h"
#include "core/graph/basic_types.h"
#include "core/providers/common.h"
#include "core/providers/coreml/builders/coreml_spec.h"
#include "core/providers/shared/utils/utils.h"

namespace onnxruntime {
class NodeArg;

namespace coreml {
class ModelBuilder;

// Try to see if we can map explicit padding to auto padding for Conv/Pool
// Since usually use auto padding is more efficient
Status HandleAutoPad(const std::vector<int64_t> input_shape,
                     const int64_t weight_size_y,
                     const int64_t weight_size_x,
                     const std::vector<int64_t>& onnx_pads,
                     const std::vector<int64_t>& onnx_strides,
                     const std::vector<int64_t>& onnx_dilations,
                     AutoPadType auto_pad_type,
                     AutoPadType& auto_pad_type_out);

//
// NeuralNetwork utils
//

// Copy an onnx initializer data to a coreml weight
Status CreateCoreMLWeight(CoreML::Specification::WeightParams& weight, const ONNX_NAMESPACE::TensorProto& tensor);

// Copy the float array to a coreml weight
void CreateCoreMLWeight(CoreML::Specification::WeightParams& weight, gsl::span<const float> data);

// Copy the MLFloat16 array to a coreml weight
void CreateCoreMLWeight(CoreML::Specification::WeightParams& weight, gsl::span<const MLFloat16> data);

// Copy the int32_t array to a coreml weight
void CreateCoreMLWeight(CoreML::Specification::WeightParams& weight, gsl::span<const int32_t> data);

// Copy the int64_t array to a coreml weight
void CreateCoreMLWeight(CoreML::Specification::WeightParams& weight, gsl::span<const int64_t> data);

//
// MLProgram utils
//

// helper for static_assert where the value needs to be dependent on a template parameter
template <typename>
constexpr bool false_for_T = false;

template <typename T>
COREML_SPEC::MILSpec::DataType DataTypeToMILSpec() {
  if constexpr (std::is_same_v<T, float>) {
    return COREML_SPEC::MILSpec::DataType::FLOAT32;
  } else if constexpr (std::is_same_v<T, double>) {
    return COREML_SPEC::MILSpec::DataType::FLOAT64;
  } else if constexpr (std::is_same_v<T, BFloat16>) {
    return COREML_SPEC::MILSpec::DataType::BFLOAT16;
  } else if constexpr (std::is_same_v<T, MLFloat16>) {
    return COREML_SPEC::MILSpec::DataType::FLOAT16;

  } else if constexpr (std::is_same_v<T, int8_t>) {
    return COREML_SPEC::MILSpec::DataType::INT8;
  } else if constexpr (std::is_same_v<T, int16_t>) {
    return COREML_SPEC::MILSpec::DataType::INT16;
  } else if constexpr (std::is_same_v<T, int32_t>) {
    return COREML_SPEC::MILSpec::DataType::INT32;
  } else if constexpr (std::is_same_v<T, int64_t>) {
    return COREML_SPEC::MILSpec::DataType::INT64;

  } else if constexpr (std::is_same_v<T, uint8_t>) {
    return COREML_SPEC::MILSpec::DataType::UINT8;
  } else if constexpr (std::is_same_v<T, uint16_t>) {
    return COREML_SPEC::MILSpec::DataType::UINT16;
  } else if constexpr (std::is_same_v<T, uint32_t>) {
    return COREML_SPEC::MILSpec::DataType::UINT32;
  } else if constexpr (std::is_same_v<T, uint64_t>) {
    return COREML_SPEC::MILSpec::DataType::UINT64;

  } else if constexpr (std::is_same_v<T, bool>) {
    return COREML_SPEC::MILSpec::DataType::BOOL;
  } else if constexpr (std::is_same_v<T, std::string>) {
    return COREML_SPEC::MILSpec::DataType::STRING;
  } else {
    static_assert(false_for_T<T>, "Unsupported type.");
  }
}

// The TensorProto.data_type field is an int, but must be a valid TensorProto_DataType value.
// Use int for the arg so the caller can pass TensorProto.data_type() value and do the cast to enum internally
// This method also automatically converts int64 to int32 since only int32 is supported for CoreML operations.
COREML_SPEC::MILSpec::DataType OnnxDataTypeToMILSpec(int onnx_type);

/// <summary>
/// Create a CoreML MILSpec::TensorValue for the given input data.
/// </summary>
/// <typeparam name="T1">Original C++ data type</typeparam>
/// <typeparam name="T2">CoreML C++ data type</typeparam>
/// <param name="data">ONNX data</param>
/// <param name="shape">ONNX data shape. Inferred to be a 1D shape of `{data.size()}` if not specified.</param>
/// <returns>TensorValue containing data.</returns>
template <typename T1, typename T2 = T1>
COREML_SPEC::MILSpec::Value CreateTensorValue(gsl::span<const T1> data,
                                              std::optional<gsl::span<const int64_t>> shape = std::nullopt);

template <typename T>
COREML_SPEC::MILSpec::Value CreateScalarTensorValue(const T& data);

/// <summary>Create a NamedValueType from an ONNX tensor NodeArg.</summary>
/// <param name="node_arg">NodeArg to create NamedValueType from.</param>
/// <param name="convert_scalar">If true, scalar shapes are converted to 1D.</param>
/// <remarks>Used to create inputs for the 'main' function in an ML Program.</remarks>
COREML_SPEC::MILSpec::NamedValueType CreateNamedTensorValueType(const NodeArg& node_arg, bool convert_scalar = false);

/// <summary>
/// Add an input argument to a MILSpec::Operation
/// </summary>
/// <param name="op">Operation to update.</param>
/// <param name="input_name">The input name defined by the spec for the operation.</param>
/// <param name="value_name">The name of the value that is providing the input.</param>
/// <see>"https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html"</see>
void AddOperationInput(COREML_SPEC::MILSpec::Operation& op,
                       std::string_view input_name, std::string_view value_name);

/// <summary>
/// Add a variadic input argument to a MILSpec::Operation
/// </summary>
/// <param name="op">Operation to update.</param>
/// <param name="input name">The input name defined by the spec for the operation. </param>
/// <param name="value_names">The input value names.</param>
void AddOperationVariadicInput(COREML_SPEC::MILSpec::Operation& op, std::string_view input_name,
                               const std::vector<std::string_view>& value_names);

/// Add an output to a MILSpec::Operation for an intermediate operation when the implementation is composed of
/// multiple MLProgram operations. In this case we don't have a NodeArg for the output.
/// </summary>
/// <param name="op">Operation to update.</param>
/// <param name="output_name">Name of the intermediate output. Create using ModelBuilder::GetUniqueName.</param>
/// <param name="element_type">onnx::TensorProto_DataType element type of the output.
///   int32_t as that is what TensorShapeProto uses to store the value.</param>
/// <param name="shape">Shape of the output if known.</param>
void AddIntermediateOperationOutput(COREML_SPEC::MILSpec::Operation& op, std::string_view output_name,
                                    int32_t element_type, std::optional<gsl::span<const int64_t>> shape);

/// <summary>
/// Add an output to a MILSpec::Operation. Name, data type and shape are used from the NodeArg.
/// </summary>
/// <param name="op">Operation to update.</param>
/// <param name="output">NodeArg with details of output to add.</param>
void AddOperationOutput(COREML_SPEC::MILSpec::Operation& op, const NodeArg& output);

/// <summary>
/// Add pad_type and pad values.
/// </summary>
/// <param name="op">Operator to update</param>
/// <param name="model_builder">ModelBuilder to add constants with.</param>
/// <param name="op_type">Operator type.</param>
/// <param name="helper">Node attribute helper.</param>
/// <param name="num_spatial_dims">Number of spatial dims in input. Generally rank - 2 (ignore N and C dims).</param>
void AddPadTypeAndPads(COREML_SPEC::MILSpec::Operation& op, ModelBuilder& model_builder, std::string_view op_type,
                       const NodeAttrHelper& helper, int num_spatial_dims);
}  // namespace coreml
}  // namespace onnxruntime
