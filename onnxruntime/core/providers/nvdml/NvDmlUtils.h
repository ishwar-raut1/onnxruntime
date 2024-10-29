
#include "core/framework/tensor.h"
#include <DirectML.h>
namespace NvDml
{
static const int MaximumDimensionCount = DML_TENSOR_DIMENSION_COUNT_MAX1;
using DimensionType = uint32_t;

inline uint32_t ComputeElementCountFromDimensions(gsl::span<const uint32_t> dimensions);


void GetShape(onnxruntime::Tensor& tensor, uint32_t* dimensions);
size_t ComputeByteSizeFromTensor(onnxruntime::Tensor& tensor) ;
} // namespace NvDml
