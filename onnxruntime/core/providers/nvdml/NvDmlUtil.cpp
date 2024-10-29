#include "NvDmlUtils.h"
namespace NvDml
{

inline uint32_t ComputeElementCountFromDimensions(gsl::span<const uint32_t> dimensions)
{
    return std::accumulate(dimensions.begin(), dimensions.end(), 1, std::multiplies<uint32_t>());
}


void GetShape(onnxruntime::Tensor& tensor, uint32_t* dimensions)
{
      size_t dimensionCount = tensor.Shape().NumDimensions();

      for (size_t i = 0; i < dimensionCount; ++i)
      {
        dimensions[i] = static_cast<uint32_t>(tensor.Shape()[i]);
      }
}

size_t ComputeByteSizeFromTensor(onnxruntime::Tensor& tensor) {

    size_t dimensionCount = tensor.Shape().NumDimensions();

    std::array<DimensionType, MaximumDimensionCount> dimensions;
    std::fill(dimensions.data(), dimensions.data() + dimensionCount, 0u);

    GetShape(tensor,  /*out*/ dimensions.data());

    auto dataType = tensor.DataType();
    auto elementSize = dataType->Size();
    auto elementCount = ComputeElementCountFromDimensions(dimensions);


  return elementSize * elementCount;
}
} // namespace NvDml
