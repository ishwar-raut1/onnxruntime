#include "onnxruntime_c_api.h"
#include <d3d12.h>
#include <DirectML.h>
#include "dml_provider_factory.h"
/**
 * Creates an NVDML Execution Provider using the specified DirectML device and D3D12 command queue.
 * Both the DirectML device and the D3D12 command queue must share the same parent ID3D12Device; otherwise, an error will be returned.
 * The D3D12 command queue must be of type DIRECT or COMPUTE (see D3D12_COMMAND_LIST_TYPE).
 * If this function succeeds, the inference session maintains strong references to both the dml_device and the command_queue objects.
 *
 */
ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_NVDML, _In_ OrtSessionOptions* options,
               _In_ IDMLDevice* dml_device, _In_ ID3D12CommandQueue* cmd_queue);



struct OrtNvDmlApi;
typedef struct OrtNvDmlApi OrtNvDmlApi;
struct OrtNvDmlApi : OrtDmlApi {
  /**
   * Creates an NVDML Execution Provider using the specified DirectML device and D3D12 command queue.
   * Both the DirectML device and the D3D12 command queue must share the same parent ID3D12Device; otherwise, an error will be returned.
   * The D3D12 command queue must be of type DIRECT or COMPUTE (see D3D12_COMMAND_LIST_TYPE).
   * If this function succeeds, the inference session maintains strong references to both the dml_device and the command_queue objects.
   *
   */
  ORT_API2_STATUS(SessionOptionsAppendExecutionProvider_NVDML, _In_ OrtSessionOptions* options,
                   _In_ IDMLDevice* dml_device, _In_ ID3D12CommandQueue* cmd_queue);


};
