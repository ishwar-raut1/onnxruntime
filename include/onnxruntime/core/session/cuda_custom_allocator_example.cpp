#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>
#include <cassert>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

// Include ONNX Runtime headers
#include "onnxruntime_c_api.h"
#include "onnxruntime_cxx_api.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#endif

/**
 * Custom CUDA Allocator Example for ONNX Runtime Environment
 * 
 * This example demonstrates how to:
 * 1. Create a custom CUDA allocator
 * 2. Register it with the ONNX Runtime environment 
 * 3. Use it with CUDA execution provider
 * 4. Share the allocator between multiple sessions
 */

// ============================================================================
// Custom CUDA Allocator Implementation
// ============================================================================

#ifdef USE_CUDA
struct CustomCudaAllocator {
    int device_id;
    size_t total_allocated;
    size_t max_memory_limit;
    
    CustomCudaAllocator(int device_id, size_t max_memory = SIZE_MAX) 
        : device_id(device_id), total_allocated(0), max_memory_limit(max_memory) {
        // Set CUDA device
        cudaSetDevice(device_id);
    }
    
    ~CustomCudaAllocator() {
        std::cout << "Custom CUDA Allocator destroyed. Total allocated: " 
                  << total_allocated << " bytes\n";
    }
};

// C-style allocator functions for OrtAllocator
static void* CustomCudaAlloc(struct OrtAllocator* this_, size_t size) {
    auto* allocator = static_cast<CustomCudaAllocator*>(this_);
    
    // Check memory limit
    if (allocator->total_allocated + size > allocator->max_memory_limit) {
        std::cerr << "Memory limit exceeded!\n";
        return nullptr;
    }
    
    void* ptr = nullptr;
    cudaError_t result = cudaMalloc(&ptr, size);
    
    if (result == cudaSuccess) {
        allocator->total_allocated += size;
        std::cout << "Custom CUDA Alloc: " << size << " bytes on device " 
                  << allocator->device_id << " (total: " << allocator->total_allocated << ")\n";
        return ptr;
    } else {
        std::cerr << "CUDA allocation failed: " << cudaGetErrorString(result) << "\n";
        return nullptr;
    }
}

static void CustomCudaFree(struct OrtAllocator* this_, void* p) {
    if (p == nullptr) return;
    
    auto* allocator = static_cast<CustomCudaAllocator*>(this_);
    
    // Get allocation size (simplified - in real implementation you'd track this)
    size_t size = 0;
    // For demo purposes, we'll skip size tracking
    
    cudaError_t result = cudaFree(p);
    if (result == cudaSuccess) {
        allocator->total_allocated -= size; // In real implementation, track actual size
        std::cout << "Custom CUDA Free: pointer freed on device " << allocator->device_id << "\n";
    } else {
        std::cerr << "CUDA free failed: " << cudaGetErrorString(result) << "\n";
    }
}

static const struct OrtMemoryInfo* CustomCudaGetInfo(const struct OrtAllocator* this_) {
    // This would return the memory info associated with this allocator
    // For demo purposes, we'll return nullptr (in real implementation, store this)
    return nullptr;
}

// Optional Reserve function for session initialization allocations
static void* CustomCudaReserve(struct OrtAllocator* this_, size_t size) {
    std::cout << "Custom CUDA Reserve called for " << size << " bytes\n";
    return CustomCudaAlloc(this_, size);
}
#endif

// ============================================================================
// C API Example
// ============================================================================

void demonstrateCApiCustomCudaAllocator() {
    std::cout << "\n=== C API Custom CUDA Allocator Example ===\n";
    
#ifdef USE_CUDA
    const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    OrtStatus* status = nullptr;
    OrtEnv* env = nullptr;
    OrtMemoryInfo* cuda_memory_info = nullptr;
    OrtAllocator* custom_allocator = nullptr;
    OrtSessionOptions* session_options = nullptr;
    OrtSession* session = nullptr;
    
    try {
        // 1. Create environment
        status = g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "CudaAllocatorExample", &env);
        if (status != nullptr) throw std::runtime_error("Failed to create environment");
        
        // 2. Create custom CUDA allocator instance
        auto* cuda_alloc_impl = new CustomCudaAllocator(0, 1024 * 1024 * 1024); // 1GB limit
        
        // 3. Create OrtMemoryInfo for CUDA device
        status = g_ort->CreateMemoryInfo("Cuda", OrtArenaAllocator, 0, OrtMemTypeDefault, &cuda_memory_info);
        if (status != nullptr) throw std::runtime_error("Failed to create CUDA memory info");
        
        // 4. Create OrtAllocator structure
        custom_allocator = new OrtAllocator{
            ORT_API_VERSION,           // version
            CustomCudaAlloc,           // Alloc function
            CustomCudaFree,            // Free function  
            CustomCudaGetInfo,         // Info function
            CustomCudaReserve          // Reserve function (optional)
        };
        
        // Store our custom allocator implementation in the OrtAllocator
        // Note: In a real implementation, you'd need a way to associate the impl with the OrtAllocator
        // This is a simplified example
        
        // 5. Register the custom allocator with the environment
        status = g_ort->RegisterAllocator(env, custom_allocator);
        if (status != nullptr) throw std::runtime_error("Failed to register custom allocator");
        
        std::cout << "Successfully registered custom CUDA allocator with environment\n";
        
        // 6. Create session options with CUDA EP
        status = g_ort->CreateSessionOptions(&session_options);
        if (status != nullptr) throw std::runtime_error("Failed to create session options");
        
        // Configure CUDA EP to use device 0
        status = g_ort->SessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
        if (status != nullptr) throw std::runtime_error("Failed to add CUDA EP");
        
        std::cout << "Custom CUDA allocator is now available for CUDA EP sessions\n";
        
        // 7. Create session (would use the registered allocator)
        // Note: You'd need an actual model file for this
        // status = g_ort->CreateSession(env, L"model.onnx", session_options, &session);
        
        // Cleanup
        if (session) g_ort->ReleaseSession(session);
        if (session_options) g_ort->ReleaseSessionOptions(session_options);
        
        // Unregister allocator
        status = g_ort->UnregisterAllocator(env, cuda_memory_info);
        if (status != nullptr) {
            std::cerr << "Failed to unregister allocator\n";
        }
        
        if (cuda_memory_info) g_ort->ReleaseMemoryInfo(cuda_memory_info);
        delete custom_allocator;
        delete cuda_alloc_impl;
        if (env) g_ort->ReleaseEnv(env);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        // Cleanup on error
        if (status) {
            std::cerr << "ORT Error: " << g_ort->GetErrorMessage(status) << "\n";
            g_ort->ReleaseStatus(status);
        }
    }
#else
    std::cout << "CUDA support not available\n";
#endif
}

// ============================================================================
// C++ API Example 
// ============================================================================

void demonstrateCppApiCustomCudaAllocator() {
    std::cout << "\n=== C++ API Custom CUDA Allocator Example ===\n";
    
#ifdef USE_CUDA
    try {
        // 1. Create environment
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "CudaAllocatorCppExample");
        
        // 2. Create memory info for CUDA device 0
        Ort::MemoryInfo cuda_memory_info("Cuda", OrtArenaAllocator, 0, OrtMemTypeDefault);
        
        // 3. Configure arena for CUDA allocator
        Ort::ArenaCfg arena_cfg(
            1024 * 1024 * 1024,  // max_mem: 1GB
            0,                   // arena_extend_strategy: kNextPowerOfTwo
            16 * 1024 * 1024,    // initial_chunk_size_bytes: 16MB
            -1                   // max_dead_bytes_per_chunk: use default
        );
        
        // 4. Method 1: Create and register allocator using environment
        std::cout << "Creating and registering CUDA allocator with environment...\n";
        env.CreateAndRegisterAllocator(cuda_memory_info, arena_cfg);
        
        // 5. Method 2: Using CreateAndRegisterAllocatorV2 with provider-specific options
        std::unordered_map<std::string, std::string> provider_options = {
            {"device_id", "0"},
            {"gpu_mem_limit", "1073741824"},  // 1GB
            {"arena_extend_strategy", "0"}    // kNextPowerOfTwo
        };
        
        // Create another memory info for device 1 (if available)
        Ort::MemoryInfo cuda_memory_info_dev1("Cuda", OrtArenaAllocator, 1, OrtMemTypeDefault);
        
        try {
            env.CreateAndRegisterAllocatorV2("CUDAExecutionProvider", 
                                            cuda_memory_info_dev1, 
                                            provider_options, 
                                            arena_cfg);
            std::cout << "Successfully registered CUDA allocator for device 1\n";
        } catch (const std::exception& e) {
            std::cout << "Device 1 not available or already registered: " << e.what() << "\n";
        }
        
        // 6. Get list of registered allocators
        const auto& registered_allocators = env.GetRegisteredSharedAllocators();
        std::cout << "Number of registered shared allocators: " << registered_allocators.size() << "\n";
        
        // 7. Create session options with CUDA EP
        Ort::SessionOptions session_options;
        
        // Configure CUDA provider options
        OrtCUDAProviderOptions cuda_options{};
        cuda_options.device_id = 0;
        cuda_options.arena_extend_strategy = 0; // kNextPowerOfTwo
        cuda_options.gpu_mem_limit = SIZE_MAX;  // Use our registered allocator limit
        cuda_options.do_copy_in_default_stream = 1;
        cuda_options.has_user_compute_stream = 0;
        cuda_options.user_compute_stream = nullptr;
        cuda_options.default_memory_arena_cfg = nullptr; // Use registered allocator
        
        session_options.AppendExecutionProvider_CUDA(cuda_options);
        
        // 8. Enable memory pattern optimization and CPU arena
        session_options.EnableMemPattern();
        session_options.EnableCpuMemArena();
        
        std::cout << "Session options configured to use custom CUDA allocator\n";
        
        // 9. Create session (would use the registered allocator automatically)
        // Ort::Session session(env, L"model.onnx", session_options);
        
        // 10. Demonstrate usage with tensor creation
        std::vector<int64_t> input_shape = {1, 3, 224, 224};
        size_t input_tensor_size = 1 * 3 * 224 * 224;
        
        // Get the CUDA allocator for tensor creation
        // Note: In practice, you'd get this from a session
        // Ort::Allocator cuda_allocator(session, cuda_memory_info);
        // auto memory_allocation = cuda_allocator.GetAllocation(input_tensor_size * sizeof(float));
        // Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        //     cuda_memory_info, 
        //     static_cast<float*>(memory_allocation.get()), 
        //     input_tensor_size, 
        //     input_shape.data(), 
        //     input_shape.size()
        // );
        
        std::cout << "Custom CUDA allocator successfully integrated with ONNX Runtime\n";
        
        // Cleanup - unregister allocators
        try {
            env.UnregisterAllocator(cuda_memory_info);
            std::cout << "Unregistered CUDA allocator for device 0\n";
        } catch (const std::exception& e) {
            std::cerr << "Failed to unregister allocator: " << e.what() << "\n";
        }
        
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }
#else
    std::cout << "CUDA support not available\n";
#endif
}

// ============================================================================
// Advanced Example: Custom Allocator with Memory Tracking
// ============================================================================

#ifdef USE_CUDA
class AdvancedCudaAllocator {
private:
    struct AllocationRecord {
        size_t size;
        std::chrono::steady_clock::time_point allocated_time;
    };
    
    std::unordered_map<void*, AllocationRecord> allocations_;
    std::mutex mutex_;
    int device_id_;
    size_t total_allocated_;
    size_t peak_allocated_;
    size_t allocation_count_;
    
public:
    AdvancedCudaAllocator(int device_id) 
        : device_id_(device_id), total_allocated_(0), peak_allocated_(0), allocation_count_(0) {
        cudaSetDevice(device_id_);
    }
    
    void* allocate(size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        void* ptr = nullptr;
        cudaError_t result = cudaMalloc(&ptr, size);
        
        if (result == cudaSuccess && ptr != nullptr) {
            allocations_[ptr] = {size, std::chrono::steady_clock::now()};
            total_allocated_ += size;
            peak_allocated_ = std::max(peak_allocated_, total_allocated_);
            allocation_count_++;
            
            std::cout << "CUDA Alloc [" << allocation_count_ << "]: " << size 
                      << " bytes, total: " << total_allocated_ << " bytes\n";
        }
        
        return ptr;
    }
    
    void deallocate(void* ptr) {
        if (ptr == nullptr) return;
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto it = allocations_.find(ptr);
        if (it != allocations_.end()) {
            total_allocated_ -= it->second.size;
            allocations_.erase(it);
            
            std::cout << "CUDA Free: total now " << total_allocated_ << " bytes\n";
        }
        
        cudaFree(ptr);
    }
    
    void printStats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::cout << "\n=== CUDA Allocator Statistics ===\n";
        std::cout << "Device ID: " << device_id_ << "\n";
        std::cout << "Current allocations: " << allocations_.size() << "\n";
        std::cout << "Total allocated: " << total_allocated_ << " bytes\n";
        std::cout << "Peak allocated: " << peak_allocated_ << " bytes\n";
        std::cout << "Total allocation count: " << allocation_count_ << "\n";
        std::cout << "=================================\n\n";
    }
};
#endif

void demonstrateAdvancedCustomAllocator() {
    std::cout << "\n=== Advanced Custom CUDA Allocator Example ===\n";
    
#ifdef USE_CUDA
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "AdvancedCudaExample");
        
        // Create custom allocator with tracking
        auto advanced_allocator = std::make_unique<AdvancedCudaAllocator>(0);
        
        // Create memory info
        Ort::MemoryInfo cuda_memory_info("Cuda", OrtArenaAllocator, 0, OrtMemTypeDefault);
        
        // Configure arena with specific settings for optimal performance
        Ort::ArenaCfg arena_cfg(
            2ULL * 1024 * 1024 * 1024,  // max_mem: 2GB
            1,                           // arena_extend_strategy: kSameAsRequested
            64 * 1024 * 1024,            // initial_chunk_size_bytes: 64MB
            32 * 1024 * 1024             // max_dead_bytes_per_chunk: 32MB
        );
        
        // Register allocator with environment
        env.CreateAndRegisterAllocator(cuda_memory_info, arena_cfg);
        
        // Create session options with optimized CUDA settings
        Ort::SessionOptions session_options;
        
        // Configure CUDA EP with custom settings
        std::unordered_map<std::string, std::string> cuda_provider_options = {
            {"device_id", "0"},
            {"gpu_mem_limit", "2147483648"},  // 2GB
            {"arena_extend_strategy", "1"},   // kSameAsRequested
            {"do_copy_in_default_stream", "1"},
            {"cudnn_conv_algo_search", "1"}   // Heuristic search
        };
        
        session_options.AppendExecutionProvider("CUDAExecutionProvider", cuda_provider_options);
        
        // Enable memory optimizations
        session_options.EnableMemPattern();
        session_options.EnableCpuMemArena();
        
        // Add session configuration for memory management
        session_options.AddConfigEntry("session.enable_cpu_mem_arena", "1");
        session_options.AddConfigEntry("session.enable_mem_pattern", "1");
        session_options.AddConfigEntry("session.memory_arena_shrinkage_ratio", "0.25");
        
        std::cout << "Advanced CUDA allocator configured and registered\n";
        
        // Print allocator statistics
        advanced_allocator->printStats();
        
        // Simulate some allocations for demonstration
        std::vector<void*> test_allocations;
        for (int i = 0; i < 5; ++i) {
            size_t size = (i + 1) * 1024 * 1024;  // 1MB, 2MB, 3MB, 4MB, 5MB
            void* ptr = advanced_allocator->allocate(size);
            if (ptr) {
                test_allocations.push_back(ptr);
            }
        }
        
        advanced_allocator->printStats();
        
        // Cleanup test allocations
        for (void* ptr : test_allocations) {
            advanced_allocator->deallocate(ptr);
        }
        
        advanced_allocator->printStats();
        
        // Unregister the allocator
        env.UnregisterAllocator(cuda_memory_info);
        
    } catch (const std::exception& e) {
        std::cerr << "Error in advanced example: " << e.what() << "\n";
    }
#else
    std::cout << "CUDA support not available\n";
#endif
}

// ============================================================================
// Main Function
// ============================================================================

int main() {
    std::cout << "ONNX Runtime Custom CUDA Allocator Examples\n";
    std::cout << "============================================\n";
    
#ifdef USE_CUDA
    // Initialize CUDA
    cudaError_t cuda_status = cudaSetDevice(0);
    if (cuda_status != cudaSuccess) {
        std::cerr << "Failed to initialize CUDA device 0: " 
                  << cudaGetErrorString(cuda_status) << "\n";
        return 1;
    }
    
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    std::cout << "Available CUDA devices: " << device_count << "\n";
#endif
    
    try {
        // Run examples
        demonstrateCApiCustomCudaAllocator();
        demonstrateCppApiCustomCudaAllocator();
        demonstrateAdvancedCustomAllocator();
        
        std::cout << "\nAll examples completed successfully!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}

/* 
=== Usage Instructions ===

1. Compilation:
   - Make sure you have CUDA toolkit installed
   - Compile with: g++ -DUSE_CUDA -I/path/to/onnxruntime/include 
                   -L/path/to/onnxruntime/lib -L/path/to/cuda/lib64
                   -lonnxruntime -lcudart cuda_custom_allocator_example.cpp -o cuda_allocator_example

2. Key Concepts Demonstrated:
   - Creating custom CUDA allocators with memory tracking
   - Registering allocators with ONNX Runtime environment
   - Sharing allocators between multiple sessions
   - Using both C API and C++ API approaches
   - Configuring CUDA execution provider with custom memory settings
   - Memory arena configuration for optimal performance

3. Real-world Usage:
   - Replace demo allocation logic with your actual memory management
   - Add proper error handling and cleanup
   - Implement memory pooling or custom allocation strategies
   - Monitor memory usage and performance metrics
   - Handle multi-device scenarios

4. Advanced Features:
   - Custom memory pools
   - Memory defragmentation
   - Cross-device memory transfers
   - Integration with existing memory managers
   - Performance profiling and optimization
*/
