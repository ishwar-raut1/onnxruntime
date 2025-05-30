# Copyright (c) 2019, 2022, Oracle and/or its affiliates. All rights reserved.
# Licensed under the MIT License.

#set(CMAKE_VERBOSE_MAKEFILE on)

# Setup Java compilation
include(FindJava)
find_package(Java REQUIRED)
include(UseJava)
if (NOT ANDROID)
    find_package(JNI REQUIRED)
endif()

set(JAVA_ROOT ${REPO_ROOT}/java)
set(JAVA_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/java)
if (onnxruntime_RUN_ONNX_TESTS)
  set(JAVA_DEPENDS onnxruntime ${test_data_target})
else()
  set(JAVA_DEPENDS onnxruntime)
endif()

set(GRADLE_EXECUTABLE "${JAVA_ROOT}/gradlew")

set(COMMON_GRADLE_ARGS --console=plain)
if(WIN32)
  list(APPEND COMMON_GRADLE_ARGS -Dorg.gradle.daemon=false)
elseif (ANDROID)
  # For Android build, we may run gradle multiple times in same build,
  # sometimes gradle JVM will run out of memory if we keep the daemon running
  # it is better to not keep a daemon running
  list(APPEND COMMON_GRADLE_ARGS --no-daemon)
endif()

# Specify the Java source files
file(GLOB_RECURSE onnxruntime4j_gradle_files "${JAVA_ROOT}/*.gradle")
file(GLOB_RECURSE onnxruntime4j_src "${JAVA_ROOT}/src/main/java/ai/onnxruntime/*.java")
set(JAVA_OUTPUT_JAR ${JAVA_ROOT}/build/libs/onnxruntime.jar)
# this jar is solely used to signaling mechanism for dependency management in CMake
# if any of the Java sources change, the jar (and generated headers) will be regenerated and the onnxruntime4j_jni target will be rebuilt
set(GRADLE_ARGS clean jar -x test)

add_custom_command(OUTPUT ${JAVA_OUTPUT_JAR}
                   COMMAND ${GRADLE_EXECUTABLE} ${COMMON_GRADLE_ARGS} ${GRADLE_ARGS}
                   WORKING_DIRECTORY ${JAVA_ROOT}
                   DEPENDS ${onnxruntime4j_gradle_files} ${onnxruntime4j_src})
add_custom_target(onnxruntime4j DEPENDS ${JAVA_OUTPUT_JAR})
set_source_files_properties(${JAVA_OUTPUT_JAR} PROPERTIES GENERATED TRUE)
set_property(TARGET onnxruntime4j APPEND PROPERTY ADDITIONAL_CLEAN_FILES "${JAVA_OUTPUT_DIR}")

# Specify the native sources
file(GLOB onnxruntime4j_native_src
    "${JAVA_ROOT}/src/main/native/*.c"
    "${JAVA_ROOT}/src/main/native/*.h"
    "${REPO_ROOT}/include/onnxruntime/core/session/*.h"
    "${REPO_ROOT}/orttraining/orttraining/training_api/include/*.h"
    )
# Build the JNI library
onnxruntime_add_shared_library_module(onnxruntime4j_jni ${onnxruntime4j_native_src})
set_property(TARGET onnxruntime4j_jni PROPERTY C_STANDARD 11)

if (APPLE)
  set_target_properties(onnxruntime4j_jni PROPERTIES
    MACOSX_RPATH TRUE
    SKIP_BUILD_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH FALSE
    BUILD_WITH_INSTALL_NAME_DIR TRUE
    INSTALL_NAME_DIR @rpath)
endif()

# depend on java sources. if they change, the JNI should recompile
add_dependencies(onnxruntime4j_jni onnxruntime4j)
onnxruntime_add_include_to_target(onnxruntime4j_jni onnxruntime_session)
# the JNI headers are generated in the onnxruntime4j target
target_include_directories(onnxruntime4j_jni PRIVATE ${REPO_ROOT}/include ${REPO_ROOT}/orttraining/orttraining/training_api/include ${JAVA_ROOT}/build/headers ${JNI_INCLUDE_DIRS})
target_link_libraries(onnxruntime4j_jni PUBLIC onnxruntime)

set(JAVA_PACKAGE_OUTPUT_DIR ${JAVA_OUTPUT_DIR}/build)
file(MAKE_DIRECTORY ${JAVA_PACKAGE_OUTPUT_DIR})
if (ANDROID)
  set(ANDROID_PACKAGE_OUTPUT_DIR ${JAVA_PACKAGE_OUTPUT_DIR}/android)
  file(MAKE_DIRECTORY ${ANDROID_PACKAGE_OUTPUT_DIR})
endif()

# Set platform and arch for packaging
# Checks the names set by MLAS on non-Windows platforms first
if(APPLE)
   get_target_property(ONNXRUNTIME4J_OSX_ARCH onnxruntime4j_jni OSX_ARCHITECTURES)
   list(LENGTH ONNXRUNTIME4J_OSX_ARCH ONNXRUNTIME4J_OSX_ARCH_LEN)
   if(ONNXRUNTIME4J_OSX_ARCH)
       if(ONNXRUNTIME4J_OSX_ARCH_LEN LESS_EQUAL 1)
               list(GET ONNXRUNTIME4J_OSX_ARCH 0 JNI_ARCH)
               message("Set Java ARCH TO macOS/iOS ${JNI_ARCH}")
       else()
               message(FATAL_ERROR "Java is currently not supported for macOS universal")
       endif()
   else()
       set(JNI_ARCH ${CMAKE_HOST_SYSTEM_PROCESSOR})
       message("Set Java ARCH TO macOS/iOS ${JNI_ARCH}")
   endif()
   if(JNI_ARCH STREQUAL "x86_64")
       set(JNI_ARCH x64)
   elseif(JNI_ARCH STREQUAL "arm64")
       set(JNI_ARCH aarch64)
   endif()
elseif (ANDROID)
  set(JNI_ARCH ${ANDROID_ABI})
elseif (ARM64)
  set(JNI_ARCH aarch64)
elseif (X86_64)
  set(JNI_ARCH x64)
elseif (POWER)
  set(JNI_ARCH ppc64)
else()
  # Now mirror the checks used with MSVC
  if(MSVC)
    if(onnxruntime_target_platform STREQUAL "ARM64")
      set(JNI_ARCH aarch64)
    elseif(onnxruntime_target_platform STREQUAL "x64")
      set(JNI_ARCH x64)
    else()
      # if everything else failed then we're on a 32-bit arch and Java isn't supported
      message(FATAL_ERROR "Java is currently not supported on 32-bit x86 architecture")
    endif()
  else()
    # if everything else failed then we're on a 32-bit arch and Java isn't supported
    message(FATAL_ERROR "Java is currently not supported on 32-bit x86 architecture")
  endif()
endif()

if (WIN32)
  set(JAVA_PLAT "win")
elseif (APPLE)
  set(JAVA_PLAT "osx")
elseif (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  set(JAVA_PLAT "linux")
else()
  # We don't do distribution for Android
  # Set for completeness
  set(JAVA_PLAT "android")
endif()

# Similar to Nuget schema
set(JAVA_OS_ARCH ${JAVA_PLAT}-${JNI_ARCH})

# expose native libraries to the gradle build process
set(JAVA_PACKAGE_DIR ai/onnxruntime/native/${JAVA_OS_ARCH})
set(JAVA_NATIVE_LIB_DIR ${JAVA_OUTPUT_DIR}/native-lib)
set(JAVA_NATIVE_JNI_DIR ${JAVA_OUTPUT_DIR}/native-jni)
set(JAVA_PACKAGE_LIB_DIR ${JAVA_NATIVE_LIB_DIR}/${JAVA_PACKAGE_DIR})
set(JAVA_PACKAGE_JNI_DIR ${JAVA_NATIVE_JNI_DIR}/${JAVA_PACKAGE_DIR})
file(MAKE_DIRECTORY ${JAVA_PACKAGE_LIB_DIR})
file(MAKE_DIRECTORY ${JAVA_PACKAGE_JNI_DIR})

# On Windows TARGET_LINKER_FILE_NAME is the .lib, TARGET_FILE_NAME is the .dll
if (WIN32)
  #Our static analysis plugin set /p:LinkCompiled=false
  if(NOT onnxruntime_ENABLE_STATIC_ANALYSIS)
    add_custom_command(TARGET onnxruntime4j_jni POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:onnxruntime> ${JAVA_PACKAGE_LIB_DIR}/$<TARGET_FILE_NAME:onnxruntime>)
    add_custom_command(TARGET onnxruntime4j_jni POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:onnxruntime4j_jni> ${JAVA_PACKAGE_JNI_DIR}/$<TARGET_FILE_NAME:onnxruntime4j_jni>)
    if (onnxruntime_USE_CUDA OR onnxruntime_USE_DNNL OR onnxruntime_USE_OPENVINO OR onnxruntime_USE_TENSORRT OR (onnxruntime_USE_QNN AND NOT onnxruntime_BUILD_QNN_EP_STATIC_LIB))
      add_custom_command(TARGET onnxruntime4j_jni POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:onnxruntime_providers_shared> ${JAVA_PACKAGE_LIB_DIR}/$<TARGET_FILE_NAME:onnxruntime_providers_shared>)
    endif()
    if (onnxruntime_USE_CUDA)
      add_custom_command(TARGET onnxruntime4j_jni POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:onnxruntime_providers_cuda> ${JAVA_PACKAGE_LIB_DIR}/$<TARGET_FILE_NAME:onnxruntime_providers_cuda>)
    endif()
    if (onnxruntime_USE_DNNL)
      add_custom_command(TARGET onnxruntime4j_jni POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:onnxruntime_providers_dnnl> ${JAVA_PACKAGE_LIB_DIR}/$<TARGET_FILE_NAME:onnxruntime_providers_dnnl>)
    endif()
    if (onnxruntime_USE_OPENVINO)
      add_custom_command(TARGET onnxruntime4j_jni POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:onnxruntime_providers_openvino> ${JAVA_PACKAGE_LIB_DIR}/$<TARGET_FILE_NAME:onnxruntime_providers_openvino>)
    endif()
    if (onnxruntime_USE_TENSORRT)
      add_custom_command(TARGET onnxruntime4j_jni POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:onnxruntime_providers_tensorrt> ${JAVA_PACKAGE_LIB_DIR}/$<TARGET_FILE_NAME:onnxruntime_providers_tensorrt>)
    endif()
    if (onnxruntime_USE_QNN AND NOT onnxruntime_BUILD_QNN_EP_STATIC_LIB)
      add_custom_command(TARGET onnxruntime4j_jni POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:onnxruntime_providers_qnn> ${JAVA_PACKAGE_LIB_DIR}/$<TARGET_FILE_NAME:onnxruntime_providers_qnn>)
    endif()
    if (onnxruntime_USE_WEBGPU)
      if (onnxruntime_ENABLE_DAWN_BACKEND_D3D12)
        # TODO: the following code is used to disable building Dawn using vcpkg temporarily
        # until we figure out how to resolve the packaging pipeline failures
        #
        # if (onnxruntime_USE_VCPKG)
        if (FALSE)
          add_custom_command(
            TARGET onnxruntime4j_jni POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:Microsoft::DXIL>
                $<TARGET_FILE:Microsoft::DirectXShaderCompiler>
                ${JAVA_PACKAGE_LIB_DIR}/
          )
        else()
          add_custom_command(
            TARGET onnxruntime4j_jni POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE_DIR:dxcompiler>/dxil.dll
                $<TARGET_FILE_DIR:dxcompiler>/dxcompiler.dll
                ${JAVA_PACKAGE_LIB_DIR}/
          )
        endif()
      endif()
      if (onnxruntime_BUILD_DAWN_MONOLITHIC_LIBRARY)
        add_custom_command(TARGET onnxruntime4j_jni POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:dawn::webgpu_dawn> ${JAVA_PACKAGE_LIB_DIR}/$<TARGET_FILE_NAME:dawn::webgpu_dawn>)
      endif()
    endif()
  endif()
else()
  add_custom_command(TARGET onnxruntime4j_jni POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:onnxruntime> ${JAVA_PACKAGE_LIB_DIR}/$<TARGET_LINKER_FILE_NAME:onnxruntime>)
  add_custom_command(TARGET onnxruntime4j_jni POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:onnxruntime4j_jni> ${JAVA_PACKAGE_JNI_DIR}/$<TARGET_LINKER_FILE_NAME:onnxruntime4j_jni>)
  if (onnxruntime_USE_CUDA OR onnxruntime_USE_DNNL OR onnxruntime_USE_OPENVINO OR onnxruntime_USE_TENSORRT OR (onnxruntime_USE_QNN AND NOT onnxruntime_BUILD_QNN_EP_STATIC_LIB))
    add_custom_command(TARGET onnxruntime4j_jni POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:onnxruntime_providers_shared> ${JAVA_PACKAGE_LIB_DIR}/$<TARGET_LINKER_FILE_NAME:onnxruntime_providers_shared>)
  endif()
  if (onnxruntime_USE_CUDA)
    add_custom_command(TARGET onnxruntime4j_jni POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:onnxruntime_providers_cuda> ${JAVA_PACKAGE_LIB_DIR}/$<TARGET_LINKER_FILE_NAME:onnxruntime_providers_cuda>)
  endif()
  if (onnxruntime_USE_DNNL)
    add_custom_command(TARGET onnxruntime4j_jni POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:onnxruntime_providers_dnnl> ${JAVA_PACKAGE_LIB_DIR}/$<TARGET_LINKER_FILE_NAME:onnxruntime_providers_dnnl>)
  endif()
  if (onnxruntime_USE_OPENVINO)
    add_custom_command(TARGET onnxruntime4j_jni POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:onnxruntime_providers_openvino> ${JAVA_PACKAGE_LIB_DIR}/$<TARGET_LINKER_FILE_NAME:onnxruntime_providers_openvino>)
  endif()
  if (onnxruntime_USE_TENSORRT)
    add_custom_command(TARGET onnxruntime4j_jni POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:onnxruntime_providers_tensorrt> ${JAVA_PACKAGE_LIB_DIR}/$<TARGET_LINKER_FILE_NAME:onnxruntime_providers_tensorrt>)
  endif()
  if (onnxruntime_USE_QNN AND NOT onnxruntime_BUILD_QNN_EP_STATIC_LIB)
    add_custom_command(TARGET onnxruntime4j_jni POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:onnxruntime_providers_qnn> ${JAVA_PACKAGE_LIB_DIR}/$<TARGET_LINKER_FILE_NAME:onnxruntime_providers_qnn>)
  endif()
  if (onnxruntime_USE_WEBGPU AND onnxruntime_BUILD_DAWN_MONOLITHIC_LIBRARY)
    add_custom_command(TARGET onnxruntime4j_jni POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:dawn::webgpu_dawn> ${JAVA_PACKAGE_LIB_DIR}/$<TARGET_LINKER_FILE_NAME:dawn::webgpu_dawn>)
  endif()
endif()

# run the build process (this copies the results back into CMAKE_CURRENT_BINARY_DIR)
set(GRADLE_ARGS cmakeBuild -DcmakeBuildDir=${CMAKE_CURRENT_BINARY_DIR})

# Append relevant native build flags to gradle command
set(GRADLE_ARGS ${GRADLE_ARGS} ${ORT_PROVIDER_FLAGS})
if (onnxruntime_ENABLE_TRAINING_APIS)
    set(GRADLE_ARGS ${GRADLE_ARGS} "-DENABLE_TRAINING_APIS=1")
endif()

message(STATUS "GRADLE_ARGS: ${GRADLE_ARGS}")
add_custom_command(TARGET onnxruntime4j_jni POST_BUILD
                   COMMAND ${GRADLE_EXECUTABLE} ${COMMON_GRADLE_ARGS} ${GRADLE_ARGS}
                   WORKING_DIRECTORY ${JAVA_ROOT})

if (ANDROID)
  set(ANDROID_PACKAGE_JNILIBS_DIR ${JAVA_OUTPUT_DIR}/android)
  set(ANDROID_PACKAGE_ABI_DIR ${ANDROID_PACKAGE_JNILIBS_DIR}/${ANDROID_ABI})
  file(MAKE_DIRECTORY ${ANDROID_PACKAGE_JNILIBS_DIR})
  file(MAKE_DIRECTORY ${ANDROID_PACKAGE_ABI_DIR})

  # Copy onnxruntime.so and onnxruntime4j_jni.so for building Android AAR package
  add_custom_command(TARGET onnxruntime4j_jni POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:onnxruntime> ${ANDROID_PACKAGE_ABI_DIR}/$<TARGET_LINKER_FILE_NAME:onnxruntime>)
  add_custom_command(TARGET onnxruntime4j_jni POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:onnxruntime4j_jni> ${ANDROID_PACKAGE_ABI_DIR}/$<TARGET_LINKER_FILE_NAME:onnxruntime4j_jni>)

  # Generate the Android AAR package
  add_custom_command(TARGET onnxruntime4j_jni
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Generating Android AAR package..."
    COMMAND ${GRADLE_EXECUTABLE}
      ${COMMON_GRADLE_ARGS}
      build
      -b build-android.gradle -c settings-android.gradle
      -DjniLibsDir=${ANDROID_PACKAGE_JNILIBS_DIR} -DbuildDir=${ANDROID_PACKAGE_OUTPUT_DIR}
      -DminSdkVer=${ANDROID_MIN_SDK} -DheadersDir=${ANDROID_HEADERS_DIR}
      $<$<BOOL:${onnxruntime_ENABLE_TRAINING_APIS}>:-DENABLE_TRAINING_APIS=1>
      --stacktrace
    WORKING_DIRECTORY ${JAVA_ROOT})

  if (onnxruntime_BUILD_UNIT_TESTS)
    set(ANDROID_TEST_PACKAGE_ROOT ${JAVA_ROOT}/src/test/android)
    set(ANDROID_TEST_PACKAGE_DIR ${JAVA_OUTPUT_DIR}/androidtest/android)
    #copy the androidtest project into cmake binary directory
    file(MAKE_DIRECTORY ${JAVA_OUTPUT_DIR}/androidtest)
    file(COPY ${ANDROID_TEST_PACKAGE_ROOT} DESTINATION ${JAVA_OUTPUT_DIR}/androidtest)
    set(ANDROID_TEST_PACKAGE_LIB_DIR ${ANDROID_TEST_PACKAGE_DIR}/app/libs)
    file(MAKE_DIRECTORY ${ANDROID_TEST_PACKAGE_LIB_DIR})
    # Copy the built Android AAR package to libs folder of our test app
    add_custom_command(TARGET onnxruntime4j_jni POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ANDROID_PACKAGE_OUTPUT_DIR}/outputs/aar/onnxruntime-debug.aar ${ANDROID_TEST_PACKAGE_LIB_DIR}/onnxruntime-android.aar)
    # Build Android test apk for java package
    add_custom_command(TARGET onnxruntime4j_jni
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E echo "Building and running Android test for Android AAR package..."
      COMMAND ${GRADLE_EXECUTABLE}
        ${COMMON_GRADLE_ARGS}
        clean assembleDebug assembleDebugAndroidTest
        -DminSdkVer=${ANDROID_MIN_SDK}
        --stacktrace
      WORKING_DIRECTORY ${ANDROID_TEST_PACKAGE_DIR})
  endif()
endif()
