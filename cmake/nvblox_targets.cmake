# -------------------------------------------------------------------
# Function for adding a device compiler optinon.
# -------------------------------------------------------------------
function(add_device_compiler_option target_name option)
  target_compile_options(${target_name} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${option}>)
endfunction()

# -------------------------------------------------------------------
# Function for adding a host compiler options.
# Will apply to both gcc and nvcc (when building host code)
# -------------------------------------------------------------------

function(add_host_compiler_option target_name option)
   target_compile_options(${target_name} PRIVATE
     $<$<COMPILE_LANGUAGE:CXX>:${option}>)

   add_device_compiler_option(${target_name} -Xcompiler=${option})
endfunction()

# -------------------------------------------------------------------
# - Function for setting relevant compiler options for a given target
# -------------------------------------------------------------------
function(set_nvblox_compiler_options target_name)
  set_nvblox_compiler_options_internal(${target_name} True)
endfunction()

# -------------------------------------------------------------------
# - Function for setting relevant compiler options for a given target
# - Compiler warnings are disabled - suitable for external code
# -------------------------------------------------------------------
function(set_nvblox_compiler_options_nowarnings target_name)
  set_nvblox_compiler_options_internal(${target_name} False)
endfunction()

# -------------------------------------------------------------------
# - Internal function for setting  compiler options
# -------------------------------------------------------------------
function(set_nvblox_compiler_options_internal target_name enable_warnings)
  #########################
  # GENERAL COMPILER FLAGS
  #########################
  # Specify the C++ standard and general options
  # target_compile_features(${target_name} PRIVATE cxx_std_17)
  # Enable position independent code
  set_property(TARGET ${target_name} PROPERTY POSITION_INDEPENDENT_CODE ON)
  # Better output for profiling
  add_host_compiler_option(${target_name} "-fno-omit-frame-pointer")
  # Flag to use relative RPATHs. This allows the libraries to find each other also when they are
  # distributed.
  set_property(TARGET ${target_name} PROPERTY CMAKE_BUILD_RPATH_USE_ORIGIN on)

  # Use relative RPATHs. This allows the libraries to find each other also when they are
  # distributed.
  set_target_properties(${target_name} PROPERTIES BUILD_RPATH_USE_ORIGIN on)

  #########################
  # PREPROCESSOR DIRECTIVES
  #########################
  # Directive for pre-cxx11 linkage support
  target_compile_definitions(${target_name}
    PRIVATE
    "$<$<BOOL:${PRE_CXX11_ABI_LINKABLE}>:_GLIBCXX_USE_CXX11_ABI=0>")
  # Nvblox directive for pre-cxx11 linkage support
  target_compile_definitions(${target_name}
    PRIVATE
    "$<$<BOOL:${PRE_CXX11_ABI_LINKABLE}>:PRE_CXX11_ABI_LINKABLE>")
  # Change namespace cub:: into nvblox::cub. This is to avoid conflicts when other modules calls non
# thread safe functions in the cub namespace. Appending nvblox:: ensures an unique symbol that is
# only accesed by this library.
  target_compile_definitions(${target_name} PRIVATE CUB_WRAPPED_NAMESPACE=nvblox)
  # Needed to ensure that pytorch use glog
  target_compile_definitions(${target_name} PRIVATE C10_USE_GLOG=1)
  # The directive __CUDA_ARCH_FLAGS__ is always set when building with nvcc. We need it also when
  # building with gcc to avoid linker errors due to thrust placing certain symbols under a
  # __CUDA_ARCH_LIST__ namepace.
  target_compile_definitions(${target_name} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:__CUDA_ARCH_LIST__=${CUDA_ARCH_LIST}>)

  #######################
  # gcc SANITIZER FLAGS
  #######################
  if (USE_SANITIZER)
    add_host_compiler_option(${target_name} "-fsanitize=address")
    target_link_options(${target_name} PRIVATE "-fsanitize=address")
  endif()

  ####################
  # EXTENDED WARNINGS
  ####################
  if (enable_warnings)
    add_host_compiler_option(${target_name} "-Wall")
    add_host_compiler_option(${target_name} "-Wextra")
    add_host_compiler_option(${target_name} "-Wshadow")

    if (WARNING_AS_ERROR)
      add_host_compiler_option(${target_name} "-Werror")
    endif()
   endif()

  #####################################
  # FLAGS DEPENDING ON CMAKE_BUILD_TYPE
  #####################################
  string(TOLOWER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE_LOWER)
  if (CMAKE_BUILD_TYPE_LOWER STREQUAL "debug")
    add_host_compiler_option(${target_name} "-g")
    add_host_compiler_option(${target_name} "-O0")

    add_device_compiler_option(${target_name} "--debug")
    add_device_compiler_option(${target_name} "--device-debug")
    add_device_compiler_option(${target_name} "--generate-line-info")
    add_device_compiler_option(${target_name} "-O0")

  elseif(CMAKE_BUILD_TYPE_LOWER STREQUAL "relwithdebinfo")
    add_host_compiler_option(${target_name} "-g")
    add_host_compiler_option(${target_name} "-O2")

    add_device_compiler_option(${target_name} "--debug")
    add_device_compiler_option(${target_name} "--generate-line-info")
    add_device_compiler_option(${target_name} "-O2")

  elseif(CMAKE_BUILD_TYPE_LOWER STREQUAL "release")
    add_host_compiler_option(${target_name} "-O3")
    add_host_compiler_option(${target_name} "-DNDEBUG")

    add_device_compiler_option(${target_name} "-DNDEBUG")
    add_device_compiler_option(${target_name} "-O3")
  else()
    message(FATAL_ERROR "Invalid build type: ${CMAKE_BUILD_TYPE}")
  endif()


  ######################
  # CUDA SPECIFIC FLAGS
  ######################
  # Allow __host__, __device__ annotations in lambda declarations
  add_device_compiler_option(${target_name} "--extended-lambda")
  # Allows sharing constexpr between host and device code
  add_device_compiler_option(${target_name} "--expt-relaxed-constexpr")
  # Display warning numbers
  add_device_compiler_option(${target_name} "-Xcudafe=--display_error_number")
  # Suppress "dynamic initialization is not supported for a function-scope static __shared__
  # variable within a __device__/__global__ function". We cannot call the constructor in these
  # cases due to race condition. To my understanding, the variables are left un-constructed which
  # is still OK for our use case.
  add_device_compiler_option(${target_name} "--diag-suppress=20054")
  # Suppress "a __constant__ variable cannot be directly read in a host function". We share
  # __constant__ between host and device in the marching cubes implementation.
  add_device_compiler_option(${target_name} "--diag-suppress=20091")
  # Suppress "using-declaration ignored -- it refers to the current namespace" from glog header
  add_device_compiler_option(${target_name} "-diag-suppress=737")

  set_target_properties(${target_name} PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
  set_property(TARGET ${target_name} APPEND PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
  set_property(TARGET ${target_name} APPEND PROPERTY EXPORT_PROPERTIES ${CMAKE_CUDA_ARCHITECTURES})

endfunction()

# -----------------------------------------------------
# - Macro for adding a binary to the project.
# - It's a macro in order to being able to parse arguments
# - passed to the calling function.
# -----------------------------------------------------
macro(add_nvblox_binary TARGET_NAME BINARY_TYPE)
  # Parse arguments passed to the function
  set(options)
  set(one_value_args TARGET_NAME)
  set(multi_value_args
    SOURCE_FILES
    LINK_LIBRARIES_PUBLIC
    LINK_LIBRARIES_PRIVATE
    INCLUDE_DIRECTORIES_PUBLIC
    INCLUDE_DIRECTORIES_PRIVATE)
  cmake_parse_arguments(PARSE_ARGV 0 arg "${options}" "${one_value_args}" "${multi_value_args}")

  # Check if required arguments were provided
  if(NOT arg_SOURCE_FILES)
    message(FATAL_ERROR "SOURCE_FILES argument is required")
  endif()

  # Add the binary, depending on type
  if (${BINARY_TYPE} STREQUAL "static_library")
    message(STATUS "Adding nvblox static library: ${TARGET_NAME}")
    add_library(${TARGET_NAME} STATIC ${arg_SOURCE_FILES})
  elseif(${BINARY_TYPE} STREQUAL "shared_library")
    message(STATUS "Adding nvblox shared library: ${TARGET_NAME}")
    add_library(${TARGET_NAME} SHARED ${arg_SOURCE_FILES})
  elseif(${BINARY_TYPE} STREQUAL "executable")
    message(STATUS "Adding nvblox executable library: ${TARGET_NAME}")
    add_executable(${TARGET_NAME} ${arg_SOURCE_FILES})
  else()
    message(FATAL_ERROR "Invalid binary type: ${BINARY_TYPE}")
  endif()

  # Add library deps
  target_link_libraries(${TARGET_NAME} PUBLIC ${arg_LINK_LIBRARIES_PUBLIC})
  target_link_libraries(${TARGET_NAME} PRIVATE ${arg_LINK_LIBRARIES_PRIVATE})

  # Add include dir deps
  target_include_directories(${TARGET_NAME} PUBLIC ${arg_INCLUDE_DIRECTORIES_PUBLIC})
  target_include_directories(${TARGET_NAME} PRIVATE ${arg_INCLUDE_DIRECTORIES_PRIVATE})

  # Setup compiler options
  set_nvblox_compiler_options(${TARGET_NAME})
endmacro()

# # -----------------------------------------------------
# # - Function for adding a static library to the project
# # -----------------------------------------------------
function(add_nvblox_static_library TARGET_NAME)
  add_nvblox_binary(${TARGET_NAME} "static_library")
endfunction()

# # -----------------------------------------------------
# # - Function for adding a shared library to the project
# # -----------------------------------------------------
function(add_nvblox_shared_library TARGET_NAME)
  add_nvblox_binary(${TARGET_NAME} "shared_library")
endfunction()

# # -----------------------------------------------------
# # - Function for adding an executable to the project
# # -----------------------------------------------------
function(add_nvblox_executable TARGET_NAME)
  add_nvblox_binary(${TARGET_NAME} "executable")
endfunction()
