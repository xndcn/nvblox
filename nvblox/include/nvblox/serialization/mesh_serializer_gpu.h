/*
Copyright 2022 NVIDIA CORPORATION

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#pragma once

#include <memory>
#include <vector>

#include "nvblox/core/cuda_stream.h"
#include "nvblox/core/types.h"
#include "nvblox/core/unified_vector.h"
#include "nvblox/map/common_names.h"
#include "nvblox/map/layer.h"
#include "nvblox/serialization/internal/serialization_gpu.h"

namespace nvblox {

/// Container for storing a serialized mesh
struct SerializedMeshLayer {
  /// Serialized mesh components
  host_vector<Vector3f> vertices;
  host_vector<Color> colors;
  host_vector<int> triangle_indices;

  /// Offsets for each mesh block in the output vector.
  /// Size of offsets is num_blocks+1. The first element is always
  /// zero and the last element always equals the total size of the serialized
  /// vector. The size of block n can be computed as offsets[n+1] -
  /// offsets[n]
  host_vector<int32_t> vertex_block_offsets;
  host_vector<int32_t> triangle_index_block_offsets;

  /// Indices of serialized mesh blocks
  std::vector<Index3D> block_indices;

  /// Get an iterator to the given triangle block
  host_vector<int32_t>::const_iterator triangleBlockItr(
      size_t block_index) const {
    CHECK(block_index < vertex_block_offsets.size());
    return std::next(triangle_indices.begin(),
                     triangle_index_block_offsets[block_index]);
  }

  /// Get a vertex given a block index and a vertex index inside the block
  const Vector3f& getVertex(size_t block_index, size_t vertex_index) const {
    CHECK(block_index < vertex_block_offsets.size() - 1);
    return vertices[vertex_block_offsets[block_index] + vertex_index];
  }

  /// Get a Color given a block index and a vertex index inside the block
  const Color& getColor(size_t block_index, size_t vertex_index) const {
    CHECK(block_index < vertex_block_offsets.size() - 1);
    return colors[vertex_block_offsets[block_index] + vertex_index];
  }

  /// Get a Triangle index given a block index and a triangle index inside the
  /// block
  const int& getTriangleIndex(size_t block_index, size_t triangle_index) const {
    CHECK(block_index < triangle_index_block_offsets.size() - 1);
    return triangle_indices[triangle_index_block_offsets[block_index] +
                            triangle_index];
  }

  /// Helper function to get num vertices in a block
  size_t getNumVerticesInBlock(size_t block_index) const {
    CHECK(block_index < vertex_block_offsets.size() - 1);
    return vertex_block_offsets[block_index + 1] -
           vertex_block_offsets[block_index];
  }

  /// Helper function to get num triangles in a block
  size_t getNumTriangleIndicesInBlock(size_t block_index) const {
    CHECK(block_index < triangle_index_block_offsets.size() - 1);
    return triangle_index_block_offsets[block_index + 1] -
           triangle_index_block_offsets[block_index];
  }
};

/// Class for serialization
///
/// Mesh needs special treatment since the data int he blocks are stored
/// as struct-of-arrays rather than array-of-structs
class MeshSerializerGpu {
 public:
  MeshSerializerGpu();
  virtual ~MeshSerializerGpu() = default;
  using SerializedLayerType = SerializedMeshLayer;

  /// Serialize a mesh layer
  ///
  /// All requested blocks will be serialized and placed in output host
  /// vectors. This implementation is more effective than issuing a memcpy
  /// for each block.
  ///
  /// @attention: Input mesh layer must be in device or unified memory
  ///
  /// @param mesh_layer                  Mesh layer to serialize
  /// @param block_indices_to_serialize  Requested block indices
  /// @param cuda_stream                 Cuda stream
  std::shared_ptr<const SerializedMeshLayer> serialize(
      const MeshLayer& mesh_layer,
      const std::vector<Index3D>& block_indices_to_serialize,
      const CudaStream& cuda_stream);

  /// Get the serialized mesh
  std::shared_ptr<const SerializedMeshLayer> getSerializedLayer() const {
    return serialized_mesh_;
  }

 private:
  LayerSerializerGpuInternal<MeshLayer, Vector3f> vertex_serializer_;
  LayerSerializerGpuInternal<MeshLayer, Color> color_serializer_;
  LayerSerializerGpuInternal<MeshLayer, int> triangle_index_serializer_;

  std::shared_ptr<SerializedMeshLayer> serialized_mesh_;
};

}  // namespace nvblox
