/*
Copyright 2022-2024 NVIDIA CORPORATION

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

#include <optional>
#include <unordered_set>

#include "nvblox/core/hash.h"
#include "nvblox/core/parameter_tree.h"
#include "nvblox/dynamics/dynamics_detection.h"
#include "nvblox/integrators/esdf_integrator.h"
#include "nvblox/integrators/freespace_integrator.h"
#include "nvblox/integrators/occupancy_decay_integrator.h"
#include "nvblox/integrators/projective_color_integrator.h"
#include "nvblox/integrators/projective_occupancy_integrator.h"
#include "nvblox/integrators/projective_tsdf_integrator.h"
#include "nvblox/integrators/shape_clearer.h"
#include "nvblox/integrators/tsdf_decay_integrator.h"
#include "nvblox/map/blocks_to_update_tracker.h"
#include "nvblox/map/blox.h"
#include "nvblox/map/common_names.h"
#include "nvblox/map/layer.h"
#include "nvblox/map/layer_cake.h"
#include "nvblox/map/voxels.h"
#include "nvblox/mapper/mapper_params.h"
#include "nvblox/mesh/mesh_integrator.h"
#include "nvblox/semantics/image_masker.h"
#include "nvblox/sensors/camera.h"
#include "nvblox/sensors/depth_preprocessing.h"
#include "nvblox/sensors/lidar.h"
#include "nvblox/serialization/layer_cake_streamer.h"
#include "nvblox/serialization/layer_streamer.h"

namespace nvblox {

/// The ESDF mode. Enum indicates if an Mapper is configured for 3D or 2D
/// Esdf production, or that this has not yet been determined (kUnset).
enum class EsdfMode { k3D, k2D, kUnset };

template <>
inline std::string toString(const EsdfMode& esdf_mode) {
  switch (esdf_mode) {
    case EsdfMode::k3D:
      return "k3D";
      break;
    case EsdfMode::k2D:
      return "k2D";
      break;
    case EsdfMode::kUnset:
      return "kUnset";
      break;
    default:
      LOG(FATAL) << "Not implemented";
      return "";
      break;
  }
}

/// Whether to update the full layer on calls to updateMesh(),
/// updateFreespace() and updateEsdf() respectively or only the blocks that
/// require and update (tracked by BlocksToUpdateTracker).
enum class UpdateFullLayer { kNo, kYes };

/// The mapper classes wraps layers and integrators together.
/// In the base class we only specify that a mapper should contain map layers
/// and leave it up to sub-classes to add functionality.
class MapperBase {
 public:
  static constexpr ProjectiveLayerType kDefaultProjectiveLayerType =
      ProjectiveLayerType::kTsdf;

  MapperBase() = default;
  virtual ~MapperBase() = default;

  /// Move
  MapperBase(MapperBase&& other) = default;
  MapperBase& operator=(MapperBase&& other) = default;

 protected:
  /// Map layers
  LayerCake layers_;
};

/// The Mapper class is what we consider the default mapping behaviour in
/// nvblox.
/// Contains:
/// - TsdfLayer, OccupancyLayer, ColorLayer, EsdfLayer, MeshLayer
/// - Integrators associated with these layer types.
///
/// Exposes functions for:
/// - Integrating depth/rgbd images, 3D LiDAR scans, and color images.
/// - Functions for generating Meshes, ESDF, and ESDF-slices.
class Mapper : public MapperBase {
 public:
  /// Parameter defaults: See mapper_params.h

  Mapper() = delete;
  /// Constructor
  /// @param voxel_size_m The voxel size in meters for the contained layers.
  /// @param projective_layer_type The layer type to which the projective
  ///        data is integrated (either tsdf or occupancy).
  /// @param memory_type In which type of memory the layers should be stored.
  /// @param cuda_stream Optional cuda stream to perform all work on.
  Mapper(float voxel_size_m, MemoryType memory_type = MemoryType::kDevice,
         ProjectiveLayerType projective_layer_type = ProjectiveLayerType::kTsdf,
         std::shared_ptr<CudaStream> cuda_stream =
             std::make_shared<CudaStreamOwning>());
  virtual ~Mapper() = default;

  /// Constructor which initializes from a saved map.
  /// @param map_filepath Path to the serialized map to be loaded.
  /// @param memory_type In which type of memory the layers should be stored.
  Mapper(const std::string& map_filepath,
         MemoryType memory_type = MemoryType::kDevice,
         std::shared_ptr<CudaStream> cuda_stream =
             std::make_shared<CudaStreamOwning>());

  Mapper(Mapper&& other) = default;
  Mapper& operator=(Mapper&& other) = default;

  /// Set the parameters of the mapper from the parameter struct.
  /// @param params The struct containing the params.
  void setMapperParams(const MapperParams& params);

  /// Integrates a depth frame
  ///
  /// The depth frame will be integrated into either the TSDF or occupancy
  /// reconstruction, depending on the mapping mode.
  ///
  /// Can be called with either a DepthImage or a MaskedDepthImage. If a mask is
  /// provided, only active (non-zero) depth pixels will become part of the
  /// reconstruction. If no mask is provided, all pixels will be treated as
  /// active. The unmasked pixels are treated differently depending on the
  /// projective integrator type used:
  ///
  /// TSDF: Unmasked depth is used to update voxels only up until the
  /// positive truncation distance. The end effect is that the surface will not
  /// be reconstructed, but any voxels in front of the surface will be cleared.
  ///
  /// Occupancy: Voxels updated with unmasked depth are treated as
  /// "unobserved".
  ///
  ///@param depth_frame Depth frame to integrate. Depth in the image is
  ///                   specified as a float representing meters.
  ///@param T_L_C Pose of the camera, specified as a transform from
  ///             Camera-frame to Layer-frame transform.
  ///@param camera Intrinsics model of the camera.
  void integrateDepth(const DepthImage& depth_frame, const Transform& T_L_C,
                      const Camera& camera);
  void integrateDepth(const MaskedDepthImageConstView& depth_frame,
                      const Transform& T_L_C, const Camera& camera);

  /// Integrates a color frame into the reconstruction.
  ///@param color_frame Color image to integrate.
  ///@param T_L_C Pose of the camera, specified as a transform from
  ///             Camera-frame to Layer-frame transform.
  ///@param camera Intrinsics model of the camera.
  void integrateColor(const ColorImage& color_frame, const Transform& T_L_C,
                      const Camera& camera);

  /// Integrates a 3D LiDAR scan into the reconstruction.
  ///@param depth_frame Depth image representing the LiDAR scan.
  ///@param T_L_C Pose of the LiDAR, specified as a transform from LiDAR-frame
  ///             to Layer-frame transform.
  ///@param lidar Intrinsics model of the LiDAR.
  void integrateLidarDepth(const DepthImage& depth_frame,
                           const Transform& T_L_C, const Lidar& lidar);

  /// Decay the TSDF layer (reduce weights)
  void decayTsdf();

  /// Decay the full occupancy layer.
  void decayOccupancy();

  /// @brief Clear the TSDF layer inside the passed shapes.
  /// @param shapes Vector of shapes to clear.
  void clearTsdfInsideShapes(const std::vector<BoundingShape>& shapes);

  /// Updates the freespace blocks.
  /// @param update_time_ms The time of the update in miliseconds.
  /// @param update_full_layer Whether to update the full layer or only the
  /// blocks that require and update.
  void updateFreespace(Time update_time_ms, UpdateFullLayer update_full_layer =
                                                UpdateFullLayer::kNo);

  /// Updates the freespace blocks (in view).
  /// @param update_time_ms The time of the update in miliseconds.
  /// @param T_L_C The pose of the camera.
  /// @param camera The intrinsics of the camera.
  /// @param depth_frame The depth image.
  /// @param update_full_layer Whether to update the full layer or only the
  /// blocks that require and update.
  void updateFreespace(
      Time update_time_ms, const Transform& T_L_C, const Camera& camera,
      const DepthImage& depth_frame,
      UpdateFullLayer update_full_layer = UpdateFullLayer::kNo);

  /// Updates the mesh blocks.
  /// @param update_full_layer Whether to update the full layer or only the
  /// blocks that require and update. Useful if loading a layer cake without a
  /// mesh layer, for example.
  void updateMesh(UpdateFullLayer update_full_layer = UpdateFullLayer::kNo);

  /// Serialize selected layers.
  ///
  /// Will update serialized layers to contain new blocks added to the map since
  /// the last call to this function. The resulting serialized layers can be
  /// accessed by individual getters.
  ///
  /// @param layer_type_bitmask Bitmask determining which layers to serialize
  /// @param bandwidth_limit_mbps Max bandwidth. Set to negative value for
  /// unlimited.
  ///        Note that this limit is per layer, i.e. the actual bandwidth will
  ///        exceed the limit if more than one layer is serialized.
  /// @param maybe_exclusion_center Optional center for radiual block exclusion.
  /// Typically set to robot translation.
  void serializeSelectedLayers(
      const LayerTypeBitMask layer_type_bitmask,
      const float bandwidth_limit_mbps,
      const BlockExclusionParams& maybe_exclusion_params =
          BlockExclusionParams());

  /// Return the serialized mesh layer.
  std::shared_ptr<const SerializedMeshLayer> serializedMeshLayer();

  /// Return the serialized TSDF layer.
  std::shared_ptr<const SerializedTsdfLayer> serializedTsdfLayer();

  /// Return the serialized ESDF layer.
  std::shared_ptr<const SerializedEsdfLayer> serializedEsdfLayer();

  /// Return the serialized color layer.
  std::shared_ptr<const SerializedColorLayer> serializedColorLayer();

  /// Return the serialized occupancy layer.
  std::shared_ptr<const SerializedOccupancyLayer> serializedOccupancyLayer();

  /// Return the serialized freespace layer.
  std::shared_ptr<const SerializedFreespaceLayer> serializedFreespaceLayer();

  /// Updates the ESDF blocks.
  /// Note that currently we limit the Mapper class to calculating *either*
  /// the 2D or 3D ESDF, not both. Which is to be calculated is determined by
  /// the first call to updateEsdf().
  /// @param update_full_layer Whether to update the full layer or only the
  /// blocks that require and update.
  ///@return std::vector<Index3D> The indices of the blocks that were updated
  ///        in this call.
  void updateEsdf(UpdateFullLayer update_full_layer = UpdateFullLayer::kNo);

  /// Updates the ESDF blocks.
  /// Note that currently we limit the Mapper class to calculating *either*
  /// the 2D or 3D ESDF, not both. Which is to be calculated is determined by
  /// the first call to updateEsdf(). This function operates by collapsing a
  /// finite thickness slice of the 3D TSDF into a binary obstacle map, and
  /// then generating the 2D ESDF. The mapper parameters define the limits of
  /// the 3D slice that are considered. Note that the resultant 2D ESDF is
  /// stored in a single voxel thick layer in ESDF layer.
  /// @param update_full_layer Whether to update the full layer or only the
  /// blocks that require and update.
  /// @param ground_plane If provided, the esdf is sliced along a parameterized
  /// instead of horizontally.
  /// @return The indices of the blocks that were updated in this call.
  ///@return std::vector<Index3D>  The indices of the blocks that were updated
  ///        in this call.
  void updateEsdfSlice(UpdateFullLayer update_full_layer = UpdateFullLayer::kNo,
                       std::optional<Plane> ground_plane = std::nullopt);

  /// Clears the reconstruction outside a radius around a center point,
  /// deallocating the memory.
  ///@param center The center of the keep-sphere.
  ///@param radius The radius of the keep-sphere.
  void clearOutsideRadius(const Vector3f& center, float radius);

  /// Allocates blocks touched by radius and gives their voxels some small
  /// positive weight.
  /// @param center The center of allocation-sphere
  /// @param radius The radius of allocation-sphere
  void markUnobservedTsdfFreeInsideRadius(const Vector3f& center, float radius);

  /// Gets the preprocessed version of the last depth image passed to
  /// integrateDepth(). Note that we return a shared_ptr to a buffered depth
  /// image inside the mapper to avoid copying the image. Subsequent calls to
  /// integrateDepth will change the contents of this image.
  /// @return The preprocessed DepthImage.
  const std::shared_ptr<const DepthImage> getPreprocessedDepthImage() const {
    return preprocessed_depth_image_;
  }

  /// Getter
  ///@return const LayerCake& The collection of layers mapped.
  const LayerCake& layers() const { return layers_; }
  /// Getter
  ///@return const TsdfLayer& TSDF layer
  const TsdfLayer& tsdf_layer() const { return layers_.get<TsdfLayer>(); }
  /// Getter
  ///@return const OccupancyLayer& occupancy layer
  const OccupancyLayer& occupancy_layer() const {
    return layers_.get<OccupancyLayer>();
  }
  /// Getter
  ///@return const FreespaceLayer& freespace layer
  const FreespaceLayer& freespace_layer() const {
    return layers_.get<FreespaceLayer>();
  }
  /// Getter
  ///@return const ColorLayer& Color layer
  const ColorLayer& color_layer() const { return layers_.get<ColorLayer>(); }
  /// Getter
  ///@return const EsdfLayer& ESDF layer
  const EsdfLayer& esdf_layer() const { return layers_.get<EsdfLayer>(); }
  /// Getter
  ///@return const MeshLayer& Mesh layer
  const MeshLayer& mesh_layer() const { return layers_.get<MeshLayer>(); }
  /// Getter
  /// @return const LayerCakeStreamer& The layer cake streamer.
  const LayerCakeStreamer& layer_streamers() const { return layer_streamers_; }

  /// Getter
  ///@return LayerCake& The collection of layers mapped.
  LayerCake& layers() { return layers_; }
  /// Getter
  ///@return TsdfLayer& TSDF layer
  TsdfLayer& tsdf_layer() { return *layers_.getPtr<TsdfLayer>(); }
  /// Getter
  ///@return OccupancyLayer& occupancy layer
  OccupancyLayer& occupancy_layer() {
    return *layers_.getPtr<OccupancyLayer>();
  }
  /// Getter
  ///@return FreespaceLayer& freespace layer
  FreespaceLayer& freespace_layer() {
    return *layers_.getPtr<FreespaceLayer>();
  }
  /// Getter
  ///@return ColorLayer& Color layer
  ColorLayer& color_layer() { return *layers_.getPtr<ColorLayer>(); }
  /// Getter
  ///@return EsdfLayer& ESDF layer
  EsdfLayer& esdf_layer() { return *layers_.getPtr<EsdfLayer>(); }
  /// Getter
  ///@return MeshLayer& Mesh layer
  MeshLayer& mesh_layer() { return *layers_.getPtr<MeshLayer>(); }
  /// Getter
  /// @return const LayerCakeStreamer& The layer cake streamer.
  LayerCakeStreamer& layer_streamers() { return layer_streamers_; }

  /// Getter
  ///@return const ProjectiveTsdfIntegrator& TSDF integrator used for
  ///        depth/rgbd frame integration.
  const ProjectiveTsdfIntegrator& tsdf_integrator() const {
    return tsdf_integrator_;
  }
  /// Getter
  ///@return const ProjectiveOccupancyIntegrator& occupancy integrator used
  /// for
  ///        depth/rgbd frame integration.
  const ProjectiveOccupancyIntegrator& occupancy_integrator() const {
    return occupancy_integrator_;
  }
  /// Getter
  ///@return const FreespaceIntegrator& freespace integrator used for
  ///        updating the freespace layer according to a tsdf layer.
  const FreespaceIntegrator& freespace_integrator() const {
    return freespace_integrator_;
  }
  /// Getter
  ///@return const ProjectiveTsdfIntegrator& TSDF integrator used for
  ///        3D LiDAR scan integration.
  const ProjectiveTsdfIntegrator& lidar_tsdf_integrator() const {
    return lidar_tsdf_integrator_;
  }
  /// Getter
  ///@return const ProjectiveOccupancyIntegrator& occupancy integrator used
  /// for 3D LiDAR scan integration.
  const ProjectiveOccupancyIntegrator& lidar_occupancy_integrator() const {
    return lidar_occupancy_integrator_;
  }
  /// Getter
  ///@return const OccupancyDecayIntegrator& occupancy integrator used for
  ///        decaying an occupancy layer towards 0.5 occupancy probability.
  const OccupancyDecayIntegrator& occupancy_decay_integrator() const {
    return occupancy_decay_integrator_;
  }
  /// Getter
  ///@return const TsdfDecayIntegrator& tsdf integrator used for
  ///        decaying an tsdf layer
  const TsdfDecayIntegrator& tsdf_decay_integrator() const {
    return tsdf_decay_integrator_;
  }
  /// Getter
  ///@return const TsdfShapeClearer& TSDF clearer used for
  ///        clearing tsdf inside given shapes.
  const TsdfShapeClearer& tsdf_shape_clearer() const {
    return tsdf_shape_clearer_;
  }
  /// Getter
  ///@return const ProjectiveColorIntegrator& Color integrator.
  const ProjectiveColorIntegrator& color_integrator() const {
    return color_integrator_;
  }
  /// Getter
  ///@return const MeshIntegrator& Mesh integrator
  const MeshIntegrator& mesh_integrator() const { return mesh_integrator_; }
  /// Getter
  ///@return const EsdfIntegrator& ESDF integrator
  const EsdfIntegrator& esdf_integrator() const { return esdf_integrator_; }

  /// Getter
  ///@return ProjectiveTsdfIntegrator& TSDF integrator used for
  ///        depth/rgbd frame integration.
  ProjectiveTsdfIntegrator& tsdf_integrator() { return tsdf_integrator_; }
  /// Getter
  ///@return ProjectiveOccupancyIntegrator& occupancy integrator used for
  ///        depth/rgbd frame integration.
  ProjectiveOccupancyIntegrator& occupancy_integrator() {
    return occupancy_integrator_;
  }
  /// Getter
  ///@return FreespaceIntegrator& freespace integrator used for
  ///        updating the freespace layer according to a tsdf layer.
  FreespaceIntegrator& freespace_integrator() { return freespace_integrator_; }
  /// Getter
  ///@return ProjectiveTsdfIntegrator& TSDF integrator used for
  ///        3D LiDAR scan integration.
  ProjectiveTsdfIntegrator& lidar_tsdf_integrator() {
    return lidar_tsdf_integrator_;
  }
  /// Getter
  ///@return ProjectiveOccupancyIntegrator& occupancy integrator used for
  ///        3D LiDAR scan integration.
  ProjectiveOccupancyIntegrator& lidar_occupancy_integrator() {
    return lidar_occupancy_integrator_;
  }
  /// Getter
  ///@return OccupancyDecayIntegrator& occupancy decay integrator used for
  ///        decaying an occupancy layer towards 0.5 occupancy probability.
  OccupancyDecayIntegrator& occupancy_decay_integrator() {
    return occupancy_decay_integrator_;
  }
  /// Getter
  ///@return TsdfDecayIntegrator& TSDF decay integrator used for decaying a
  /// TSDF
  ///        layer (through reduction of voxel weights).
  TsdfDecayIntegrator& tsdf_decay_integrator() {
    return tsdf_decay_integrator_;
  }
  /// Getter
  ///@return TsdfShapeClearer& TSDF clearer used for
  ///        clearing tsdf inside given shapes.
  TsdfShapeClearer& tsdf_shape_clearer() { return tsdf_shape_clearer_; }
  /// Getter
  ///@return ProjectiveColorIntegrator& Color integrator.
  ProjectiveColorIntegrator& color_integrator() { return color_integrator_; }
  /// Getter
  ///@return MeshIntegrator& Mesh integrator
  MeshIntegrator& mesh_integrator() { return mesh_integrator_; }
  /// Getter
  ///@return EsdfIntegrator& ESDF integrator
  EsdfIntegrator& esdf_integrator() { return esdf_integrator_; }
  /// Getter
  /// @return The voxel size in meters
  float voxel_size_m() const { return voxel_size_m_; };
  /// Getter
  /// @return The type of projective layer we're mapping
  ProjectiveLayerType projective_layer_type() const {
    return projective_layer_type_;
  };

  /// Getter
  /// @return Whether we should perform preprocessing on input DepthImages
  bool do_depth_preprocessing() const { return do_depth_preprocessing_; }
  /// Setter
  /// @param do_depth_preprocessing Whether to perform depth preprocessing.
  void do_depth_preprocessing(const bool do_depth_preprocessing) {
    do_depth_preprocessing_ = do_depth_preprocessing;
  }
  /// Getter How many times to run a 3x3 dilation kernel on the invalid mask
  /// of the depth image.
  /// @return The number of application of the 3x3 dilation kernel.
  int depth_preprocessing_num_dilations() const {
    return depth_preprocessing_num_dilations_;
  }
  /// Setter. See depth_preprocessing_num_dilations()
  /// @param depth_preprocessing_num_dilations How many times to run the
  /// kernel.
  void depth_preprocessing_num_dilations(
      const int depth_preprocessing_num_dilations) {
    CHECK_GE(depth_preprocessing_num_dilations, 0);
    depth_preprocessing_num_dilations_ = depth_preprocessing_num_dilations;
  }

  /// Whether to exclude voxel contained observed in the the last depth frame
  /// passed to integrateDepth from the voxels which are decayed.
  bool exclude_last_view_from_decay() const {
    return exclude_last_view_from_decay_;
  }
  /// A parameter setter
  /// See exclude_last_view_from_decay()
  /// @param exclude_last_view_from_decay
  void exclude_last_view_from_decay(const bool exclude_last_view_from_decay) {
    exclude_last_view_from_decay_ = exclude_last_view_from_decay;
  }

  /// Saving and loading functions.
  /// Saving a map will serialize the TSDF and ESDF layers to a file.
  ///@param filename
  ///@return true
  ///@return false
  bool saveLayerCake(const std::string& filename) const;
  bool saveLayerCake(const char* filename) const;
  /// Loading the map will load a the TSDF and ESDF layers from a file.
  /// Will clear anything in the map already.
  bool loadMap(const std::string& filename);
  bool loadMap(const char* filename);

  /// Write mesh as a PLY
  /// @param filename Path to output PLY file.
  /// @return bool Flag indicating if write was successful.
  bool saveMeshAsPly(const std::string& filename) const;

  /// Writes the Esdf as a PLY
  /// @param filename Path to the output PLY file.
  /// @return bool Flag indicating if the write was successful.
  bool saveEsdfAsPly(const std::string& filename) const;

  /// Writes the Tsdf as a PLY
  /// @param filename Path to the output PLY file.
  /// @return bool Flag indicating if the write was successful.
  bool saveTsdfAsPly(const std::string& filename) const;

  /// Writes the freespace as a PLY
  /// @param filename Path to the output PLY file.
  /// @return bool Flag indicating if the write was successful.
  bool saveFreespaceAsPly(const std::string& filename) const;

  /// Writes the occupancy as a PLY
  /// @param filename Path to the output PLY file.
  /// @return bool Flag indicating if the write was successful.
  bool saveOccupancyAsPly(const std::string& filename) const;

  /// Return the parameter tree.
  /// @return the parameter tree
  virtual parameters::ParameterTreeNode getParameterTree(
      const std::string& name_remap = std::string()) const;

  /// Return the parameter tree represented as a string
  /// @return the parameter tree string
  virtual std::string getParametersAsString() const;

  /// @brief Get the blocks that have been cleared
  /// since the last call of the function. This information is needed to
  /// remove them from the visualizer.
  /// @param blocks_to_ignore Blocks that should not part of the returned
  /// vector.
  /// @return Vector of cleared block indices.
  std::vector<Index3D> getClearedBlocks(
      const std::vector<Index3D>& blocks_to_ignore);

  /// @brief Marks a list of block indices as needing an update.
  /// the mapper is tracking changes to map to batch updates to dependent parts.
  /// This can be useful when directly modify the layers in the mapper.
  /// Internally
  /// @param blocks Indices that require an update.
  void markBlocksForUpdate(const std::vector<Index3D>& blocks);

 protected:
  /// Update the freespace layer, with an optional viewpoint.
  void updateFreespace(Time update_time_ms,
                       std::optional<ViewBasedInclusionData> view_to_update,
                       UpdateFullLayer update_full_layer);

  /// Serialize layers needed for color visualization
  void serializeColorTsdfAndFreespaceLayers(
      const std::vector<Index3D>& blocks_to_serialize,
      const LayerTypeBitMask layer_type_bitmask,
      const float bandwidth_limit_mbps,
      const BlockExclusionParams& exclusion_params);

  /// Perform preprocessing on a depth image
  const DepthImage& preprocessDepthImageAsync(
      const DepthImageConstView& depth_image);

  /// @brief Get the esdf, mesh or freespace blocks that need and update.
  /// @param blocks_to_update_type The type of blocks you want to get the
  /// vector for.
  /// @param update_full_layer Whether to return all block indices (for
  /// updating the full layer) or only the blocks that need an update.
  /// @return Vector of block indices to update.
  std::vector<Index3D> getBlocksToUpdate(
      BlocksToUpdateType blocks_to_update_type,
      UpdateFullLayer update_full_layer) const;

  /// @brief Deallocate blocks int the esdf, mesh and freespace layer.
  /// @param blocks_to_clear Vector of blocks to clear.
  void clearBlocksInLayers(const std::vector<Index3D>& blocks_to_clear);

  /// The CUDA stream that mapper work is processed on
  std::shared_ptr<CudaStream> cuda_stream_;

  /// The size of the voxels to be used in the TSDF, ESDF, Color layers.
  float voxel_size_m_;
  /// The storage location for the TSDF, ESDF, Color, and Mesh Layers.
  MemoryType memory_type_;
  /// The layer type to which the projective data is integrated (either tsdf
  /// or occupancy).
  ProjectiveLayerType projective_layer_type_ = kDefaultProjectiveLayerType;

  /// This class can be used to generate *either* (not both) the 2D or 3D
  /// ESDF. The mode used is determined by the first call to either
  /// updateEsdf() or updateEsdfSlice(). This member tracks which mode we're
  /// in.
  EsdfMode esdf_mode_ = EsdfMode::kUnset;

  /// Integrators
  ProjectiveTsdfIntegrator tsdf_integrator_;
  ProjectiveTsdfIntegrator lidar_tsdf_integrator_;
  FreespaceIntegrator freespace_integrator_;
  ProjectiveOccupancyIntegrator occupancy_integrator_;
  ProjectiveOccupancyIntegrator lidar_occupancy_integrator_;
  OccupancyDecayIntegrator occupancy_decay_integrator_;
  TsdfDecayIntegrator tsdf_decay_integrator_;
  TsdfShapeClearer tsdf_shape_clearer_;
  ProjectiveColorIntegrator color_integrator_;
  MeshIntegrator mesh_integrator_;
  EsdfIntegrator esdf_integrator_;

  // Layer Streamers
  LayerCakeStreamer layer_streamers_;

  /// Preprocessing depth maps prior to integration.
  /// Currently, the only preprocessing step is to dilate the invalid regions
  /// of the input depth image. We have found this useful to reduce the
  /// depth-bleeding effects on the intel realsense.
  bool do_depth_preprocessing_ = kDoDepthPrepocessingParamDesc.default_value;
  int depth_preprocessing_num_dilations_ =
      kDepthPreprocessingNumDilationsParamDesc.default_value;
  DepthPreprocessor depth_preprocessor_;
  std::shared_ptr<DepthImage> preprocessed_depth_image_ =
      std::make_shared<DepthImage>(MemoryType::kDevice);

  /// Helper to keep track of which blocks need to be updated on the next
  /// calls to updateMesh(), updateFreespace() upd updateEsdf() respectively.
  BlocksToUpdateTracker blocks_to_update_tracker_;

  /// Keeping track of the mesh blocks that got deleted in the mesh layer.
  Index3DSet cleared_blocks_;

  /// Whether to exclude the last depth frustum from the decay
  bool exclude_last_view_from_decay_ =
      kExcludeLastViewFromDecayParamDesc.default_value;
  /// Last known depth viewpoint for view-based decay exclusion
  std::optional<DepthImage> last_depth_image_;
  std::optional<Camera> last_depth_camera_;
  std::optional<Transform> last_depth_T_L_C_;
};

}  // namespace nvblox
