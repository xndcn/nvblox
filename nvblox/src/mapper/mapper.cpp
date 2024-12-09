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
#include "nvblox/mapper/mapper.h"

#include "nvblox/geometry/bounding_boxes.h"
#include "nvblox/geometry/bounding_spheres.h"
#include "nvblox/io/layer_cake_io.h"
#include "nvblox/io/mesh_io.h"
#include "nvblox/io/pointcloud_io.h"
#include "nvblox/mapper/internal/mapper_common.h"
#include "nvblox/utils/rates.h"

namespace nvblox {

Mapper::Mapper(float voxel_size_m, MemoryType memory_type,
               ProjectiveLayerType projective_layer_type,
               std::shared_ptr<CudaStream> cuda_stream)
    : cuda_stream_(cuda_stream),
      voxel_size_m_(voxel_size_m),
      memory_type_(memory_type),
      projective_layer_type_(projective_layer_type),
      tsdf_integrator_(cuda_stream),
      lidar_tsdf_integrator_(cuda_stream),
      freespace_integrator_(cuda_stream),
      occupancy_integrator_(cuda_stream),
      lidar_occupancy_integrator_(cuda_stream),
      tsdf_shape_clearer_(cuda_stream),
      color_integrator_(cuda_stream),
      mesh_integrator_(cuda_stream),
      esdf_integrator_(cuda_stream),
      depth_preprocessor_(cuda_stream),
      blocks_to_update_tracker_(projective_layer_type) {
  layers_ =
      LayerCake::create<TsdfLayer, ColorLayer, FreespaceLayer, OccupancyLayer,
                        EsdfLayer, MeshLayer>(voxel_size_m_, memory_type);
  layer_streamers_ =
      LayerCakeStreamer::create<TsdfLayer, ColorLayer, FreespaceLayer,
                                OccupancyLayer, EsdfLayer, MeshLayer>();
  // Make the camera integrators share the same viewpoint cache.
  shareViewpointCaches(&tsdf_integrator_, &occupancy_integrator_,
                       &color_integrator_);
  // Make the LiDAR integrators share the same viewpoint cache
  shareViewpointCaches(&lidar_tsdf_integrator_, &lidar_occupancy_integrator_);
}

Mapper::Mapper(const std::string& map_filepath, MemoryType memory_type,
               std::shared_ptr<CudaStream> cuda_stream)
    : cuda_stream_(cuda_stream),
      memory_type_(memory_type),
      tsdf_integrator_(cuda_stream),
      lidar_tsdf_integrator_(cuda_stream),
      freespace_integrator_(cuda_stream),
      occupancy_integrator_(cuda_stream),
      lidar_occupancy_integrator_(cuda_stream),
      tsdf_shape_clearer_(cuda_stream),
      color_integrator_(cuda_stream),
      mesh_integrator_(cuda_stream),
      esdf_integrator_(cuda_stream),
      depth_preprocessor_(cuda_stream),
      blocks_to_update_tracker_(kDefaultProjectiveLayerType) {
  loadMap(map_filepath);
}

void Mapper::setMapperParams(const MapperParams& params) {
  // ======= MAPPER =======
  // depth preprocessing
  do_depth_preprocessing(params.do_depth_preprocessing);
  depth_preprocessing_num_dilations(params.depth_preprocessing_num_dilations);

  // ======= ESDF INTEGRATOR =======
  esdf_integrator().esdf_slice_min_height(
      params.esdf_integrator_params.esdf_slice_min_height);
  esdf_integrator().esdf_slice_max_height(
      params.esdf_integrator_params.esdf_slice_max_height);
  esdf_integrator().esdf_slice_height(
      params.esdf_integrator_params.esdf_slice_height);
  esdf_integrator().slice_height_above_plane_m(
      params.esdf_integrator_params.slice_height_above_plane_m);
  esdf_integrator().slice_height_thickness_m(
      params.esdf_integrator_params.slice_height_thickness_m);
  esdf_integrator().max_esdf_distance_m(
      params.esdf_integrator_params.esdf_integrator_max_distance_m);
  esdf_integrator().min_weight(
      params.esdf_integrator_params.esdf_integrator_min_weight);
  esdf_integrator().max_site_distance_vox(
      params.esdf_integrator_params.esdf_integrator_max_site_distance_vox);

  // Decay
  exclude_last_view_from_decay(params.exclude_last_view_from_decay);

  // ======= PROJECTIVE INTEGRATOR (TSDF/COLOR/OCCUPANCY)
  // max integration distance
  tsdf_integrator().max_integration_distance_m(
      params.projective_integrator_params
          .projective_integrator_max_integration_distance_m);
  occupancy_integrator().max_integration_distance_m(
      params.projective_integrator_params
          .projective_integrator_max_integration_distance_m);
  color_integrator().max_integration_distance_m(
      params.projective_integrator_params
          .projective_integrator_max_integration_distance_m);
  lidar_tsdf_integrator().max_integration_distance_m(
      params.projective_integrator_params
          .lidar_projective_integrator_max_integration_distance_m);
  lidar_occupancy_integrator().max_integration_distance_m(
      params.projective_integrator_params
          .lidar_projective_integrator_max_integration_distance_m);
  // truncation distance
  tsdf_integrator().truncation_distance_vox(
      params.projective_integrator_params
          .projective_integrator_truncation_distance_vox);
  occupancy_integrator().truncation_distance_vox(
      params.projective_integrator_params
          .projective_integrator_truncation_distance_vox);
  lidar_tsdf_integrator().truncation_distance_vox(
      params.projective_integrator_params
          .projective_integrator_truncation_distance_vox);
  lidar_occupancy_integrator().truncation_distance_vox(
      params.projective_integrator_params
          .projective_integrator_truncation_distance_vox);
  // weighting
  tsdf_integrator().weighting_function_type(
      params.projective_integrator_params.projective_integrator_weighting_mode);
  color_integrator().weighting_function_type(
      params.projective_integrator_params.projective_integrator_weighting_mode);
  // max weight
  tsdf_integrator().max_weight(
      params.projective_integrator_params.projective_integrator_max_weight);
  lidar_tsdf_integrator().max_weight(
      params.projective_integrator_params.projective_integrator_max_weight);
  color_integrator().max_weight(
      params.projective_integrator_params.projective_integrator_max_weight);
  // invalid depth decay
  tsdf_integrator().invalid_depth_decay_factor(
      params.projective_integrator_params
          .projective_tsdf_integrator_invalid_depth_decay_factor);
  lidar_tsdf_integrator().invalid_depth_decay_factor(
      params.projective_integrator_params
          .projective_tsdf_integrator_invalid_depth_decay_factor);

  // ======= OCCUPANCY INTEGRATOR =======
  occupancy_integrator().free_region_occupancy_probability(
      params.occupancy_integrator_params.free_region_occupancy_probability);
  lidar_occupancy_integrator().free_region_occupancy_probability(
      params.occupancy_integrator_params.free_region_occupancy_probability);
  occupancy_integrator().occupied_region_occupancy_probability(
      params.occupancy_integrator_params.occupied_region_occupancy_probability);
  lidar_occupancy_integrator().occupied_region_occupancy_probability(
      params.occupancy_integrator_params.occupied_region_occupancy_probability);
  occupancy_integrator().unobserved_region_occupancy_probability(
      params.occupancy_integrator_params
          .unobserved_region_occupancy_probability);
  lidar_occupancy_integrator().unobserved_region_occupancy_probability(
      params.occupancy_integrator_params
          .unobserved_region_occupancy_probability);
  occupancy_integrator().occupied_region_half_width_m(
      params.occupancy_integrator_params.occupied_region_half_width_m);
  lidar_occupancy_integrator().occupied_region_half_width_m(
      params.occupancy_integrator_params.occupied_region_half_width_m);

  // ======= VIEW CALCULATOR =======
  tsdf_integrator().view_calculator().raycast_subsampling_factor(
      params.view_calculator_params.raycast_subsampling_factor);
  tsdf_integrator().view_calculator().workspace_bounds_type(
      params.view_calculator_params.workspace_bounds_type);
  tsdf_integrator().view_calculator().workspace_bounds_min_corner_m(
      Vector3f(params.view_calculator_params.workspace_bounds_min_corner_x_m,
               params.view_calculator_params.workspace_bounds_min_corner_y_m,
               params.view_calculator_params.workspace_bounds_min_height_m));
  tsdf_integrator().view_calculator().workspace_bounds_max_corner_m(
      Vector3f(params.view_calculator_params.workspace_bounds_max_corner_x_m,
               params.view_calculator_params.workspace_bounds_max_corner_y_m,
               params.view_calculator_params.workspace_bounds_max_height_m));
  color_integrator().view_calculator().raycast_subsampling_factor(
      params.view_calculator_params.raycast_subsampling_factor);
  color_integrator().view_calculator().workspace_bounds_type(
      params.view_calculator_params.workspace_bounds_type);
  color_integrator().view_calculator().workspace_bounds_min_corner_m(
      Vector3f(params.view_calculator_params.workspace_bounds_min_corner_x_m,
               params.view_calculator_params.workspace_bounds_min_corner_y_m,
               params.view_calculator_params.workspace_bounds_min_height_m));
  color_integrator().view_calculator().workspace_bounds_max_corner_m(
      Vector3f(params.view_calculator_params.workspace_bounds_max_corner_x_m,
               params.view_calculator_params.workspace_bounds_max_corner_y_m,
               params.view_calculator_params.workspace_bounds_max_height_m));

  // ======= MESH INTEGRATOR =======
  mesh_integrator().min_weight(
      params.mesh_integrator_params.mesh_integrator_min_weight);
  mesh_integrator().weld_vertices(
      params.mesh_integrator_params.mesh_integrator_weld_vertices);

  // ======= DECAY INTEGRATOR (TSDF/OCCUPANCY)=======
  tsdf_decay_integrator().deallocate_decayed_blocks(
      params.decay_integrator_base_params
          .decay_integrator_deallocate_decayed_blocks);
  occupancy_decay_integrator().deallocate_decayed_blocks(
      params.decay_integrator_base_params
          .decay_integrator_deallocate_decayed_blocks);

  // ======= TSDF DECAY INTEGRATOR =======
  tsdf_decay_integrator().decay_factor(
      params.tsdf_decay_integrator_params.tsdf_decay_factor);
  tsdf_decay_integrator().decayed_weight_threshold(
      params.tsdf_decay_integrator_params.tsdf_decayed_weight_threshold);
  tsdf_decay_integrator().set_free_distance_on_decayed(
      params.tsdf_decay_integrator_params.tsdf_set_free_distance_on_decayed);
  tsdf_decay_integrator().free_distance_vox(
      params.tsdf_decay_integrator_params.tsdf_decayed_free_distance_vox);

  // ======= OCCUPANCY DECAY INTEGRATOR =======
  occupancy_decay_integrator().free_region_decay_probability(
      params.occupancy_decay_integrator_params.free_region_decay_probability);
  occupancy_decay_integrator().occupied_region_decay_probability(
      params.occupancy_decay_integrator_params
          .occupied_region_decay_probability);
  occupancy_decay_integrator().decay_to_free(
      params.occupancy_decay_integrator_params.occupancy_decay_to_free);

  // ======= FREESPACE INTEGRATOR =======
  freespace_integrator().max_tsdf_distance_for_occupancy_m(
      params.freespace_integrator_params.max_tsdf_distance_for_occupancy_m);
  freespace_integrator().max_unobserved_to_keep_consecutive_occupancy_ms(
      params.freespace_integrator_params
          .max_unobserved_to_keep_consecutive_occupancy_ms);
  freespace_integrator().min_duration_since_occupied_for_freespace_ms(
      params.freespace_integrator_params
          .min_duration_since_occupied_for_freespace_ms);
  freespace_integrator().min_consecutive_occupancy_duration_for_reset_ms(
      params.freespace_integrator_params
          .min_consecutive_occupancy_duration_for_reset_ms);
  freespace_integrator().check_neighborhood(
      params.freespace_integrator_params.check_neighborhood);
}

const DepthImage& Mapper::preprocessDepthImageAsync(
    const DepthImageConstView& depth_image) {
  // NOTE(alexmillane): We return a const reference to an image, to
  // avoid reallocating.
  // Copy in the depth image
  preprocessed_depth_image_->copyFromAsync(depth_image, *cuda_stream_);
  // Dilate the invalid regions
  if (depth_preprocessing_num_dilations_ > 0) {
    depth_preprocessor_.dilateInvalidRegionsAsync(
        depth_preprocessing_num_dilations_, preprocessed_depth_image_.get());
  } else {
    LOG(WARNING) << "You requested preprocessing, but requested "
                 << depth_preprocessing_num_dilations_
                 << "invalid region dilations. Currenly dilation is the only "
                    "preprocessing step, so doing nothing.";
  }
  return *preprocessed_depth_image_;
}

void Mapper::integrateDepth(const DepthImage& depth_frame,
                            const Transform& T_L_C, const Camera& camera) {
  integrateDepth(MaskedDepthImageConstView(depth_frame), T_L_C, camera);
}

void Mapper::integrateDepth(const MaskedDepthImageConstView& depth_frame,
                            const Transform& T_L_C, const Camera& camera) {
  CHECK(projective_layer_type_ != ProjectiveLayerType::kNone)
      << "You are trying to update on an inexistent projective layer.";
  // If requested, we perform preprocessing of the depth image. At the moment
  // this is just (optional) dilation of the invalid regions.
  MaskedDepthImageConstView depth_image_for_integration = depth_frame;
  if (do_depth_preprocessing_) {
    depth_image_for_integration = MaskedDepthImageConstView(
        preprocessDepthImageAsync(depth_frame), depth_frame.mask());
  }

  // Call the integrator.
  std::vector<Index3D> updated_blocks;
  if (hasTsdfLayer(projective_layer_type_)) {
    tsdf_integrator_.integrateFrame(
        MaskedDepthImageConstView(depth_image_for_integration), T_L_C, camera,
        layers_.getPtr<TsdfLayer>(), &updated_blocks);

    layers_.getPtr<TsdfLayer>()->updateGpuHash(*cuda_stream_);
  } else if (projective_layer_type_ == ProjectiveLayerType::kOccupancy) {
    occupancy_integrator_.integrateFrame(
        depth_image_for_integration, T_L_C, camera,
        layers_.getPtr<OccupancyLayer>(), &updated_blocks);

    layers_.getPtr<OccupancyLayer>()->updateGpuHash(*cuda_stream_);
  }

  // Save the viewpoint for use in viewpoint exclusion.
  if (exclude_last_view_from_decay_) {
    if (!last_depth_image_.has_value()) {
      LOG(INFO) << "Allocating space for last depth image";
      last_depth_image_ =
          DepthImage(depth_image_for_integration.rows(),
                     depth_image_for_integration.cols(), MemoryType::kDevice);
    } else {
      last_depth_image_.value().resizeAsync(depth_image_for_integration.rows(),
                                            depth_image_for_integration.cols(),
                                            *cuda_stream_);
    }

    // NOTE(alexmillane): We could get rid this copy by using a double buffer.
    last_depth_image_.value().copyFromAsync(depth_image_for_integration,
                                            *cuda_stream_);
    last_depth_camera_ = camera;
    last_depth_T_L_C_ = T_L_C;
  }

  blocks_to_update_tracker_.addBlocksToUpdate(updated_blocks);
}

void Mapper::integrateLidarDepth(const DepthImage& depth_frame,
                                 const Transform& T_L_C, const Lidar& lidar) {
  CHECK(projective_layer_type_ != ProjectiveLayerType::kNone)
      << "You are trying to update on an inexistent projective layer.";
  // Call the integrator.
  std::vector<Index3D> updated_blocks;
  if (hasTsdfLayer(projective_layer_type_)) {
    lidar_tsdf_integrator_.integrateFrame(depth_frame, T_L_C, lidar,
                                          layers_.getPtr<TsdfLayer>(),
                                          &updated_blocks);

    layers_.getPtr<TsdfLayer>()->updateGpuHash(*cuda_stream_);
  } else if (projective_layer_type_ == ProjectiveLayerType::kOccupancy) {
    lidar_occupancy_integrator_.integrateFrame(depth_frame, T_L_C, lidar,
                                               layers_.getPtr<OccupancyLayer>(),
                                               &updated_blocks);

    layers_.getPtr<OccupancyLayer>()->updateGpuHash(*cuda_stream_);
  }

  blocks_to_update_tracker_.addBlocksToUpdate(updated_blocks);
}

void Mapper::integrateColor(const ColorImage& color_frame,
                            const Transform& T_L_C, const Camera& camera) {
  // Color is only integrated for Tsdf layers (not for occupancy)
  if (hasTsdfLayer(projective_layer_type_)) {
    color_integrator_.integrateFrame(color_frame, T_L_C, camera,
                                     layers_.get<TsdfLayer>(),
                                     layers_.getPtr<ColorLayer>());

    layers_.getPtr<ColorLayer>()->updateGpuHash(*cuda_stream_);
  }
}

void Mapper::decayTsdf() {
  // TODO(remos): In the future we could exclude the blocks not decayed, from
  // the blocks requiring an update.
  const std::vector<Index3D> all_blocks =
      layers_.get<TsdfLayer>().getAllBlockIndices();
  blocks_to_update_tracker_.addBlocksToUpdate(all_blocks);

  // Decay - either all blocks or exclude a view
  std::vector<Index3D> deallocated_blocks;
  if (exclude_last_view_from_decay_) {
    if (last_depth_image_.has_value() && last_depth_camera_.has_value() &&
        last_depth_T_L_C_.has_value()) {
      deallocated_blocks = tsdf_decay_integrator_.decay(
          layers_.getPtr<TsdfLayer>(),
          ViewBasedInclusionData(
              last_depth_T_L_C_.value(), last_depth_camera_.value(),
              &last_depth_image_.value(),
              tsdf_integrator_.max_integration_distance_m(),
              tsdf_integrator_.get_truncation_distance_m(voxel_size_m_)),
          *cuda_stream_);
    }
  } else {
    deallocated_blocks = tsdf_decay_integrator_.decay(
        layers_.getPtr<TsdfLayer>(), *cuda_stream_);
  }

  // Clear the blocks that got deallocated in the tsdf layer also in the esdf,
  // freespace and mesh layers.
  clearBlocksInLayers(deallocated_blocks);
  layers_.getPtr<TsdfLayer>()->updateGpuHash(*cuda_stream_);
}

void Mapper::decayOccupancy() {
  // TODO(remos): In the future we could exclude the blocks not decayed, from
  // the blocks requiring an update.
  const std::vector<Index3D> all_blocks =
      layers_.get<OccupancyLayer>().getAllBlockIndices();
  blocks_to_update_tracker_.addBlocksToUpdate(all_blocks);

  // Decay - either all blocks or exclude a view
  std::vector<Index3D> deallocated_blocks;
  if (exclude_last_view_from_decay_) {
    if (last_depth_image_ && last_depth_camera_ && last_depth_T_L_C_) {
      deallocated_blocks = occupancy_decay_integrator_.decay(
          layers_.getPtr<OccupancyLayer>(), std::nullopt,
          ViewBasedInclusionData(
              last_depth_T_L_C_.value(), last_depth_camera_.value(),
              &last_depth_image_.value(),
              occupancy_integrator_.max_integration_distance_m(),
              occupancy_integrator_.get_truncation_distance_m(voxel_size_m_)),
          *cuda_stream_);
    }
  } else {
    deallocated_blocks = occupancy_decay_integrator_.decay(
        layers_.getPtr<OccupancyLayer>(), *cuda_stream_);
  }

  // Clear the blocks that got deallocated in the occupancy layer also in the
  // esdf, freespace and mesh layers.
  clearBlocksInLayers(deallocated_blocks);
  layers_.getPtr<OccupancyLayer>()->updateGpuHash(*cuda_stream_);
}

void Mapper::clearTsdfInsideShapes(const std::vector<BoundingShape>& shapes) {
  const std::vector<Index3D> updated_blocks =
      tsdf_shape_clearer_.clear(shapes, layers_.getPtr<TsdfLayer>());
  blocks_to_update_tracker_.addBlocksToUpdate(updated_blocks);
}

void Mapper::updateFreespace(Time update_time_ms,
                             UpdateFullLayer update_full_layer) {
  updateFreespace(update_time_ms, std::nullopt, update_full_layer);
}

void Mapper::updateFreespace(Time update_time_ms, const Transform& T_L_C,
                             const Camera& camera,
                             const DepthImage& depth_frame,
                             UpdateFullLayer update_full_layer) {
  // The freespace integrator only updates voxel that are in view and within the
  // negative truncation distance. Due to noisy depth measurements, a voxel
  // might occasionaly end up on the "wrong" side of the truncation distance
  // and would thus not be updated. To mitigate the effect of this on/off
  // switching, we inflate the truncation distance.
  constexpr float kTruncationDistanceMultipler = 2.F;

  updateFreespace(
      update_time_ms,
      ViewBasedInclusionData(
          T_L_C, camera, &depth_frame,
          tsdf_integrator_.max_integration_distance_m(),
          kTruncationDistanceMultipler *
              tsdf_integrator_.get_truncation_distance_m(voxel_size_m_)),
      update_full_layer);
}

void Mapper::updateFreespace(
    Time update_time_ms, std::optional<ViewBasedInclusionData> view_to_update,
    UpdateFullLayer update_full_layer) {
  CHECK(hasFreespaceLayer(projective_layer_type_))
      << "Trying to update the freespace layer while it is not enabled.";

  // Get the freespace blocks that need an update
  std::vector<Index3D> blocks_to_update =
      getBlocksToUpdate(BlocksToUpdateType::kFreespace, update_full_layer);

  // Call the integrator.
  freespace_integrator_.updateFreespaceLayer(
      blocks_to_update, update_time_ms, layers_.get<TsdfLayer>(),
      view_to_update, layers_.getPtr<FreespaceLayer>());

  // Mark blocks as updated
  blocks_to_update_tracker_.markBlocksAsUpdated(BlocksToUpdateType::kFreespace);
  layers_.getPtr<FreespaceLayer>()->updateGpuHash(*cuda_stream_);
}

void Mapper::updateMesh(UpdateFullLayer update_full_layer) {
  // Mesh is only updated for Tsdf layers (not for occupancy)
  if (!hasTsdfLayer(projective_layer_type_)) {
    return;
  } else {
    // Get the mesh blocks that need an update
    std::vector<Index3D> blocks_to_update =
        getBlocksToUpdate(BlocksToUpdateType::kMesh, update_full_layer);

    // Call the integrator.
    mesh_integrator_.integrateBlocksGPU(layers_.get<TsdfLayer>(),
                                        blocks_to_update,
                                        layers_.getPtr<MeshLayer>());

    mesh_integrator_.colorMesh(layers_.get<ColorLayer>(), blocks_to_update,
                               layers_.getPtr<MeshLayer>());

    blocks_to_update_tracker_.markBlocksAsUpdated(BlocksToUpdateType::kMesh);
  }
}

void Mapper::updateEsdf(UpdateFullLayer update_full_layer) {
  CHECK(esdf_mode_ != EsdfMode::k2D) << "Currently, we limit computation of "
                                        "the ESDF to 2d *or* 3d. Not both.";
  esdf_mode_ = EsdfMode::k3D;

  // Get the esdf blocks that need an update
  std::vector<Index3D> blocks_to_update =
      getBlocksToUpdate(BlocksToUpdateType::kEsdf, update_full_layer);

  if (projective_layer_type_ == ProjectiveLayerType::kTsdfWithFreespace) {
    // Passing a freespace layer to the integrator for checking if
    // candidate esdf sites fall into freespace
    esdf_integrator_.integrateBlocks(
        layers_.get<TsdfLayer>(), layers_.get<FreespaceLayer>(),
        blocks_to_update, layers_.getPtr<EsdfLayer>());
  } else if (projective_layer_type_ == ProjectiveLayerType::kTsdf) {
    esdf_integrator_.integrateBlocks(layers_.get<TsdfLayer>(), blocks_to_update,
                                     layers_.getPtr<EsdfLayer>());
  } else if (projective_layer_type_ == ProjectiveLayerType::kOccupancy) {
    esdf_integrator_.integrateBlocks(layers_.get<OccupancyLayer>(),
                                     blocks_to_update,
                                     layers_.getPtr<EsdfLayer>());
  }

  // Mark blocks as updated
  blocks_to_update_tracker_.markBlocksAsUpdated(BlocksToUpdateType::kEsdf);
}

void Mapper::updateEsdfSlice(UpdateFullLayer update_full_layer,
                             std::optional<Plane> ground_plane) {
  CHECK(esdf_mode_ != EsdfMode::k3D) << "Currently, we limit computation of "
                                        "the ESDF to 2d *or* 3d. Not both.";
  esdf_mode_ = EsdfMode::k2D;

  // Get the esdf blocks that need an update
  std::vector<Index3D> blocks_to_update =
      getBlocksToUpdate(BlocksToUpdateType::kEsdf, update_full_layer);

  if (ground_plane) {
    if (projective_layer_type_ == ProjectiveLayerType::kTsdfWithFreespace) {
      // Passing a freespace layer to the integrator for checking if
      // candidate esdf sites fall into freespace
      esdf_integrator_.integrateSlice(
          layers_.get<TsdfLayer>(), layers_.get<FreespaceLayer>(),
          blocks_to_update, ground_plane.value(), layers_.getPtr<EsdfLayer>());
    } else if (projective_layer_type_ == ProjectiveLayerType::kTsdf) {
      esdf_integrator_.integrateSlice(layers_.get<TsdfLayer>(),
                                      blocks_to_update, ground_plane.value(),
                                      layers_.getPtr<EsdfLayer>());

    } else if (projective_layer_type_ == ProjectiveLayerType::kOccupancy) {
      esdf_integrator_.integrateSlice(layers_.get<OccupancyLayer>(),
                                      blocks_to_update, ground_plane.value(),
                                      layers_.getPtr<EsdfLayer>());
    }
  } else {
    if (projective_layer_type_ == ProjectiveLayerType::kTsdfWithFreespace) {
      // Passing a freespace layer to the integrator for checking if
      // candidate esdf sites fall into freespace
      esdf_integrator_.integrateSlice(
          layers_.get<TsdfLayer>(), layers_.get<FreespaceLayer>(),
          blocks_to_update, layers_.getPtr<EsdfLayer>());
    } else if (projective_layer_type_ == ProjectiveLayerType::kTsdf) {
      esdf_integrator_.integrateSlice(layers_.get<TsdfLayer>(),
                                      blocks_to_update,
                                      layers_.getPtr<EsdfLayer>());
    } else if (projective_layer_type_ == ProjectiveLayerType::kOccupancy) {
      esdf_integrator_.integrateSlice(layers_.get<OccupancyLayer>(),
                                      blocks_to_update,
                                      layers_.getPtr<EsdfLayer>());
    }
  }
  // Mark blocks as updated
  blocks_to_update_tracker_.markBlocksAsUpdated(BlocksToUpdateType::kEsdf);
}

void Mapper::clearOutsideRadius(const Vector3f& center, float radius) {
  std::vector<Index3D> block_indices_for_deletion;
  if (hasTsdfLayer(projective_layer_type_)) {
    block_indices_for_deletion = getBlocksOutsideRadius(
        layers_.get<TsdfLayer>().getAllBlockIndices(),
        layers_.get<TsdfLayer>().block_size(), center, radius);
    layers_.getPtr<TsdfLayer>()->clearBlocksAsync(block_indices_for_deletion,
                                                  *cuda_stream_);
  } else if (projective_layer_type_ == ProjectiveLayerType::kOccupancy) {
    block_indices_for_deletion = getBlocksOutsideRadius(
        layers_.get<OccupancyLayer>().getAllBlockIndices(),
        layers_.get<OccupancyLayer>().block_size(), center, radius);
    layers_.getPtr<OccupancyLayer>()->clearBlocksAsync(
        block_indices_for_deletion, *cuda_stream_);
  }

  // Clear the blocks that got deallocated in the tsdf/occupancy layer also in
  // the esdf, freespace and mesh layers.
  clearBlocksInLayers(block_indices_for_deletion);
}

void Mapper::markUnobservedTsdfFreeInsideRadius(const Vector3f& center,
                                                float radius) {
  CHECK_GT(radius, 0.0f);
  std::vector<Index3D> updated_blocks;
  if (hasTsdfLayer(projective_layer_type_)) {
    tsdf_integrator_.markUnobservedFreeInsideRadius(
        center, radius, layers_.getPtr<TsdfLayer>(), &updated_blocks);
  } else if (projective_layer_type_ == ProjectiveLayerType::kOccupancy) {
    occupancy_integrator_.markUnobservedFreeInsideRadius(
        center, radius, layers_.getPtr<OccupancyLayer>(), &updated_blocks);
  }

  blocks_to_update_tracker_.addBlocksToUpdate(updated_blocks);
}

std::vector<Index3D> Mapper::getClearedBlocks(
    const std::vector<Index3D>& blocks_to_ignore) {
  // Remove the blocks_to_ignore from the set.
  for (const Index3D& idx : blocks_to_ignore) {
    cleared_blocks_.erase(idx);
  }
  // Get the vector of cleared  blocks
  const std::vector<Index3D> cleared_blocks_vec(cleared_blocks_.begin(),
                                                cleared_blocks_.end());
  // Clear the set.
  cleared_blocks_.clear();
  return cleared_blocks_vec;
}

std::vector<Index3D> Mapper::getBlocksToUpdate(
    BlocksToUpdateType blocks_to_update_type,
    UpdateFullLayer update_full_layer) const {
  if (update_full_layer == UpdateFullLayer::kYes) {
    if (hasTsdfLayer(projective_layer_type_)) {
      return layers_.get<TsdfLayer>().getAllBlockIndices();
    } else {
      return layers_.get<OccupancyLayer>().getAllBlockIndices();
    }
  } else {
    return blocks_to_update_tracker_.getBlocksToUpdate(blocks_to_update_type);
  }
}

void Mapper::clearBlocksInLayers(const std::vector<Index3D>& blocks_to_clear) {
  // Clear the mesh and color blocks.
  layers_.getPtr<ColorLayer>()->clearBlocksAsync(blocks_to_clear,
                                                 *cuda_stream_);
  if (hasTsdfLayer(projective_layer_type_)) {
    layers_.getPtr<MeshLayer>()->clearBlocksAsync(blocks_to_clear,
                                                  *cuda_stream_);
  }
  // Clear the freespace blocks, if existent.
  if (hasFreespaceLayer(projective_layer_type_)) {
    layers_.getPtr<FreespaceLayer>()->clearBlocksAsync(blocks_to_clear,
                                                       *cuda_stream_);
  }

  // Clear the blocks in the esdf layer.
  if (esdf_mode_ == EsdfMode::k3D) {
    // In the 3D case this is easy.
    layers_.getPtr<EsdfLayer>()->clearBlocksAsync(blocks_to_clear,
                                                  *cuda_stream_);
  } else {
    // In the 2D case we need to check if an occupancy/tsdf block is left in the
    // vertical column (z-axis) for every 2d esdf block.

    // Get the slice bound indices.
    const float block_size = layers_.get<EsdfLayer>().block_size();
    const int min_slice_bound_index_z =
        getBlockIndexFromPositionInLayer(
            block_size,
            Vector3f(0.0f, 0.0f, esdf_integrator().esdf_slice_min_height()))
            .z();
    const int max_slice_bound_index_z =
        getBlockIndexFromPositionInLayer(
            block_size,
            Vector3f(0.0f, 0.0f, esdf_integrator().esdf_slice_max_height()))
            .z();
    const int esdf_index_z =
        getBlockIndexFromPositionInLayer(
            block_size,
            Vector3f(0.0f, 0.0f, esdf_integrator().esdf_slice_height()))
            .z();
    CHECK_GE(max_slice_bound_index_z, min_slice_bound_index_z);
    const size_t num_blocks_in_vertical_column =
        max_slice_bound_index_z - min_slice_bound_index_z + 1;

    // Iterate over the blocks that got cleared in the projective layer.
    for (const Index3D& block_index : blocks_to_clear) {
      Index3D esdf_block_index(block_index.x(), block_index.y(), esdf_index_z);
      if (!layers_.getPtr<EsdfLayer>()->isBlockAllocated(esdf_block_index)) {
        // If there is no esdf block we are done.
        continue;
      }
      // Otherwise we check if there is an projective block in the vertical
      // column.
      bool has_block_in_column = false;
      Index3D projective_block_index(block_index.x(), block_index.y(), 0);
      for (size_t i = 0; i < num_blocks_in_vertical_column; i++) {
        projective_block_index.z() = min_slice_bound_index_z + i;
        // Check on tsdf or occupancy layer.
        if (hasTsdfLayer(projective_layer_type_)) {
          has_block_in_column |=
              layers_.get<TsdfLayer>().isBlockAllocated(projective_block_index);
        } else if (projective_layer_type_ == ProjectiveLayerType::kOccupancy) {
          has_block_in_column |= layers_.get<OccupancyLayer>().isBlockAllocated(
              projective_block_index);
        }
        if (has_block_in_column) {
          break;
        }
      }
      if (!has_block_in_column) {
        // No corresponding projective block found. So let's clear this esdf
        // block.
        layers_.getPtr<EsdfLayer>()->clearBlockAsync(esdf_block_index,
                                                     *cuda_stream_);
      }
    }
  }

  // We don't need to update the deallocated blocks.
  blocks_to_update_tracker_.removeBlocksToUpdate(blocks_to_clear);

  // We need to keep track of cleared blocks to delete them in our
  // visualizer.
  cleared_blocks_.insert(blocks_to_clear.begin(), blocks_to_clear.end());
}

bool Mapper::saveLayerCake(const std::string& filename) const {
  return io::writeLayerCakeToFile(filename, layers_, *cuda_stream_);
}

bool Mapper::saveLayerCake(const char* filename) const {
  return saveLayerCake(std::string(filename));
}

bool Mapper::loadMap(const std::string& filename) {
  LayerCake new_cake = io::loadLayerCakeFromFile(filename, memory_type_);
  // Will return an empty cake if anything went wrong.
  if (new_cake.empty()) {
    LOG(ERROR) << "Failed to load map from file: " << filename;
    return false;
  }

  TsdfLayer* tsdf_layer = new_cake.getPtr<TsdfLayer>();

  if (tsdf_layer == nullptr) {
    LOG(ERROR) << "No TSDF layer could be loaded from file: " << filename
               << ". Aborting loading.";
    return false;
  }
  // Double check what's going on with the voxel sizes.
  if (tsdf_layer->voxel_size() != voxel_size_m_) {
    LOG(INFO) << "Setting the voxel size from the loaded map as: "
              << tsdf_layer->voxel_size();
    voxel_size_m_ = tsdf_layer->voxel_size();
  }

  // Now we're happy, let's swap the cakes.
  layers_ = std::move(new_cake);
  blocks_to_update_tracker_.markBlocksAsUpdated(BlocksToUpdateType::kEsdf);

  // We can't serialize mesh layers yet so we have to add a new mesh layer.
  std::unique_ptr<MeshLayer> mesh(
      new MeshLayer(layers_.getPtr<TsdfLayer>()->block_size(), memory_type_));
  layers_.insert(typeid(MeshLayer), std::move(mesh));
  updateMesh(UpdateFullLayer::kYes);

  return true;
}

bool Mapper::loadMap(const char* filename) {
  return loadMap(std::string(filename));
}

bool Mapper::saveMeshAsPly(const std::string& filepath) const {
  return io::outputMeshLayerToPly(mesh_layer(), filepath);
}

bool Mapper::saveEsdfAsPly(const std::string& filename) const {
  return io::outputVoxelLayerToPly(esdf_layer(), filename);
}

bool Mapper::saveTsdfAsPly(const std::string& filename) const {
  return io::outputVoxelLayerToPly(tsdf_layer(), filename);
}

bool Mapper::saveOccupancyAsPly(const std::string& filename) const {
  return io::outputVoxelLayerToPly(occupancy_layer(), filename);
}

bool Mapper::saveFreespaceAsPly(const std::string& filename) const {
  return io::outputVoxelLayerToPly(freespace_layer(), filename);
}

parameters::ParameterTreeNode Mapper::getParameterTree(
    const std::string& name_remap) const {
  using parameters::ParameterTreeNode;
  const std::string name = (name_remap.empty()) ? "mapper" : name_remap;
  return ParameterTreeNode(
      name,
      {ParameterTreeNode("voxel_size_m", voxel_size_m_),
       ParameterTreeNode("memory_type", memory_type_),
       ParameterTreeNode("projective_layer_type", projective_layer_type_),
       ParameterTreeNode("esdf_mode", esdf_mode_),
       ParameterTreeNode("do_depth_preprocessing", do_depth_preprocessing_),
       ParameterTreeNode("depth_preprocessing_num_dilations",
                         depth_preprocessing_num_dilations_),
       ParameterTreeNode("exclude_last_view_from_decay",
                         exclude_last_view_from_decay_),
       tsdf_integrator_.getParameterTree("camera_tsdf_integrator"),
       lidar_tsdf_integrator_.getParameterTree("lidar_tsdf_integrator"),
       color_integrator_.getParameterTree(),
       occupancy_integrator_.getParameterTree("camera_occupancy_integrator"),
       lidar_occupancy_integrator_.getParameterTree(
           "lidar_occupancy_integrator"),
       esdf_integrator_.getParameterTree(), mesh_integrator_.getParameterTree(),
       occupancy_decay_integrator_.getParameterTree(),
       tsdf_decay_integrator_.getParameterTree(),
       freespace_integrator_.getParameterTree()});
}

std::string Mapper::getParametersAsString() const {
  return parameterTreeToString(getParameterTree());
}

std::shared_ptr<const SerializedMeshLayer> Mapper::serializedMeshLayer() {
  return layer_streamers_.getSerializedLayer<MeshLayer>();
}

std::shared_ptr<const SerializedTsdfLayer> Mapper::serializedTsdfLayer() {
  return layer_streamers_.getSerializedLayer<TsdfLayer>();
}

std::shared_ptr<const SerializedOccupancyLayer>
Mapper::serializedOccupancyLayer() {
  return layer_streamers_.getSerializedLayer<OccupancyLayer>();
}

std::shared_ptr<const SerializedFreespaceLayer>
Mapper::serializedFreespaceLayer() {
  return layer_streamers_.getSerializedLayer<FreespaceLayer>();
}

std::shared_ptr<const SerializedEsdfLayer> Mapper::serializedEsdfLayer() {
  return layer_streamers_.getSerializedLayer<EsdfLayer>();
}

std::shared_ptr<const SerializedColorLayer> Mapper::serializedColorLayer() {
  return layer_streamers_.getSerializedLayer<ColorLayer>();
}

void Mapper::serializeColorTsdfAndFreespaceLayers(
    const std::vector<Index3D>& blocks_to_serialize,
    const LayerTypeBitMask layer_type_bitmask, const float bandwidth_limit_mbps,
    const BlockExclusionParams& exclusion_params) {
  // Color layer doesn't contain any geometry. Therefore we first need to
  // serialize the TSDF layer and make sure that the color layer is serialized
  // with the same block indices.
  CHECK(layer_type_bitmask & LayerType::kTsdf);

  // Serialize TSDF
  layer_streamers_.estimateBandwidthAndSerialize(
      tsdf_layer(), blocks_to_serialize, "tsdf", exclusion_params,
      bandwidth_limit_mbps, *cuda_stream_);

  // Serialize color
  layer_streamers_.serializeAllBlocks(
      color_layer(), serializedTsdfLayer()->block_indices, *cuda_stream_);
  CHECK_EQ(serializedTsdfLayer()->block_indices.size(),
           serializedColorLayer()->block_indices.size());

  // Optionally serialize freespace
  if (projective_layer_type() == ProjectiveLayerType::kTsdfWithFreespace &&
      layer_type_bitmask & LayerType::kFreespace) {
    layer_streamers_.serializeAllBlocks(
        freespace_layer(), serializedTsdfLayer()->block_indices, *cuda_stream_);
    CHECK_EQ(serializedFreespaceLayer()->block_indices.size(),
             serializedColorLayer()->block_indices.size());
  }
}

void Mapper::serializeSelectedLayers(
    const LayerTypeBitMask layer_type_bitmask, const float bandwidth_limit_mbps,
    const BlockExclusionParams& exclusion_params) {
  // Figure out which blocks to serialize
  // Note that all layers need to be serialized simultaneously since they all
  // share the same BlocksToUpdate tracker
  std::vector<Index3D> blocks_to_serialize =
      blocks_to_update_tracker_.getBlocksToUpdate(
          BlocksToUpdateType::kLayerStreamer);

  // Color layer is handled separately since we also need to serialize geometry
  // blocks in order to visualize it.
  if (layer_type_bitmask & LayerType::kColor) {
    serializeColorTsdfAndFreespaceLayers(
        blocks_to_serialize, layer_type_bitmask, bandwidth_limit_mbps,
        exclusion_params);
  } else {
    // Mesh
    if (layer_type_bitmask & LayerType::kMesh) {
      layer_streamers_.estimateBandwidthAndSerialize(
          mesh_layer(), blocks_to_serialize, "mesh", exclusion_params,
          bandwidth_limit_mbps, *cuda_stream_);
    }

    // TSDF layer
    if (layer_type_bitmask & LayerType::kTsdf) {
      layer_streamers_.estimateBandwidthAndSerialize(
          tsdf_layer(), blocks_to_serialize, "tsdf", exclusion_params,
          bandwidth_limit_mbps, *cuda_stream_);
    }

    // ESDF layer
    if (layer_type_bitmask & LayerType::kEsdf) {
      layer_streamers_.estimateBandwidthAndSerialize(
          esdf_layer(), blocks_to_serialize, "esdf", exclusion_params,
          bandwidth_limit_mbps, *cuda_stream_);
    }

    // Occupancy layer
    if (layer_type_bitmask & LayerType::kOccupancy) {
      layer_streamers_.estimateBandwidthAndSerialize(
          occupancy_layer(), blocks_to_serialize, "occupancy", exclusion_params,
          bandwidth_limit_mbps, *cuda_stream_);
    }

    // Freespace layer
    if (layer_type_bitmask & LayerType::kFreespace) {
      layer_streamers_.estimateBandwidthAndSerialize(
          freespace_layer(), blocks_to_serialize, "freespace", exclusion_params,
          bandwidth_limit_mbps, *cuda_stream_);
    }
  }

  // Keep track of serialized blocks
  blocks_to_update_tracker_.markBlocksAsUpdated(
      BlocksToUpdateType::kLayerStreamer);
}

void Mapper::markBlocksForUpdate(const std::vector<Index3D>& blocks) {
  blocks_to_update_tracker_.addBlocksToUpdate(blocks);
}

}  // namespace nvblox
