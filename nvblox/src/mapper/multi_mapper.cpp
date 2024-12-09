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
#include "nvblox/mapper/multi_mapper.h"

#include "nvblox/geometry/bounding_spheres.h"
#include "nvblox/io/layer_cake_io.h"
#include "nvblox/mapper/internal/mapper_common.h"
#include "nvblox/sensors/mask_preprocessor.h"

namespace nvblox {

// The background mapper is handling static objects and freespace (with
// occupancy or tsdf)
ProjectiveLayerType findBackgroundLayerType(MappingType mapping_type) {
  switch (mapping_type) {
    case MappingType::kStaticTsdf:
      return ProjectiveLayerType::kTsdf;
    case MappingType::kStaticOccupancy:
      return ProjectiveLayerType::kOccupancy;
    case MappingType::kDynamic:
      return ProjectiveLayerType::kTsdfWithFreespace;
    case MappingType::kHumanWithStaticTsdf:
      return ProjectiveLayerType::kTsdf;
    case MappingType::kHumanWithStaticOccupancy:
      return ProjectiveLayerType::kOccupancy;
    default:
      LOG(FATAL) << "Requested mapping type not implemented.";
  };
}

// The foreground mapper is handling general dynamics and humans (with
// occupancy)
ProjectiveLayerType findForegroundLayerType(MappingType mapping_type) {
  switch (mapping_type) {
    case MappingType::kStaticTsdf:
      return ProjectiveLayerType::kNone;
    case MappingType::kStaticOccupancy:
      return ProjectiveLayerType::kNone;
    case MappingType::kDynamic:
      return ProjectiveLayerType::kOccupancy;
    case MappingType::kHumanWithStaticTsdf:
      return ProjectiveLayerType::kOccupancy;
    case MappingType::kHumanWithStaticOccupancy:
      return ProjectiveLayerType::kOccupancy;
    default:
      LOG(FATAL) << "Requested mapping type not implemented.";
  };
}

MultiMapper::MultiMapper(float voxel_size_m, MappingType mapping_type,
                         EsdfMode esdf_mode, MemoryType memory_type,
                         std::shared_ptr<CudaStream> cuda_stream)
    : mapping_type_(mapping_type),
      esdf_mode_(esdf_mode),
      dynamic_detector_(cuda_stream),
      mask_preprocessor_(cuda_stream),
      ground_plane_estimator_(cuda_stream),
      cuda_stream_(cuda_stream) {
  // Initialize the multi mapper. Composed of:
  // - foreground occupancy mapper for dynamic objects or humans
  // - background mapper for static objects (with either an occupancy or tsdf
  //   layer)
  const ProjectiveLayerType background_layer_type =
      findBackgroundLayerType(mapping_type);
  const ProjectiveLayerType foreground_layer_type =
      findForegroundLayerType(mapping_type);

  // Note that we're creating new cuda streams for the two mappers so we can
  // parallelize work on the GPU.
  background_mapper_ =
      std::make_shared<Mapper>(voxel_size_m, memory_type, background_layer_type,
                               std::make_shared<CudaStreamOwning>());
  foreground_mapper_ =
      std::make_shared<Mapper>(voxel_size_m, memory_type, foreground_layer_type,
                               std::make_shared<CudaStreamOwning>());

  // NOTE(alexmillane): Right now we don't consider the mask state when
  // determining the blocks in view in the integrators. For this reason we're
  // able to share the caches between foreground and background mappers. This
  // provides a significant performance boost. However, if the situation changes
  // in the future, and we start to consider the masks for viewpoint
  // calculations, we should not share caches, because it will lead to incorrect
  // results.

  // Make the camera integrators share the same viewpoint cache.
  shareViewpointCaches(&foreground_mapper_->tsdf_integrator(),       // NOLINT
                       &foreground_mapper_->occupancy_integrator(),  // NOLINT
                       &foreground_mapper_->color_integrator(),      // NOLINT
                       &background_mapper_->tsdf_integrator(),       // NOLINT
                       &background_mapper_->occupancy_integrator(),  // NOLINT
                       &background_mapper_->color_integrator());
  // Make the LiDAR integrators share the same viewpoint cache
  shareViewpointCaches(
      &foreground_mapper_->lidar_tsdf_integrator(),       // NOLINT
      &foreground_mapper_->lidar_occupancy_integrator(),  // NOLINT
      &background_mapper_->lidar_tsdf_integrator(),       // NOLINT
      &background_mapper_->lidar_occupancy_integrator());
}

void MultiMapper::setMultiMapperParams(
    const MultiMapperParams& multi_mapper_params) {
  params_ = multi_mapper_params;
  ground_plane_estimator().ground_points_candidates_min_z_m(
      params_.ground_plane_estimator_params.ground_points_candidates_min_z_m);
  ground_plane_estimator().ground_points_candidates_max_z_m(
      params_.ground_plane_estimator_params.ground_points_candidates_max_z_m);
  ground_plane_estimator().ransac_plane_fitter().ransac_distance_threshold_m(
      params_.ransac_plane_fitter_params.ransac_distance_threshold_m);
  ground_plane_estimator().ransac_plane_fitter().num_ransac_iterations(
      params_.ransac_plane_fitter_params.num_ransac_iterations);
}

void MultiMapper::setMapperParams(
    const MapperParams& background_mapper_params,
    const std::optional<MapperParams>& foreground_mapper_params) {
  background_mapper_->setMapperParams(background_mapper_params);
  if (foreground_mapper_params) {
    foreground_mapper_->setMapperParams(foreground_mapper_params.value());
  }
}

void MultiMapper::integrateDepth(const DepthImage& depth_frame,
                                 const Transform& T_L_CD,
                                 const Camera& depth_camera,
                                 const std::optional<Time>& update_time_ms) {
  if (!isDynamicMapping(mapping_type_)) {
    // For static mapping only integrate to the background mapper
    background_mapper_->integrateDepth(depth_frame, T_L_CD, depth_camera);
  } else {
    CHECK(update_time_ms);

    // To speed-up execution, the functions called here are grouped into blocks
    // that can be launched in parallel without race conditions.
    //
    // The blocks are determinied by tabulating the R/W access of each
    // function/resource and then group functions that does not read and write
    // to the same resource. We also select a flow that doesn't require storing
    // any data from the previous iteration.
    //
    // FUNCTION  	         TL  FL  OL  DI  FD  DD
    // integrateTsdf             W           R
    // updateFreespace	         R   W
    // computeDynamics	             R       R       W
    // removeSmall                               W   R
    // integrateOccupancy                W       R
    //
    // TL: TSDF layer
    // FL: Freespace Layer
    // OL: Occupacy Layer
    // DI: Depth image
    // FD: Foreground depth image
    // DD: DynamicsDetector
    //
    // The selected blocks:
    //    1. IntegreateTsdf + computeDynamcs
    //    2. removeSmallConnetedComponents
    //    3. updateFreespace + integrateOccupancy

    {  // Block 1
      timing::Timer timer("multi_mapper/integrate_depth/dynamic_block1");

      // TODO(dtingdahl) Reduce overhead by recycling threads instead of
      // re-creating them.
      std::vector<std::thread> threads;

      // Integrate TSDF
      threads.push_back(std::thread([&]() {
        background_mapper_->integrateDepth(depth_frame, T_L_CD, depth_camera);
      }));

      // Compute dynamic mask. Note that we're using the freespace layer
      // computed during the previous update. This should be fine since the
      // freespace layer is designed to reacting slow to changes.
      threads.push_back(std::thread([&]() {
        dynamic_detector_.computeDynamics(depth_frame,
                                          background_mapper_->freespace_layer(),
                                          depth_camera, T_L_CD);
      }));

      // sync threads
      std::for_each(threads.begin(), threads.end(),
                    [](std::thread& t) { t.join(); });
      threads.clear();
    }

    {  // Block 2 (This block is launched synchronously since there is ony one
       // call)
      timing::Timer timer("multi_mapper/integrate_depth/dynamic_block2");

      // Update dynamic mask
      const MonoImage& dynamic_mask = dynamic_detector_.getDynamicMaskImage();

      // Remove small components (assumed to be noise) from the mask
      if (params_.remove_small_connected_components) {
        mask_preprocessor_.removeSmallConnectedComponents(
            dynamic_mask, params_.connected_mask_component_size_threshold,
            &cleaned_dynamic_mask_);
      } else {
        cleaned_dynamic_mask_.copyFromAsync(dynamic_mask, *cuda_stream_);
      }
    }

    // Block 3
    timing::Timer timer("multi_mapper/integrate_depth/dynamic_block3");
    std::vector<std::thread> threads;

    // Update occupancy
    threads.push_back(std::thread([&]() {
      foreground_mapper_->integrateDepth(
          MaskedDepthImageConstView(depth_frame, cleaned_dynamic_mask_), T_L_CD,
          depth_camera);
    }));

    // Update freespace.
    threads.push_back(std::thread([&]() {  // NOLINT
      background_mapper_->updateFreespace(update_time_ms.value(), T_L_CD,
                                          depth_camera, depth_frame);
    }));

    // sync threads
    std::for_each(threads.begin(), threads.end(),
                  [](std::thread& t) { t.join(); });
  }
}

void MultiMapper::integrateDepth(const DepthImage& depth_frame,
                                 const MonoImage& mask, const Transform& T_L_CD,
                                 const Transform& T_CM_CD,
                                 const Camera& depth_camera,
                                 const Camera& mask_camera) {
  CHECK(isHumanMapping(mapping_type_))
      << "Passing a mask to integrateDepth is only valid for human "
         "mapping.";

  // Remove small components (assumed to be noise) from the mask
  if (params_.remove_small_connected_components) {
    mask_preprocessor_.removeSmallConnectedComponents(
        mask, params_.connected_mask_component_size_threshold,
        &cleaned_semantic_mask_);
  } else {
    cleaned_semantic_mask_.copyFromAsync(mask, *cuda_stream_);
  }
  // Split into foreground and background depth frame
  image_masker_.splitImageOnGPU(
      depth_frame, cleaned_semantic_mask_, T_CM_CD, depth_camera, mask_camera,
      &depth_frame_background_, &depth_frame_foreground_,
      &foreground_depth_overlay_);

  // Integrate the frames to the respective layer cake
  background_mapper_->integrateDepth(depth_frame_background_, T_L_CD,
                                     depth_camera);
  foreground_mapper_->integrateDepth(depth_frame_foreground_, T_L_CD,
                                     depth_camera);
}

void MultiMapper::integrateDepth(
    const DepthImage& depth_frame,
    const std::vector<ImageBoundingBox>& detection_boxes,
    const Transform& T_L_CD, const Transform& T_CM_CD,
    const Camera& depth_camera, const Camera& detection_boxes_camera) {
  // Possibly reallocate the mask image.
  cleaned_semantic_mask_.resizeAsync(detection_boxes_camera.height(),
                                     detection_boxes_camera.width(),
                                     *cuda_stream_);

  // Compute segmentation mask out of the detections
  maskFromDetections(detection_boxes, depth_frame,
                     params_.segmentation_mask_mode_proximity_threshold,
                     &cleaned_semantic_mask_, *cuda_stream_);

  integrateDepth(depth_frame, cleaned_semantic_mask_, T_L_CD, T_CM_CD,
                 depth_camera, detection_boxes_camera);
}

void MultiMapper::integrateColor(const ColorImage& color_frame,
                                 const Transform& T_L_C, const Camera& camera) {
  // TODO(remos): For kDynamic we should split the image and only integrate
  // background pixels. As the dynamic mask is not a direct overlay of the
  // color image, this requires implementing a new splitImageOnGPU for color
  // images.
  background_mapper_->integrateColor(color_frame, T_L_C, camera);
}

void MultiMapper::integrateColor(const ColorImage& color_frame,
                                 const MonoImage& mask, const Transform& T_L_C,
                                 const Camera& camera) {
  CHECK(isHumanMapping(mapping_type_))
      << "Passing a mask to integrateColor is only valid for human "
         "mapping.";
  if (mapping_type_ == MappingType::kHumanWithStaticOccupancy) {
    // We do nothing because color integration is only implemented for
    // static tsdf.
    return;
  }

  // Remove small components (assumed to be noise) from the mask
  // We do this again incase the mask is not synced with the depth mask
  if (params_.remove_small_connected_components) {
    mask_preprocessor_.removeSmallConnectedComponents(
        mask, params_.connected_mask_component_size_threshold,
        &cleaned_semantic_mask_);
  } else {
    cleaned_semantic_mask_.copyFromAsync(mask, *cuda_stream_);
  }

  // Split into background and foreground color frame
  image_masker_.splitImageOnGPU(
      color_frame, cleaned_semantic_mask_, &color_frame_background_,
      &color_frame_foreground_, &foreground_color_overlay_);

  // Integrate the frames to the respective layer cake
  foreground_mapper_->integrateColor(color_frame_foreground_, T_L_C, camera);
  background_mapper_->integrateColor(color_frame_background_, T_L_C, camera);
}

void MultiMapper::updateEsdf() {
  std::optional<Plane> maybe_ground_plane = std::nullopt;
  if (params_.use_ground_plane_estimation) {
    // We compute the ground plane on the static map to reuse on the dynamic
    // map.
    const TsdfLayer& tsdf_layer = background_mapper_->tsdf_layer();
    maybe_ground_plane = ground_plane_estimator_.computeGroundPlane(tsdf_layer);
  }

  updateEsdfOfMapper(background_mapper_, maybe_ground_plane);
  if (foreground_mapper_->projective_layer_type() !=
      ProjectiveLayerType::kNone) {
    // Only update the foreground mapper in case we run dynamics or human
    // detection
    updateEsdfOfMapper(foreground_mapper_, maybe_ground_plane);
  }
}

void MultiMapper::integrateColor(const ColorImage& color_frame,
                                 const std::vector<ImageBoundingBox>&,
                                 const Transform& T_L_C, const Camera& camera) {
  // TODO(alexmillane, 2024.09.12): Right now we ignore the detection boxes and
  // integrate the whole color image into the background mapper. We can solve
  // this by implementing a version of maskFromDetections() which doesn't do
  // depth-based filtering.
  integrateColor(color_frame, T_L_C, camera);
}

void MultiMapper::updateMesh() {
  // At the moment we never have a mesh for the foreground mapper as it always
  // uses a occupancy layer.
  background_mapper_->updateMesh(UpdateFullLayer::kNo);
}

const DepthImage& MultiMapper::getLastDepthFrameBackground() {
  return depth_frame_background_;
}
const DepthImage& MultiMapper::getLastDepthFrameForeground() {
  return depth_frame_foreground_;
}
const ColorImage& MultiMapper::getLastColorFrameBackground() {
  return color_frame_background_;
}
const ColorImage& MultiMapper::getLastColorFrameForeground() {
  return color_frame_foreground_;
}
const ColorImage& MultiMapper::getLastDepthFrameMaskOverlay() {
  return foreground_depth_overlay_;
}
const ColorImage& MultiMapper::getLastColorFrameMaskOverlay() {
  return foreground_color_overlay_;
}
const ColorImage& MultiMapper::getLastDynamicFrameMaskOverlay() {
  return dynamic_detector_.getDynamicOverlayImage();
}
const Pointcloud& MultiMapper::getLastDynamicPointcloud() {
  return dynamic_detector_.getDynamicPointcloudDevice();
}

void MultiMapper::updateEsdfOfMapper(const std::shared_ptr<Mapper> mapper,
                                     std::optional<Plane> ground_plane) {
  switch (esdf_mode_) {
    case EsdfMode::kUnset:
      LOG(WARNING) << "ESDF mode not set. Doing nothing.";
      break;
    case EsdfMode::k3D:
      mapper->updateEsdf();
      break;
    case EsdfMode::k2D:
      mapper->updateEsdfSlice(UpdateFullLayer::kNo, ground_plane);
      break;
  }
}

parameters::ParameterTreeNode MultiMapper::getParameterTree(
    const std::string& name_remap) const {
  using parameters::ParameterTreeNode;
  const std::string name = (name_remap.empty()) ? "multi_mapper" : name_remap;
  return ParameterTreeNode(
      name, {ParameterTreeNode("connected_mask_component_size_threshold",
                               params_.connected_mask_component_size_threshold),
             background_mapper_->getParameterTree("background_mapper"),
             foreground_mapper_->getParameterTree("foreground_mapper"),
             image_masker_.getParameterTree(),
             ground_plane_estimator_.getParameterTree()});
}

std::string MultiMapper::getParametersAsString() const {
  return parameterTreeToString(getParameterTree());
}

}  // namespace nvblox
