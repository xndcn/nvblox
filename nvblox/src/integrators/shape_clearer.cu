/*
Copyright 2024 NVIDIA CORPORATION

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
#include "nvblox/integrators/internal/cuda/impl/shape_clearer_impl.cuh"
#include "nvblox/integrators/shape_clearer.h"

namespace nvblox {

// Compile Specializations.
template class ShapeClearer<TsdfLayer>;
template class ShapeClearer<OccupancyLayer>;
template class ShapeClearer<ColorLayer>;

}  // namespace nvblox
