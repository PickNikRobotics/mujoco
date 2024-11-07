// Copyright 2022 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <cstdint>
#include <utility>
#include <chrono>

#include <mujoco/mjplugin.h>
#include <mujoco/mjtnum.h>
#include <mujoco/mjvisualize.h>
#include <mujoco/mujoco.h>
#include "sdf.h"
#include "point_cloud.h"

namespace mujoco::plugin::sdf {
namespace {
auto last_flip_time = std::chrono::system_clock::now();
auto deviation = 0.5;
std::vector<std::array<mjtNum, 3>> getPoints() {
  auto now = std::chrono::system_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_flip_time);

  if (elapsed.count() >= 2) {
    last_flip_time = now;
    deviation = (deviation == 0.5) ? 0.25 : 0.5;
  }

  std::vector<std::array<mjtNum, 3>> ret_points = {{0.0, 0.0, 0.0}, {deviation, 0.0, 0.0}, {0.0, deviation, 0.0}};
  return ret_points;
}

static mjtNum distance(const mjtNum p[3], const mjtNum attributes[PointCloudAttribute::nattribute]) {
  std::vector<mjtNum> distances;
  for (const std::array<mjtNum, 3> point: getPoints()) {
    // Euclidean distance - radius.
    distances.push_back(
            std::sqrt(std::pow(p[0] - point[0], 2) + std::pow(p[1] - point[1], 2) + std::pow(p[2] - point[2], 2)) -
            attributes[0]);
  }
  return *std::min_element(distances.begin(), distances.end());
}

}  // namespace

// factory function
std::optional<PointCloud> PointCloud::Create(const mjModel* m, mjData* d, int instance) {
  if (CheckAttr("radius", m, instance)) {
      return PointCloud(m, d, instance);
  } else {
      mju_warning("Invalid parameter specification in PointCloud plugin");
      return std::nullopt;
  }
}

// plugin constructor
PointCloud::PointCloud(const mjModel* m, mjData* d, int instance) {
  SdfDefault<PointCloudAttribute> defattribute;

  for (int i=0; i < PointCloudAttribute::nattribute; i++) {
      attribute[i] = defattribute.GetDefault(
          PointCloudAttribute::names[i],
          mj_getPluginConfig(m, instance, PointCloudAttribute::names[i]));
  }
}

// plugin computation
void PointCloud::Compute(const mjModel* m, mjData* d, int instance) {
  visualizer_.Next();
}

// plugin reset
void PointCloud::Reset() {
  visualizer_.Reset();
}

// plugin visualization
void PointCloud::Visualize(const mjModel* m, mjData* d, const mjvOption* opt,
                     mjvScene* scn, int instance) {
  for (const std::array<mjtNum, 3> point: getPoints()) {
    mjvGeom *geom = scn->geoms + scn->ngeom;
    mjtNum pos[3], size[3];
    pos[0] = point[0];
    pos[1] = point[1];
    pos[2] = point[2];
    size[0] = attribute[0];
    size[1] = attribute[0];
    size[2] = attribute[0];
    float rgba[4];
    rgba[1] = 1.0;
    rgba[3] = 1.0;
    scn->ngeom++;
    mjv_initGeom(geom, mjGEOM_SPHERE, size, pos, NULL, rgba);
  }
  visualizer_.Visualize(m, d, opt, scn, instance);
}

// sdf
mjtNum PointCloud::Distance(const mjtNum point[3]) const {
  return distance(point, attribute);
}

// gradient of sdf
void PointCloud::Gradient(mjtNum grad[3], const mjtNum point[3]) const {
  mjtNum eps = 1e-8;
  mjtNum dist0 = distance(point, attribute);

  mjtNum pointX[3] = {point[0]+eps, point[1], point[2]};
  mjtNum distX = distance(pointX, attribute);
  mjtNum pointY[3] = {point[0], point[1]+eps, point[2]};
  mjtNum distY = distance(pointY, attribute);
  mjtNum pointZ[3] = {point[0], point[1], point[2]+eps};
  mjtNum distZ = distance(pointZ, attribute);

  grad[0] = (distX - dist0) / eps;
  grad[1] = (distY - dist0) / eps;
  grad[2] = (distZ - dist0) / eps;
}

// plugin registration
void PointCloud::RegisterPlugin() {
  mju_warning("Attempting to register PointCloud plugin");
  mjpPlugin plugin;
  mjp_defaultPlugin(&plugin);

  plugin.name = "mujoco.sdf.point_cloud";
  plugin.capabilityflags |= mjPLUGIN_SDF;

  plugin.nattribute = PointCloudAttribute::nattribute;
  plugin.attributes = PointCloudAttribute::names;
  plugin.nstate = +[](const mjModel* m, int instance) { return 0; };

  plugin.init = +[](const mjModel* m, mjData* d, int instance) {
    auto sdf_or_null = PointCloud::Create(m, d, instance);
    if (!sdf_or_null.has_value()) {
      return -1;
    }
    d->plugin_data[instance] = reinterpret_cast<uintptr_t>(
        new PointCloud(std::move(*sdf_or_null)));
    return 0;
  };
  plugin.destroy = +[](mjData* d, int instance) {
    delete reinterpret_cast<PointCloud*>(d->plugin_data[instance]);
    d->plugin_data[instance] = 0;
  };
  plugin.reset = +[](const mjModel* m, double* plugin_state, void* plugin_data,
                     int instance) {
    // Try doing nothing like torus.cc
    auto sdf = reinterpret_cast<PointCloud*>(plugin_data);
    sdf->Reset();
  };
  plugin.visualize = +[](const mjModel* m, mjData* d, const mjvOption* opt,
                         mjvScene* scn, int instance) {
    auto* sdf = reinterpret_cast<PointCloud*>(d->plugin_data[instance]);
    sdf->Visualize(m, d, opt, scn, instance);
  };
  plugin.compute =
      +[](const mjModel* m, mjData* d, int instance, int capability_bit) {
      // Try doing nothing like torus.cc
        auto* sdf = reinterpret_cast<PointCloud*>(d->plugin_data[instance]);
        sdf->Compute(m, d, instance);
      };
  plugin.sdf_distance =
      +[](const mjtNum point[3], const mjData* d, int instance) {
        auto* sdf = reinterpret_cast<PointCloud*>(d->plugin_data[instance]);
        return sdf->Distance(point);
      };
  plugin.sdf_gradient = +[](mjtNum gradient[3], const mjtNum point[3],
                        const mjData* d, int instance) {
    auto* sdf = reinterpret_cast<PointCloud*>(d->plugin_data[instance]);
    sdf->visualizer_.AddPoint(point);
    sdf->Gradient(gradient, point);
  };
  plugin.sdf_staticdistance =
      +[](const mjtNum point[3], const mjtNum* attributes) {
        return distance(point, attributes);
      };
  plugin.sdf_aabb =
      +[](mjtNum aabb[6], const mjtNum* attributes) {
        aabb[0] = aabb[1] = aabb[2] = 0;
        // Need to consider how far away the farthest point is in the point cloud (deviation)
        // I think this function may only be called on startup for visualization (rendering marching cubes mesh of the
        // object) so it may not matter for it to be dynamic.
        aabb[3] = aabb[4] = aabb[5] = deviation + 2 * attributes[0];
  };
  plugin.sdf_attribute =
      +[](mjtNum attribute[], const char* name[], const char* value[]) {
        SdfDefault<PointCloudAttribute> defattribute;
        defattribute.GetDefaults(attribute, name, value);
      };

  mjp_registerPlugin(&plugin);
}

}  // namespace mujoco::plugin::sdf
