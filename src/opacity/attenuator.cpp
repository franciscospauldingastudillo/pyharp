// harp
#include <utils/find_resource.hpp>

#include "absorber.hpp"

namespace harp {
//! get attenuation coefficient [1/m]
torch::Tensor AbsorberImpl::forward(torch::Tensor var_x) const {
  // shape of the original tensor
  auto var_shape = var_x.sizes().vec();
  var_shape.insert(var_shape.begin(), 1);
  auto out = torch::zeros(var_shape, var_x.options());
  return out;
}

torch::Tensor AbsorberImp::forward(torch::Tensor var_x) const {};

void Absorber::LoadOpacity(int bid) {
  auto app = Application::GetInstance();
  auto log = app->GetMonitor("opacity");

  if (opacity_filename_.empty()) return;

  std::string full_path = app->FindResource(opacity_filename_);
  log->Log("Load opacity from " + full_path);
  LoadCoefficient(full_path, bid);
}
}  // namespace harp
