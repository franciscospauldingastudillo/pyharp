// harp
#include <utils/find_resource.hpp>
#include "absorber.hpp"

namespace harp {
//! main forward function
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> 
AbsorberImpl::forward(
    torch::Tensor spec, torch::Tensor var_x) {
  return {attenuation(spec, var_x), single_scattering_albedo(spec, var_x),
          phase_moments(spec, var_x)};
}

//! get attenuation coefficient [1/m]
torch::Tensor AbsorberImpl::attenuation(torch::Tensor var_x) const {
  // shape of the original tensor
  auto var_shape = var_x.sizes().vec();
  var_shape.push_back(1);
  auto out = torch::zeros(var_shape, var_x.options());
  return out;
}

//! get single scattering albedo [1]
torch::Tensor AbsorberImpl::single_scattering_albedo(torch::Tensor var_x) const {
  // shape of the original tensor
  auto var_shape = var_x.sizes().vec();
  var_shape.push_back(1);
  auto out = torch::zeros(var_shape, var_x.options());
  return out;
}

//! get phase function moments [1]
torch::Tensor AbsorberImpl::phase_moments(torch::Tensor var_x) const {
  auto var_shape = var_x.sizes().vec();
  var_shape.push_back(1);
  var_shape.push_back(options.npmom());
  auto out = torch::zeros(var_shape, var_x.options());
  return out;
}

torch::Tensor AbsorberImp::scaled_interp_xpt(torch::Tensor var_x) const

void Absorber::LoadOpacity(int bid) {
  auto app = Application::GetInstance();
  auto log = app->GetMonitor("opacity");

  if (opacity_filename_.empty()) return;

  std::string full_path = app->FindResource(opacity_filename_);
  log->Log("Load opacity from " + full_path);
  LoadCoefficient(full_path, bid);
}
}  // namespace harp
