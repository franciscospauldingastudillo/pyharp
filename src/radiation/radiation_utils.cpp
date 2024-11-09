#include "radiation_utils.hpp"

#include "deg2rad.hpp"
#include "vectorize.hpp"

namespace harp {
bool real_close(float num1, float num2, float tolerance) {
  return std::fabs(num1 - num2) <= tolerance;
}

std::pair<std::vector<float>, std::vector<float>> get_direction_grids(
    std::vector<Direction> const &dirs) {
  std::vector<float> uphi;
  std::vector<float> umu;

  for (auto &dir : dirs) {
    // find phi
    bool found = false;
    for (auto &phi : uphi)
      if (real_close(phi, dir.phi, 1.e-3)) {
        found = true;
        break;
      }
    if (!found) uphi.push_back(dir.phi);

    // find mu
    found = false;
    for (auto &mu : umu)
      if (real_close(mu, dir.mu, 1.e-3)) {
        found = true;
        break;
      }
    if (!found) umu.push_back(dir.mu);
  }

  std::sort(uphi.begin(), uphi.end());
  std::sort(umu.begin(), umu.end());

  return std::make_pair(uphi, umu);
}

torch::Tensor parse_radiation_direction(std::string const &str) {
  float mu = 0.;
  float phi = 0.;

  sscanf(str.data(), "(%lf,%lf)", &mu, &phi);
  mu = cos(deg2rad(mu));
  phi = deg2rad(phi);

  return torch::tensor({mu, phi}, torch::kFloat32);
}

torch::Tensor parse_radiation_directions(std::string const &str) {
  std::vector<std::string> dstr = Vectorize<std::string>(str.c_str());
  int nray = dstr.size();

  torch::Tensor ray = torch::zeros({nray, 2}, torch::kFloat32);

  for (int i = 0; i < nray; ++i) {
    ray[i] = parse_radiation_direction(dstr[i]);
  }

  return ray;
}

torch::Tensor get_phase_moment(std::string const &phase_func, int npmom,
                               float gg1, float gg2, float ff) {
  torch::Tensor pmom = torch::zeros({1 + npmom}, torch::kFloat32);

  if (phase_func == "henyey_greenstein") {
    if (gg1 <= -1. || gg1 >= 1.) {
      throw std::runtime_error("get_phase_moment::bad input variable gg1");
    }
    for (int k = 1; k <= npmom; k++) {
      pmom[k] = pow(gg1, (float)k);
    }
  } else if (phase_func == "double_henyey_greenstein") {
    if (gg1 <= -1. || gg1 >= 1. || gg2 <= -1. || gg2 >= 1.) {
      throw std::runtime_error(
          "get_phase_moment::bad input variable gg1 or gg2");
    }

    for (int k = 1; k <= npmom; k++) {
      pmom[k] = ff * pow(gg1, (float)k) + (1. - ff) * pow(gg2, (float)k);
    }
  } else if (phase_func == "rayleigh") {
    if (npmom < 2) {
      throw std::runtime_error("get_phase_moment::npmom < 2");
    }
    pmom[2] = 0.1;
  } else {
    throw std::runtime_error("get_phase_moment::unknown phase function");
  }

  return pmom;
}

std::string parse_unit_with_default(YAML::Node const &my) {
  std::string units = my["units"] ? my["units"].as<std::string>() : "cm-1";

  if (units == "cm-1") {
    return "wavenumber";
  } else if (units == "um" || units == "nm" || units == "A") {
    return "wavelength";
  } else if (units == "GHz") {
    return "frequency";
  } else {
    throw std::runtime_error(
        "parse_unit_with_default::unknown spectral unit type");
  }
}

std::pair<float, float> parse_wave_range(YAML::Node const &my) {
  auto unit = parse_unit_with_default(my);

  char str[80];
  snprintf(str, sizeof(str), "%s-%s", unit.c_str(), "range");

  if (!my[str]) {
    throw std::runtime_error("parse_wave_range::" + str + " not found");
  }

  /// wavenumber-range, wavelength-range, frequency-range, etc
  float wmin = my[str][0].as<float>();
  float wmax = my[str][1].as<float>();
  if (wmin > wmax) {
    throw std::runtime_error("parse_wave_range::wmin > wmax");
  }

  return std::make_pair(wmin, wmax);
}

}  // namespace harp
