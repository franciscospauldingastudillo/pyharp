// C/C++ headers
#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>

// external
#include <yaml-cpp/yaml.h>

// harp
#include "radiation.hpp"
#include "radiation_utils.hpp"
#include "rt_solvers.hpp"
#include "vectorize.hpp"

namespace harp {
void RadiationOptions::set_flags(std::string const& str) {
  std::vector<std::string> dstr = Vectorize<std::string>(str.c_str(), " ,");

  for (int i = 0; i < dstr.size(); ++i) {
    if (dstr[i] == "time_dependent") {
      time_dependent(true);
    } else if (dstr[i] == "broad_band") {
      broad_band(true);
    } else if (dstr[i] == "stellar_beam") {
      stellar_beam(true);
    } else if (dstr[i] == "thermal_emission") {
      thermal_emission(true);
    } else if (dstr[i] == "normalize") {
      normalize(true);
    } else if (dstr[i] == "write_bin_radiance") {
      write_bin_radiance(true);
    } else {
      std::stringstream msg;
      msg << "flag: '" << dstr[i] << "' unrecognized" << std::endl;
      throw std::runtime_error("parse_radiation_flags::" + msg.str());
    }
  }
}

RadiationImpl::RadiationImpl(RadiationOptions const& options_)
    : options(options_) {
  reset();
}

void RadiationImpl::reset() {
  // radiation bands
  bands_ = RadiationBandsFactory::CreateFrom(pin, input_key);

  for (auto name : options.bands()) {
    bands[name] = RadiationBand(options.band_options().at(name));
    register_module(name, bands[name]);
  }

  // incoming radiation direction (mu, phi) in degrees
  rayInput = parse_radiation_directions(options.indirs());
  register_buffer("rayInput", rayInput);

  // set default outgoing radiation directions
  if (!options.outdirs().empty()) {
    for (auto& [name, band] : bands) {
      // outgoing radiation direction (mu,phi) in degrees
      if (band.rayOutput.empty()) {
        auto str = options.outdirs();
        band.rayOutput = parse_radiation_directions(str);
      }
    }
  }
}

torch::Tensor RadiationImpl::forward(torch::Tensor ftoa, torch::Tensor hydro_x,
                                     torch::Tensor scalar_x) {
  torch::Tensor out = torch::zeros_like(ftoa);

  if (options.flux_flag()) {
    for (auto& [name, band] : bands) {
      out += band.forward(x1f, ftoa * hydro_x, hydro_x, scalar_x);
    }
  } else {
    for (auto& [name, band] : bands) {
      band.forward(x1f, ftoa, hydro_x, scalar_x);
    }
  }

  return out;
}

}  // namespace harp
