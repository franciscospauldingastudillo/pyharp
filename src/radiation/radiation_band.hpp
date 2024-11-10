#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/any.h>

// harp
#include "absorber.hpp"
#include "add_arg.h"
#include "configure.h"

namespace harp {
struct RadiationBandOptions {
  ADD_ARG(std::string, name) = "B1";
  ADD_ARG(std::string, outdirs) = "";
  ADD_ARG(std::vector<std::string>, absorbers) = {};
  ADD_ARG(std::string, solver) = "lambert";
  ADD_ARG(std::map<std::string, float>, parameters) = {};
  ADD_ARG(std::map<std::string, AbsorberOptions>, absorber_options) = {};
  // ADD_ARG(SolverOptions, solver_options) = {};

  ADD_ARG(int, nstr) = 1;
  ADD_ARG(int, nspec) = 1;
  ADD_ARG(int, nc1) = 1;
  ADD_ARG(int, nc2) = 1;
  ADD_ARG(int, nc3) = 1;

  ADD_ARG(float, wmin) = 0.0;
  ADD_ARG(float, wmax) = 1.0;
};

class RadiationBandImpl : public torch::nn::Cloneable<RadiationBandImpl> {
 public:  // public access data
  //! options with which this `RadiationBandImpl` was constructed
  RadiationOptions options;

  //! radiative transfer solver
  RTSolver rt_solver;

  //! all absorbers
  std::map<std::string, Absorber> attenuators;

  //! spectral grid weights (nspec)
  torch::Tensor weight;

  //! outgoing rays
  //! (nout, 2)
  tourch::Tensor rayOutput;

  //! band/bin optical data, (tau + ssa + pmom, C, ..., nlayer)
  torch::Tensor opt;

  //! Constructor to initialize the layers
  RadiationBandImpl() = default;
  explicit RadiationBandImpl(RadiationBandOptions const &options_);
  void reset() override;

  //! \brief Calculate the radiance/radiative flux
  torch::Tensor forward(torch::Tensor x1f, torch::Tensor ftoa,
                        torch::Tensor var_x);

 protected:
  void set_temperature_level_(torch::Tensor hydro_x);
};
TORCH_MODULE(RadiationBandImpl);

class RadiationBandsFactory {
 public:
  static RadiationBandContainer CreateFrom(ParameterInput *pin,
                                           std::string key);
  static RadiationBandContainer CreateFrom(std::string filename);

  static int GetBandId(std::string const &bname) {
    if (band_id_.find(bname) == band_id_.end()) {
      return -1;
    }
    return band_id_.at(bname);
  }

 protected:
  static std::map<std::string, int> band_id_;
  static int next_band_id_;
};

}  // namespace harp
