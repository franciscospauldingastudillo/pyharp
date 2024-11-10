#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/any.h>

// harp
#include <configure.h>

#include "add_arg.h"

namespace harp {
enum index {
  ITM = 0,  // temperature
}

struct AttenuatorOptions {
  AttenuatorOptions() = default;

  ADD_ARG(int, npmom) = 1;
  ADD_ARG(int, nspec) = 1;
  ADD_ARG(int, ncomp) = 1;
  ADD_ARG(int, npres) = 1;
  ADD_ARG(int, ntemp) = 1;
  ADD_ARG(bool, spectral_bin) = false;
  ADD_ARG(std::vector<int>, species_id) = {0};

  ADD_ARG(std::string, name) = "";
  ADD_ARG(std::string, band_name) = "";
  ADD_ARG(std::string, model_name) = "";
  ADD_ARG(std::string, opacity_file) = "";
};

//! \brief base class of all absorbers
class AttenuatorImpl {
 public:
  //! options with which this `Attenuator` was constructed
  AttenuatorOptions options;

  //! constructor to initialize the layer
  AttenuatorImpl() = default;
  virtual ~AttenuatorImpl() {}

  //! main forward function
  virtual std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(
      torch::Tensor var_x);

  //! get attenuation coefficient [1/m]
  virtual torch::Tensor attenuation(torch::Tensor var_x) const;

  //! get single scattering albedo [1]
  virtual torch::Tensor single_scattering_albedo(torch::Tensor var_x) const;

  //! get phase function moments [1]
  virtual torch::Tensor phase_moments(torch::Tensor var_x) const;
};

using Attenuator = std::shared_ptr<AttenuatorImpl>;

class InterpAttenuatorImpl : public AttenuatorImpl {
 public:
  //! constructor to initialize the layer
  explicit InterpAttenuatorImpl(AttenuatorOptions const& options_)
      : options(options_) {}
  void load() override;
  torch::Tensor scaled_interp_xpt(torch::Tensor var_x) const override;

  torch::Tensor InterpAttenuator::attenuation(
      torch::Tensor var_x) const override;

 protected:
  //! reference atmosphere
  //! (X, levels)
  torch::Tensor refatm_;

  //! log pressure
  //! (levels,)
  torch::Tensor logp_;

  //! temperature
  //! (levels,)
  torch::Tensor temp_;

  //! composition
  //! (levels,)
  torch::Tensor comp_;

  //! absorption x-section
  //! (specs, comps, levels, temps)
  torch::Tensor kcross_;

  //! single scattering albedo
  //! (specs, comps, levels, temps)
  torch::Tensor kssa_;

  //! phase function moments
  //! (pmoms, specs, comps, levels, temps)
  torch::Tensor kpmom_;
};

class AbsorberRFMImpl;
}  // namespace harp
