#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/any.h>

// harp
#include <add_arg.h>
#include <configure.h>
#include <index.h>

#include "atm_to_standard_grid.hpp"

namespace harp {
struct AttenuatorOptions {
  AttenuatorOptions() = default;

  ADD_ARG(int, npmom) = 0;
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
  AttenuatorImpl(AttenuatorOptions const& options_) : options(options_) {}
  virtual ~AttenuatorImpl() {}

  //! main forward function
  //! \param var_x input atmospheric variables of size (X, levels)
  //!              where X is the number of variables
  //!              and levels is the number of vertical levels
  //! \return tensor of size (batch, specs, comps, levels, temps)
  //!         where the first index in batch dimension is the attenuation
  //!         coefficient and the second index is the single scattering albedo
  //!         and the rest of the indices are the phase function moments
  //!         (excluding the zeroth moment). The units of the attenuation
  //!         coefficient are [1/m] The single scattering albedo is
  //!         dimensionless
  virtual torch::Tensor forward(torch::Tensor var_x);
};

using Attenuator = std::shared_ptr<AttenuatorImpl>;

class AbsorberRFMImpl : public AttenuatorImpl,
                        public torch::nn::Cloneable<AbsorberRFMImpl> {
 public:
  //! extinction x-section + single scattering albedo + phase function moments
  //! (batch, specs, comps, levels, temps)
  torch::Tensor kdata;

  //! scale the atmospheric variables to the standard grid
  AtmToStandardGrid scale_grid;

  //! constructor to initialize the layer
  explicit AbsorberRFMImpl(AttenuatorOptions const& options_)
      : AttenuatorImpl(options_) {}
  virtual void load();

  torch::Tensor forward(torch::Tensor var_x) override;
};
TORCH_MODULE(AbsorberRFM);

}  // namespace harp
