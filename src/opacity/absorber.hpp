#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/any.h>

// harp
#include "add_arg.h"
#include "configure.h"

namespace harp {
enum index {
  ITM = 0,  // temperature
  IPR = 1,  // pressure
}

struct AbsorberOptions {
  AbsorberOptions() = default;

  ADD_ARG(std::string, name) = "B1";
  ADD_ARG(std::string, model_name) = "";
  ADD_ARG(std::string, opacity_file) = "";
};

//! \brief base class of all absorbers
class AbsorberImpl {
 public:
  //! options with which this `Absorber` was constructed
  AbsorberOptions options;

  virtual ~AbsorberImpl() {}

 public:  // member functions
  //! Load from file
  virtual void load() { opacity_loaded_ = true; }
  bool opacity_loaded() const { return opacity_loaded_; }

  virtual std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(
      torch::Tensor spec, torch::Tensor var_x) {
    return {attenuation(spec, var_x), single_scattering_albedo(spec, var_x),
            phase_moments(spec, var_x)};
  }

  //! Get attenuation coefficient [1/m]
  virtual torch::Tensor attenuation(torch::Tensor spec,
                                    torch::Tensor var_x) const {
    return 0.;
  }

  //! Get single scattering albedo [1]
  virtual torch::Tensor single_scattering_albedo(torch::Tensor spec,
                                                 torch::Tensor var_x) const {
    return 0.;
  }

  //! Get phase function moments [1]
  virtual torch::Tensor phase_moments(torch::Tensor spec,
                                      torch::Tensor var_x) const {
    return 0.;
  }

 protected:
  //! Disable constructor
  AbsorberImpl() = default;

  bool opacity_loaded_ = false;
};

using Absorber = std::shared_ptr<AbsorberImpl>;

class HitranAbsorberImpl : public torch::nn::Cloneable<HitranAbsorberImpl>,
                           public AbsorberImpl {
 public:
  //! Constructor to initialize the layer
  explicit HitranAbsorberImpl() = default;
  void reset() override {}

  void load() override;
  torch::Tensor attenuation(torch::Tensor spec, torch::Tensor hydro_x,
                            torch::Tensor scalar_x) const override;

  virtual std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(
      torch::Tensor spec, torch::Tensor var_x) override;

 protected:
  size_t len_[3];        /**< length of interpolation axis */
  torch::Tensor coords_; /**< interpolation axis */
  torch::Tensor kcoeff_; /**< absorption coefficient */
  torch::Tensor refatm_; /**< reference atmosphere */

  torch::Tensor get_reftemp_(torch::Tensor pres) const;
};

}  // namespace harp
