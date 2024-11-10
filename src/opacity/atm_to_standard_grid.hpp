#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/any.h>

namespace harp {
class AtmToStandardGrid : public torch::nn::Cloneable<AtmToStandardGrid> {
 public:
  //! temperature
  torch::Tensor taxis;

  //! composition
  torch::Tensor xaxis;

  //! reference atmosphere
  //! (X, levels)
  torch::Tensor refatm;

  //! constructor to initialize the layer
  AtmToStandardGrid() = default;

  //! standard grid in [-1, 1] x [-1, 1] x [-1, 1]
  torch::Tensor forward(torch::Tensor var_x, int ix);
};
}  // namespace harp
