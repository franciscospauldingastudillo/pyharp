// cnaoe
#include <configure.hpp>

// athena
#include <athena/coordinates/coordinates.hpp>
#include <athena/hydro/hydro.hpp>
#include <athena/mesh/mesh.hpp>
#include <athena/reconstruct/interpolation.hpp>
#include <athena/scalars/scalars.hpp>

// canoe
#include <air_parcel.hpp>
#include <constants.hpp>
#include <impl.hpp>

// snap
#include <snap/stride_iterator.hpp>
#include <snap/thermodynamics/thermodynamics.hpp>

// opacity
#include <opacity/absorber.hpp>

// harp
#include "radiation.hpp"
#include "radiation_band.hpp"

// setting optical properties
torch::Tensor RadiationBand::forward(torch::Tensor x1f, torch::Tensor ftoa,
                                     torch::Tensor var_x) {
  // set tau, ssalb, pmom, etc...
  tau_.ZeroClear();
  ssa_.ZeroClear();
  pmom_.ZeroClear();

  for (auto& a : absorbers_) {
    auto [kcoeff, ssalb, moments] = a.forward(spec, hydro_x, scalar_x);
    tau_ += kcoeff;
    ssa_ += ssalb * kcoeff;
    pmom_ += moments * ssalb * kcoeff;
  }

  // absorption coefficients -> optical thickness
  if (tau_(m, i) > 1e-6 && ssa_(m, i) > 1e-6) {  // has scattering
    pmom_ /= ssa_;
    ssa_ /= tau_;
  } else {
    ssa_ = 0.;
    pmom_(m, i, 0) = 1.;
  }

  tau_ *= x1f[i + 1] - x1f[i];

  // aggregated band properties
  for (int i = 0; i < ac.size(); ++i) {
    btau(k, j, i) = 0;
    bssa(k, j, i) = 0;
    for (int p = 0; p <= npmom; ++p) bpmom(p, k, j, i) = 0.;

    for (int m = 0; m < nspec; ++m) {
      btau(k, j, i) += tau_(m, i);
      bssa(k, j, i) += ssa_(m, i) * tau_(m, i);
      for (int p = 0; p <= npmom; ++p)
        bpmom(p, k, j, i) += pmom_(m, i, p) * ssa_(m, i);
    }

    for (int p = 0; p <= npmom; ++p) bpmom(p, k, j, i) /= bssa(k, j, i);
    bssa(k, j, i) /= btau(k, j, i);
    btau(k, j, i) /= nspec;
  }

  return solver.forward(ftoa, tau_, ssa_, pmom_);
}
