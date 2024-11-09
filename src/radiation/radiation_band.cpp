// C/C++
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <vector>

// external
#include <yaml-cpp/yaml.h>

// athena
#include <athena/mesh/mesh.hpp>
#include <athena/outputs/outputs.hpp>
#include <athena/parameter_input.hpp>
#include <athena/utils/utils.hpp>

// canoe
#include <configure.hpp>

// climath
#include <climath/core.h>

// utils
#include <utils/extract_substring.hpp>
#include <utils/fileio.hpp>
#include <utils/ndarrays.hpp>

// opacity
#include <opacity/absorber.hpp>

// harp
#include "radiation.hpp"
#include "radiation_band.hpp"
#include "rt_solvers.hpp"

RadiationBandImpl::RadiationBandImpl(RadiationBandOptions const &options)
    : options(options) {
  reset();
}

RadiationBandImpl::reset() {
  auto my = rad[myname];

  spec = register_buffer("spec",
                         torch::tensor({options.nspec()}, torch::kFloat32));

  btau = register_buffer(
      "btau", torch::zeros({options.nc3(), options.nc2(), options.nc1()},
                           torch::kFloat32));

  tau = register_buffer(
      "tau", torch::zeros({options.nc2(), options.nc1(), options.nspec()},
                          torch::kFloat32));

  bssa = register_buffer(
      "bssa", torch::zeros({options.nc3(), options.nc2(), options.nc1()},
                           torch::kFloat32));

  ssa = register_buffer(
      "ssa", torch::zeros({options.nc2(), options.nc1(), options.nspec()},
                          torch::kFloat32));

  bpmom =
      register_buffer("bpmom", torch::zeros({options.nc3(), options.nc2(),
                                             options.nc1(), 1 + options.nstr()},
                                            torch::kFloat32));

  pmom = register_buffer(
      "pmom", torch::zeros({options.nc2(), options.nc1(), options.nspec(),
                            1 + options.nstr()},
                           torch::kFloat32));

  bflxup = register_buffer(
      "bflxup", torch::zeros({options.nc3(), options.nc2(), options.nc1() + 1},
                             torch::kFloat32));

  flxup = register_buffer(
      "flxup", torch::zeros({options.nc2(), options.nc1(), options.nspec()},
                            torch::kFloat32));

  bflxdn = register_buffer(
      "bflxdn", torch::zeros({options.nc3(), options.nc2(), options.nc1() + 1},
                             torch::kFloat32));

  flxdn = register_buffer(
      "flxdn", torch::zeros({options.nc2(), options.nc1(), options.nspec()},
                            torch::kFloat32));

  btoa = register_buffer(
      "btoa", torch::zeros({options.nc3(), options.nc2()}, torch::kFloat32));

  toa = register_buffer(
      "toa", torch::zeros({options.nc3(), options.nc2(), options.nspec()},
                          torch::kFloat32));

  auto str = options.outdirs();
  if (!str.empty()) {
    band.rayOutput = parse_radiation_directions(str);
    register_buffer("rayOutput", band.rayOutput);
  }

  // set absorbers
  for (auto name : options.absorbers()) {
    absorbers[name] = Absorber(options.absorber_options().at(name));
  }

  if (my["opacity"]) {
    if (!my["opacity"].IsSequence()) {
      throw RuntimeError("RadiationBand", "opacity must be a sequence");
    }

    auto names = my["opacity"].as<std::vector<std::string>>();
    absorbers_ = AbsorberFactory::CreateFrom(names, GetName(), rad);

    if (load_opacity) {
      for (auto &ab : absorbers_) {
        ab->LoadOpacity(RadiationBandsFactory::GetBandId(myname));
        // Correlated-k absorbers need to modify the spectral grid
        ab->ModifySpectralGrid(pgrid_->spec);
      }
    }
  }

  // set flags
  if (my["flags"]) {
    if (!my["flags"].IsSequence()) {
      throw RuntimeError("RadiationBand", "flags must be a sequence");
    }

    auto flag_strs = my["flags"].as<std::vector<std::string>>();
    for (auto flag : flag_strs) {
      SetFlag(RadiationHelper::parse_radiation_flags(flag));
    }
  }

  // set rt solver
  // rt_solver = CreateRTSolverFrom(my["rt-solver"].as<std::string>(), rad);
}

void RadiationBand::Resize(int nc1, int nc2, int nc3, int nstr,
                           MeshBlock const *pmb) {
  // allocate memory for spectral properties
  tem_.resize(nc1);
  temf_.resize(nc1 + 1);

  tau_.NewAthenaArray(pgrid_->spec.size(), nc1);
  tau_.ZeroClear();

  ssa_.NewAthenaArray(pgrid_->spec.size(), nc1);
  ssa_.ZeroClear();

  pmom_.NewAthenaArray(pgrid_->spec.size(), nc1, nstr + 1);
  pmom_.ZeroClear();

  // spectral grids properties
  toa_.NewAthenaArray(pgrid_->spec.size(), rayOutput_.size(), nc3, nc2);
  toa_.ZeroClear();

  flxup_.NewAthenaArray(pgrid_->spec.size(), nc3, nc2, nc1);
  flxup_.ZeroClear();

  flxdn_.NewAthenaArray(pgrid_->spec.size(), nc3, nc2, nc1);
  flxdn_.ZeroClear();

  // band properties
  btau.NewAthenaArray(nc3, nc2, nc1);
  bssa.NewAthenaArray(nc3, nc2, nc1);
  bpmom.NewAthenaArray(nstr + 1, nc3, nc2, nc1);

  btoa.NewAthenaArray(rayOutput_.size(), nc3, nc2);
  bflxup.NewAthenaArray(nc3, nc2, nc1 + 1);
  bflxdn.NewAthenaArray(nc3, nc2, nc1 + 1);

  // exchange buffer
  pexv = std::make_shared<LinearExchanger<Real, 2>>(GetName());

  int nlayers = GetNumLayers();
  int npmom = GetNumPhaseMoments();
  pexv->send_buffer[0].resize(temf_.size());
  pexv->send_buffer[1].resize(nlayers * (npmom + 3));

  pexv->Regroup(pmb, X1DIR);
  int nblocks = pexv->GetGroupSize();
  pexv->recv_buffer[0].resize(nblocks * pexv->send_buffer[0].size());
  pexv->recv_buffer[1].resize(nblocks * pexv->send_buffer[1].size());

  if (psolver_ != nullptr) {
    psolver_->Resize(nblocks * (nc1 - 2 * NGHOST), nstr);
  }
}
RadiationBand const *RadiationBand::CalBandFlux(MeshBlock const *pmb, int k,
                                                int j) {
  // reset flux of this column
  for (int i = pmb->is; i <= pmb->ie + 1; ++i) {
    bflxup(k, j, i) = 0.;
    bflxdn(k, j, i) = 0.;
  }

  psolver_->Prepare(pmb, k, j);
  psolver_->CalBandFlux(pmb, k, j);

  return this;
}

RadiationBand const *RadiationBand::CalBandRadiance(MeshBlock const *pmb, int k,
                                                    int j) {
  // reset radiance of this column
  for (int n = 0; n < GetNumOutgoingRays(); ++n) {
    btoa(n, k, j) = 0.;
  }

  psolver_->Prepare(pmb, k, j);
  psolver_->CalBandRadiance(pmb, k, j);

  return this;
}

void RadiationBand::WriteAsciiHeader(OutputParameters const *pout) const {
  if (!TestFlag(RadiationFlags::WriteBinRadiance)) return;

  char fname[80], number[6];
  snprintf(number, sizeof(number), "%05d", pout->file_number);
  snprintf(fname, sizeof(fname), "%s.radiance.%s.txt", GetName().c_str(),
           number);
  FILE *pfile = fopen(fname, "w");

  fprintf(pfile, "# Bin Radiances of Band %s: %.3g - %.3g\n", GetName().c_str(),
          wrange_.first, wrange_.second);
  fprintf(pfile, "# Ray output size: %lu\n", rayOutput_.size());

  fprintf(pfile, "# Polar angles: ");
  for (auto &ray : rayOutput_) {
    fprintf(pfile, "%.3f", rad2deg(acos(ray.mu)));
  }
  fprintf(pfile, "\n");

  fprintf(pfile, "# Azimuthal angles: ");
  for (auto &ray : rayOutput_) {
    fprintf(pfile, "%.3f", rad2deg(ray.phi));
  }
  fprintf(pfile, "\n");

  fprintf(pfile, "#%12s%12s", "wave1", "wave2");
  for (size_t j = 0; j < rayOutput_.size(); ++j) {
    fprintf(pfile, "%12s%lu", "Radiance", j + 1);
  }
  fprintf(pfile, "%12s\n", "weight");

  fclose(pfile);
}

void RadiationBand::WriteAsciiData(OutputParameters const *pout) const {
  if (!TestFlag(RadiationFlags::WriteBinRadiance)) return;

  char fname[80], number[6];
  snprintf(number, sizeof(number), "%05d", pout->file_number);
  snprintf(fname, sizeof(fname), "%s.radiance.%s.txt", GetName().c_str(),
           number);
  FILE *pfile = fopen(fname, "w");

  for (size_t i = 0; i < pgrid_->spec.size(); ++i) {
    fprintf(pfile, "%13.3g%12.3g", pgrid_->spec[i].wav1, pgrid_->spec[i].wav2);
    for (size_t j = 0; j < rayOutput_.size(); ++j) {
      fprintf(pfile, "%12.3f", toa_(i, j));
    }
    if (TestFlag(RadiationFlags::Normalize) &&
        (wrange_.first != wrange_.second)) {
      fprintf(pfile, "%12.3g\n",
              pgrid_->spec[i].wght / (wrange_.second - wrange_.first));
    } else {
      fprintf(pfile, "%12.3g\n", pgrid_->spec[i].wght);
    }
  }

  fclose(pfile);
}

std::string RadiationBand::ToString() const {
  std::stringstream ss;
  ss << "RadiationBand: " << GetName() << std::endl;
  ss << "Absorbers: ";
  for (auto &ab : absorbers_) {
    ss << ab->GetName() << ", ";
  }
  ss << std::endl << "RT-Solver: " << psolver_->GetName();
  return ss.str();
}

std::shared_ptr<RadiationBand::RTSolver> RadiationBand::CreateRTSolverFrom(
    std::string const &rt_name, YAML::Node const &rad) {
  std::shared_ptr<RTSolver> psolver;

  if (rt_name == "Lambert") {
    psolver = std::make_shared<RTSolverLambert>(this, rad);
#ifdef RT_DISORT
  } else if (rt_name == "Disort") {
    psolver = std::make_shared<RTSolverDisort>(this, rad);
#endif  // RT_DISORT
  } else {
    throw NotFoundError("RadiationBand", rt_name);
  }

  return psolver;
}

void RadiationBand::set_temperature_level(torch::Tensor hydro_x) {
  tem_ = hydro_w[0];

  // set temperature at cell interface
  int il = NGHOST, iu = ac.size() - 1 - NGHOST;
  temf_[il] = (3. * tem_[il] - tem_[il + 1]) / 2.;
  temf_[il + 1] = (tem_[il] + tem_[il + 1]) / 2.;
  for (int i = il + 2; i <= iu - 1; ++i)
    temf_[i] = interp_cp4(tem_[i - 2], tem_[i - 1], tem_[i], tem_[i + 1]);
  temf_[iu] = (tem_[iu] + tem_[iu - 1]) / 2.;
  // temf_[iu + 1] = (3. * tem_[iu] - tem_[iu - 1]) / 2.;
  temf_[iu + 1] = tem_[iu];  // isothermal top boundary

  for (int i = 0; i < il; ++i) temf_[i] = tem_[il];
  for (int i = iu + 2; i < ac.size(); ++i) temf_[i] = tem_[iu + 1];

  bool error = false;
  for (int i = 0; i < ac.size(); ++i) {
    if (temf_[i] < 0.) {
      temf_[i] = tem_[i];
      // error = true;
    }
  }
  for (int i = il; i <= iu; ++i) {
    if (tem_[i] < 0.) error = true;
  }
  if (error) {
    for (int i = il; i <= iu; ++i) {
      std::cout << "--- temf[" << i << "] = " << temf_[i] << std::endl;
      std::cout << "tem[" << i << "] = " << tem_[i] << std::endl;
    }
    std::cout << "--- temf[" << iu + 1 << "] = " << temf_[iu + 1] << std::endl;
    throw std::runtime_error("Negative temperature at cell interface");
  }
}
