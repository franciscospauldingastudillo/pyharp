// harp
#include <constants.hpp>
#include "attenuator.hpp"

// netcdf
#ifdef NETCDFOUTPUT
extern "C" {
#include <netcdf.h>
}
#endif

namespace harp {
InterpAttenuatorImpl(AttenuatorOptions const& options_) : options(options_) {
  reset();
}

void InterpAttenuatorImpl::reset() {
  refatm_ = register_buffer("refatm", torch::zeros({1 + optioins.ncomp(), options.nlevel()}, torch::kFloat));

  logp_ = register_buffer("logp", torch::zeros({options.nlevel()}, torch::kFloat));

  temp_ = register_buffer("temp", torch::zeros({options.ntemp()}, torch::kFloat));

  comp_ = register_buffer("comp", torch::zeros({options.ncomp()}, torch::kFloat));

  kcross_ = register_buffer("kcross", torch::zeros({options.nspec(), options.ncomp(), options.nlevel(), options.ntemp()}, torch::kFloat));
  kssa_ = register_buffer("kssa", torch::zeros({options.nspec(), options.ncomp(), options.nlevel(), options.ntemp()}, torch::kFloat));
  kpmom_ = register_buffer("kpmom", torch::zeros({options.npmom(), options.nspec(), options.ncomp(), options.nlevel(), options.ntemp()}, torch::kFloat));

  load();
}

torch::Tensor InterpAttenuatorImpl::scaled_interp_xpt(torch::Tensor var_x) const {
  // log pressure
  auto log_refp = refatm_[index::IPR];
  auto logp = pres.log().flatten();

  auto log_refp_min = log_refp.min();
  auto log_refp_max = log_refp.max();

  // rescale logp to [-1, 1]
  return (2.0 * (logp - log_refp_min) / (log_refp_max - log_refp_min) - 1.0).view(pres.sizes());

  auto logp_scaled = torch::zeros({logp.sizes(0), 2}, logp.options());
  logp_scaled.select(1, 1) = pscale;

  auto tem = refatm_[index::ITM].view({1, 1, 1, -1}).expand({1, 1, 2, -1});
  auto grid = logp_scaled.view({1, 1, -1, 2});

  // rescale tema to [-1, 1]
  auto tgrid = 2.0 * (tema - tema_.min()) / (tema_max - tema_.max()) - 1.0;

  // rescale xcomp to [-1, 1]
  auto xgrid = 2.0 * (xcom - xcom_.min()) / (xcom_max - xcom_.max()) - 1.0;

  return {torch::grid_sample(tem, grid, "bilinear", "border").view(pres.sizes()), logp_scaled.select(1, 1)};
}

void InterpAttenuator::load() {
#ifdef NETCDFOUTPUT
  int fileid, dimid, varid, err;
  nc_open(options.opacity_file.c_str(), NC_NETCDF4, &fileid);

  nc_inq_dimid(fileid, "Wavenumber", &dimid);
  nc_inq_dimlen(fileid, dimid, len_);
  nc_inq_dimid(fileid, "Pressure", &dimid);
  nc_inq_dimlen(fileid, dimid, len_ + 1);
  nc_inq_dimid(fileid, "TempGrid", &dimid);
  nc_inq_dimlen(fileid, dimid, len_ + 2);

  axis_.resize(len_[0] + len_[1] + len_[2]);

  nc_inq_varid(fileid, "Wavenumber", &varid);
  nc_get_var_double(fileid, varid, axis_.data());

  err = nc_inq_varid(fileid, "Pressure", &varid);
  if (err != NC_NOERR) {
    throw std::runtime_error(nc_strerror(err));
  }

  err = nc_get_var_double(fileid, varid, axis_.data() + len_[0]);
  if (err != NC_NOERR) {
    throw std::runtime_error(nc_strerror(err));
  }

  err = nc_inq_varid(fileid, "TempGrid", &varid);
  if (err != NC_NOERR) {
    throw std::runtime_error(nc_strerror(err));
  }

  err = nc_get_var_double(fileid, varid, axis_.data() + len_[0] + len_[1]);
  if (err != NC_NOERR) {
    throw std::runtime_error(nc_strerror(err));
  }

  Real *temp = new Real[len_[1]];
  nc_inq_varid(fileid, "Temperature", &varid);
  nc_get_var_double(fileid, varid, temp);

  refatm_.NewAthenaArray(NHYDRO, len_[1]);
  for (int i = 0; i < len_[1]; i++) {
    refatm_(IPR, i) = axis_[len_[0] + i];
    refatm_(IDN, i) = temp[i];
  }

  kcross_.resize(len_[0] * len_[1] * len_[2]);
  nc_inq_varid(fileid, GetName().c_str(), &varid);
  nc_get_var_double(fileid, varid, kcross_.data());
  nc_close(fileid);
  delete[] temp;
#endif
}

torch::Tensor InterpAttenuator::attenuation(torch::Tensor var_x) const {
  auto grid = get_scaled_xpt(var_x);

  // interpolate to model grid
  auto kcross = torch::grid_sample(kcross_, grid, "bilinear", "border");

  auto x0 = var_x[options.species_id[0]];
  auto dens = x0 * var_x[index::IPR] / (Constants::Rgas * var_x[index::ITM]);
  return 1.E-3 * kcross.exp() * dens;  // ln(m^2/kmol) -> 1/m
}

} // namespace harp
