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
AbsorberRFMImpl(AttenuatorOptions const& options_) : options(options_) {
  reset();
}

void AbsorberRFMImpl::reset() {
  kdata = register_buffer(
      "kdata",
      torch::zeros({1 + options.npmom(), options.nspec(), options.ncomp(),
                    options.nlevel(), options.ntemp()},
                   torch::kFloat));
  scale_grid = register_module("scale_grid", AtmToStandardGrid(options));
  load();
}

void AbsorberRFMImpl::load() {
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

  Real* temp = new Real[len_[1]];
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

torch::Tensor AbsorberRFMImpl::forward(torch::Tensor var_x) const {
  auto grid = scale_grid.forward(var_x, options.species_id[0]);

  // interpolate to model grid
  auto kcross = torch::grid_sample(kdata, grid, "bilinear", "border");

  auto x0 = var_x[options.species_id[0]];
  auto dens = x0 * var_x[index::IPR] / (Constants::Rgas * var_x[index::ITM]);
  return 1.E-3 * kcross.exp() * dens;  // ln(m^2/kmol) -> 1/m
}

}  // namespace harp
