// C/C++
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

// canoe
#include <air_parcel.hpp>
#include <configure.hpp>
#include <constants.hpp>

// climath
#include <climath/interpolation.h>

// harp
#include "absorber.hpp"

// netcdf
#ifdef NETCDFOUTPUT
extern "C" {
#include <netcdf.h>
}
#endif

torch::Tensor HitranAbsorber::get_reftemp_(torch::Tensor pres) const {
  auto log_refp = refatm_[index::IPR];
  auto logp = pres.log();

  int nlevel = refatm_.GetDim1();
  int jl = -1, ju = nlevel, jm;
  // make sure pressure is in ascending order
  while (ju - jl > 1) {
    jm = (ju + jl) >> 1;
    if (pres < refatm_(IPR, jm))
      ju = jm;
    else
      jl = jm;
  }

  // prevent interpolation problem at boundary
  if (jl == -1) jl = 0;
  if (ju == nlevel) ju = nlevel - 1;
  if (jl == ju) return refatm_(IDN, jl);

  // assert(jl >= 0 && ju < nlevel);
  Real result = log(refatm_(IPR, jl) / pres) * log(refatm_(IDN, ju)) +
                log(pres / refatm_(IPR, ju)) * log(refatm_(IDN, jl));
  result = exp(result / log(refatm_(IPR, jl) / refatm_(IPR, ju)));
  return result;
}

void HitranAbsorber::load() {
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

  kcoeff_.resize(len_[0] * len_[1] * len_[2]);
  nc_inq_varid(fileid, GetName().c_str(), &varid);
  nc_get_var_double(fileid, varid, kcoeff_.data());
  nc_close(fileid);
  delete[] temp;
#endif
}

torch::Tensor HitranAbsorber::attenuation(torch::Tensor spec,
                                          torch::Tensor var_x) const {
  // first axis is wavenumber, second is pressure, third is temperature anomaly
  Real val,
      coord[3] = {spec, hydro_x[IPR], hydro_x[ITM] - get_reftemp_(var.w[IPR])};
  interpn(&val, coord, kcoeff_.data(), axis_.data(), len_, 3, 1);

  auto x0 = var_x[species_id[0]];
  auto dens = x0 * var_x[index::IPR] / (Constants::Rgas * var_x[index::ITM]);
  return 1.E-3 * val.exp() * dens;  // ln(m*2/kmol) -> 1/m
}

void HitranAbsorberCK::LoadCoefficient(std::string fname, int b) {
  // load lbl hitran absorber data
  Base::LoadCoefficient(fname, b);

  // load weights
#ifdef NETCDFOUTPUT
  int fileid, dimid, varid, err;
  nc_open(fname.c_str(), NC_NETCDF4, &fileid);

  size_t len = 0;
  nc_inq_dimid(fileid, "weights", &dimid);
  nc_inq_dimlen(fileid, dimid, &len);
  weights_.resize(len);

  err = nc_inq_varid(fileid, "weights", &varid);
  if (err != NC_NOERR) {
    throw std::runtime_error(nc_strerror(err));
  }

  err = nc_get_var_double(fileid, varid, weights_.data());
  if (err != NC_NOERR) {
    throw std::runtime_error(nc_strerror(err));
  }

  nc_close(fileid);
#endif
}

//(cmetz) spec is initially unsized because
// CKTableSpectralGrid::CKTableSpectralGrid is left undefined but this is OK,
// because ModifySpectralGrid is called right after LoadCoeff in RadiationBand
// constructor
void HitranAbsorberCK::ModifySpectralGrid(
    std::vector<SpectralBin> &spec) const {
  spec.resize(weights_.size());

  for (size_t i = 0; i < weights_.size(); ++i) {
    spec[i].wav1 = axis_[i];
    spec[i].wav2 = axis_[i];
    spec[i].wght = weights_[i];
  }
}
