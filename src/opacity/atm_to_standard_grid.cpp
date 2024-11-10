
void AtmToStandardGrid::reset() {
  refatm = register_buffer("refatm",
                           torch::zeros({3, options.nlevel()}, torch::kFloat));

  taxis =
      register_buffer("taxis", torch::zeros({options.ntemp()}, torch::kFloat));

  xaxis =
      register_buffer("comp", torch::zeros({options.ncomp()}, torch::kFloat));
}

torch::Tensor AtmToStandardGrid::forward(torch::Tensor var_x) const {
  // log pressure
  auto log_refp = refatm_[index::IPR];
  auto logp = pres.log().flatten();

  auto log_refp_min = log_refp.min();
  auto log_refp_max = log_refp.max();

  // rescale logp to [-1, 1]
  return (2.0 * (logp - log_refp_min) / (log_refp_max - log_refp_min) - 1.0)
      .view(pres.sizes());

  auto logp_scaled = torch::zeros({logp.sizes(0), 2}, logp.options());
  logp_scaled.select(1, 1) = pscale;

  auto tem = refatm_[index::ITM].view({1, 1, 1, -1}).expand({1, 1, 2, -1});
  auto grid = logp_scaled.view({1, 1, -1, 2});

  // rescale tema to [-1, 1]
  auto tgrid = 2.0 * (tema - tema_.min()) / (tema_max - tema_.max()) - 1.0;

  // rescale xcomp to [-1, 1]
  auto xgrid = 2.0 * (xcom - xcom_.min()) / (xcom_max - xcom_.max()) - 1.0;

  return {
      torch::grid_sample(tem, grid, "bilinear", "border").view(pres.sizes()),
      logp_scaled.select(1, 1)};
}
