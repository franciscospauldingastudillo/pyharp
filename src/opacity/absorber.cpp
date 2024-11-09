// C/C++
#include <sstream>
#include <string>

// external
#include <yaml-cpp/yaml.h>

// harp
#include "absorber.hpp"
#include "find_resource.hpp"

Absorber::Absorber(std::string name) : NamedGroup(name), opacity_filename_("") {
  Application::Logger app("opacity");
  app->Log("Create Absorber " + name);
}

void Absorber::LoadOpacityFromFile(std::string filename) {
  SetOpacityFile(filename);
  LoadOpacity(-1);
}

void Absorber::LoadOpacity(int bid) {
  auto app = Application::GetInstance();
  auto log = app->GetMonitor("opacity");

  if (opacity_filename_.empty()) return;

  std::string full_path = app->FindResource(opacity_filename_);
  log->Log("Load opacity from " + full_path);
  LoadCoefficient(full_path, bid);
}
