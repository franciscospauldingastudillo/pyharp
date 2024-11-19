#pragma once

namespace harp {
enum index {
  ITM = 0,  // temperature
  IPR = 1,  // pressure
  ICX = 2,  // composition
            //
  IAB = 0,  // absorption
  ISS = 1,  // single scattering
  IPM = 2,  // phase moments
            //
  IUP = 0,  // upward
  IDN = 1,  // downward
};
}  // namespace harp
