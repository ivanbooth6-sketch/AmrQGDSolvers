#include "QGDHeatFlux.H"

// The heat-flux helpers are header-defined so AMReX GPU kernels can inline them
// on host and device builds.  This translation unit makes the helper module
// explicit in the source tree and build system.
