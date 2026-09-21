// Gazebo Garden / Ogre 2.3 uses the host CPU count inside containers.
// Ogre 2.3 asserts when destroying >=128 scene worker threads. Interpose
// only Ogre's CPU-count method, scoped to this project's Gazebo processes.
#include <OgrePlatformInformation.h>
#include <algorithm>
#include <cstdlib>
#include <unistd.h>

Ogre::uint32 Ogre::PlatformInformation::getNumLogicalCores(void)
{
  long requested = 8;
  if (const char *value = std::getenv("ARBOIDS_RENDER_THREADS")) {
    char *end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end != value && *end == '\0' && parsed > 0)
      requested = parsed;
  }
  const long available = std::max(1L, ::sysconf(_SC_NPROCESSORS_ONLN));
  return static_cast<Ogre::uint32>(std::max(1L, std::min({requested, available, 64L})));
}
