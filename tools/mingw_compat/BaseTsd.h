/* Case-normalizing shim: MSVC-style headers use "BaseTsd.h" (mixed case);
   mingw-w64 ships the lowercase "basetsd.h". Windows' filesystem is
   case-insensitive so this only matters when cross-compiling from a
   case-sensitive host (Linux). -I this directory (ahead of the mingw
   system include path) resolves the mixed-case #include. */
#include <basetsd.h>
