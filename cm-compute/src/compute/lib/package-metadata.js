/**
 * Package Metadata
 *
 * Maps common C++ headers to their debian packages and linker flags.
 * Used for auto-detection of required dependencies when compiling C++ code.
 */

/**
 * Mapping of header patterns to package info and compiler/linker flags
 */
const HEADER_TO_PACKAGE = {
  // Boost libraries
  'boost/': {
    pkg: 'libboost-all-dev',
    include: [],
    libs: []  // Most boost is header-only, specific libs need explicit linking
  },
  'boost/filesystem': {
    pkg: 'libboost-filesystem-dev',
    include: [],
    libs: ['-lboost_filesystem']
  },
  'boost/system': {
    pkg: 'libboost-system-dev',
    include: [],
    libs: ['-lboost_system']
  },
  'boost/thread': {
    pkg: 'libboost-thread-dev',
    include: [],
    libs: ['-lboost_thread', '-lpthread']
  },
  'boost/regex': {
    pkg: 'libboost-regex-dev',
    include: [],
    libs: ['-lboost_regex']
  },
  'boost/program_options': {
    pkg: 'libboost-program-options-dev',
    include: [],
    libs: ['-lboost_program_options']
  },

  // Linear algebra
  'Eigen/': {
    pkg: 'libeigen3-dev',
    include: ['-I/usr/include/eigen3'],
    libs: []
  },
  'eigen3/': {
    pkg: 'libeigen3-dev',
    include: ['-I/usr/include/eigen3'],
    libs: []
  },
  'cblas.h': {
    pkg: 'libopenblas-dev',
    include: [],
    libs: ['-lopenblas']
  },
  'openblas/': {
    pkg: 'libopenblas-dev',
    include: [],
    libs: ['-lopenblas']
  },
  'lapack': {
    pkg: 'liblapack-dev',
    include: [],
    libs: ['-llapack', '-lblas']
  },
  'armadillo': {
    pkg: 'libarmadillo-dev',
    include: [],
    libs: ['-larmadillo']
  },

  // CUDA
  'cuda_runtime.h': {
    pkg: 'cuda',
    include: ['-I/usr/local/cuda/include'],
    libs: ['-L/usr/local/cuda/lib64', '-lcudart']
  },
  'cuda.h': {
    pkg: 'cuda',
    include: ['-I/usr/local/cuda/include'],
    libs: ['-L/usr/local/cuda/lib64', '-lcuda']
  },
  'cublas': {
    pkg: 'cuda',
    include: ['-I/usr/local/cuda/include'],
    libs: ['-L/usr/local/cuda/lib64', '-lcublas']
  },
  'cufft': {
    pkg: 'cuda',
    include: ['-I/usr/local/cuda/include'],
    libs: ['-L/usr/local/cuda/lib64', '-lcufft']
  },
  'cusparse': {
    pkg: 'cuda',
    include: ['-I/usr/local/cuda/include'],
    libs: ['-L/usr/local/cuda/lib64', '-lcusparse']
  },
  'cudnn': {
    pkg: 'cudnn',
    include: ['-I/usr/local/cuda/include'],
    libs: ['-L/usr/local/cuda/lib64', '-lcudnn']
  },

  // OpenMP
  'omp.h': {
    pkg: 'libomp-dev',
    include: [],
    libs: ['-fopenmp']
  },

  // MPI
  'mpi.h': {
    pkg: 'libopenmpi-dev',
    include: [],
    libs: ['-lmpi']
  },

  // HDF5
  'hdf5.h': {
    pkg: 'libhdf5-dev',
    include: [],
    libs: ['-lhdf5']
  },
  'H5Cpp.h': {
    pkg: 'libhdf5-cpp-dev',
    include: [],
    libs: ['-lhdf5_cpp', '-lhdf5']
  },

  // Image processing
  'opencv2/': {
    pkg: 'libopencv-dev',
    include: [],
    libs: ['-lopencv_core', '-lopencv_imgproc', '-lopencv_highgui']
  },
  'png.h': {
    pkg: 'libpng-dev',
    include: [],
    libs: ['-lpng']
  },
  'jpeglib.h': {
    pkg: 'libjpeg-dev',
    include: [],
    libs: ['-ljpeg']
  },

  // JSON
  'nlohmann/json': {
    pkg: 'nlohmann-json3-dev',
    include: [],
    libs: []
  },
  'rapidjson/': {
    pkg: 'rapidjson-dev',
    include: [],
    libs: []
  },

  // XML
  'libxml/': {
    pkg: 'libxml2-dev',
    include: ['-I/usr/include/libxml2'],
    libs: ['-lxml2']
  },
  'tinyxml2.h': {
    pkg: 'libtinyxml2-dev',
    include: [],
    libs: ['-ltinyxml2']
  },

  // Networking
  'curl/curl.h': {
    pkg: 'libcurl4-openssl-dev',
    include: [],
    libs: ['-lcurl']
  },
  'zmq.h': {
    pkg: 'libzmq3-dev',
    include: [],
    libs: ['-lzmq']
  },

  // Compression
  'zlib.h': {
    pkg: 'zlib1g-dev',
    include: [],
    libs: ['-lz']
  },
  'bzlib.h': {
    pkg: 'libbz2-dev',
    include: [],
    libs: ['-lbz2']
  },
  'lz4.h': {
    pkg: 'liblz4-dev',
    include: [],
    libs: ['-llz4']
  },

  // Cryptography
  'openssl/': {
    pkg: 'libssl-dev',
    include: [],
    libs: ['-lssl', '-lcrypto']
  },

  // Database
  'sqlite3.h': {
    pkg: 'libsqlite3-dev',
    include: [],
    libs: ['-lsqlite3']
  },
  'libpq-fe.h': {
    pkg: 'libpq-dev',
    include: [],
    libs: ['-lpq']
  },

  // Graphics
  'GL/gl.h': {
    pkg: 'libgl-dev',
    include: [],
    libs: ['-lGL']
  },
  'GL/glew.h': {
    pkg: 'libglew-dev',
    include: [],
    libs: ['-lGLEW', '-lGL']
  },
  'GLFW/glfw3.h': {
    pkg: 'libglfw3-dev',
    include: [],
    libs: ['-lglfw', '-lGL']
  },
  'vulkan/': {
    pkg: 'libvulkan-dev',
    include: [],
    libs: ['-lvulkan']
  },

  // Threading
  'pthread.h': {
    pkg: null,  // Usually built-in
    include: [],
    libs: ['-lpthread']
  },
  'tbb/': {
    pkg: 'libtbb-dev',
    include: [],
    libs: ['-ltbb']
  },

  // Scientific computing
  'fftw3.h': {
    pkg: 'libfftw3-dev',
    include: [],
    libs: ['-lfftw3']
  },
  'gsl/': {
    pkg: 'libgsl-dev',
    include: [],
    libs: ['-lgsl', '-lgslcblas']
  },

  // Logging
  'spdlog/': {
    pkg: 'libspdlog-dev',
    include: [],
    libs: []  // Header-only by default
  },
  'fmt/': {
    pkg: 'libfmt-dev',
    include: [],
    libs: ['-lfmt']
  },

  // Testing
  'gtest/': {
    pkg: 'libgtest-dev',
    include: [],
    libs: ['-lgtest', '-lgtest_main', '-lpthread']
  },
  'gmock/': {
    pkg: 'libgmock-dev',
    include: [],
    libs: ['-lgmock', '-lgmock_main']
  },

  // Molecular dynamics / chemistry
  'rdkit/': {
    pkg: 'librdkit-dev',
    include: [],
    libs: ['-lRDKitGraphMol', '-lRDKitSmilesParse']
  },
  'openbabel/': {
    pkg: 'libopenbabel-dev',
    include: [],
    libs: ['-lopenbabel']
  }
};

/**
 * Standard C++ headers that don't need external packages
 */
const STANDARD_HEADERS = new Set([
  'iostream', 'vector', 'string', 'map', 'set', 'unordered_map', 'unordered_set',
  'algorithm', 'memory', 'functional', 'utility', 'tuple', 'array', 'deque',
  'list', 'queue', 'stack', 'bitset', 'complex', 'valarray', 'numeric',
  'cmath', 'cstdlib', 'cstdio', 'cstring', 'ctime', 'cassert', 'cctype',
  'climits', 'cfloat', 'cstdint', 'cinttypes', 'cstddef', 'cerrno',
  'fstream', 'sstream', 'iomanip', 'iosfwd', 'streambuf',
  'exception', 'stdexcept', 'typeinfo', 'type_traits', 'limits',
  'thread', 'mutex', 'condition_variable', 'future', 'atomic',
  'chrono', 'ratio', 'random', 'regex', 'locale', 'codecvt',
  'filesystem', 'optional', 'variant', 'any', 'string_view', 'span',
  'new', 'initializer_list', 'typeindex', 'source_location',
  'concepts', 'coroutine', 'compare', 'version', 'numbers', 'bit', 'ranges',
  'format', 'expected', 'print', 'stacktrace', 'generator',
  // C headers available in C++
  'stdio.h', 'stdlib.h', 'string.h', 'math.h', 'time.h', 'assert.h',
  'ctype.h', 'errno.h', 'float.h', 'limits.h', 'locale.h', 'setjmp.h',
  'signal.h', 'stdarg.h', 'stddef.h', 'stdint.h', 'inttypes.h',
  'wchar.h', 'wctype.h', 'iso646.h', 'stdbool.h', 'fenv.h', 'tgmath.h',
  'uchar.h', 'stdalign.h', 'stdnoreturn.h', 'threads.h', 'stdatomic.h'
]);

/**
 * Check if a header is a standard C/C++ header
 */
function isStandardHeader(header) {
  // Remove .h extension for checking
  const base = header.replace(/\.h$/, '');
  return STANDARD_HEADERS.has(header) || STANDARD_HEADERS.has(base);
}

/**
 * Find package info for a given header
 */
function findPackageForHeader(header) {
  // Check exact match first
  if (HEADER_TO_PACKAGE[header]) {
    return HEADER_TO_PACKAGE[header];
  }

  // Check prefix matches
  for (const [pattern, info] of Object.entries(HEADER_TO_PACKAGE)) {
    if (header.startsWith(pattern) || header.includes('/' + pattern)) {
      return info;
    }
  }

  return null;
}

/**
 * Get all compiler flags needed for a set of includes
 */
function getFlagsForIncludes(includes) {
  const includeFlags = new Set();
  const libFlags = new Set();
  const packages = new Set();

  for (const include of includes) {
    if (isStandardHeader(include)) {
      continue;
    }

    const info = findPackageForHeader(include);
    if (info) {
      if (info.pkg) {
        packages.add(info.pkg);
      }
      info.include.forEach(f => includeFlags.add(f));
      info.libs.forEach(f => libFlags.add(f));
    }
  }

  return {
    includes: [...includeFlags],
    libs: [...libFlags],
    packages: [...packages]
  };
}

module.exports = {
  HEADER_TO_PACKAGE,
  STANDARD_HEADERS,
  isStandardHeader,
  findPackageForHeader,
  getFlagsForIncludes
};
