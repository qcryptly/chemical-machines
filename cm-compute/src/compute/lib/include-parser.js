/**
 * Include Parser
 *
 * Parses C++ source code to extract #include directives
 * and determine required dependencies.
 */

const { getFlagsForIncludes, isStandardHeader } = require('./package-metadata');

/**
 * Parse #include directives from C++ source code
 * @param {string} code - C++ source code
 * @returns {string[]} Array of included headers
 */
function parseIncludes(code) {
  const includes = [];

  // Match both <header> and "header" style includes
  // Also handle multi-line and various whitespace patterns
  const includeRegex = /#\s*include\s*[<"]([^>"]+)[>"]/gm;

  let match;
  while ((match = includeRegex.exec(code)) !== null) {
    includes.push(match[1]);
  }

  return [...new Set(includes)]; // Remove duplicates
}

/**
 * Analyze code and return required compiler/linker flags
 * @param {string} code - C++ source code
 * @param {Object} cppEnv - C++ environment with packages
 * @param {Object} vendorEnv - Vendor environment with installations
 * @param {Object} options - Compiler options { compiler, cppStandard }
 * @returns {Object} Compiler configuration
 */
function analyzeCode(code, cppEnv = null, vendorEnv = null, options = {}) {
  const includes = parseIncludes(code);
  const { compiler: preferredCompiler = 'clang++', cppStandard = 'c++23' } = options;

  // Get flags from package metadata
  const packageFlags = getFlagsForIncludes(includes);

  // Add vendor environment paths
  const vendorIncludes = [];
  const vendorLibs = [];

  if (vendorEnv && vendorEnv.installations) {
    for (const install of vendorEnv.installations) {
      const prefix = install.install_prefix;
      vendorIncludes.push(`-I${prefix}/include`);
      vendorLibs.push(`-L${prefix}/lib`);
      vendorLibs.push(`-Wl,-rpath,${prefix}/lib`);
    }
  }

  // Detect if CUDA is needed
  const usesCuda = includes.some(h =>
    h.includes('cuda') ||
    h.includes('cublas') ||
    h.includes('cufft') ||
    h.includes('cusparse') ||
    h.includes('cudnn')
  );

  // Detect if OpenMP is needed
  const usesOpenMP = includes.some(h => h === 'omp.h') ||
    code.includes('#pragma omp');

  // Determine compiler - use nvcc for CUDA, otherwise use preferred compiler
  let compiler = preferredCompiler;
  if (usesCuda && (code.includes('__global__') || code.includes('<<<'))) {
    compiler = 'nvcc';
  }

  // Build compile flags with user-specified C++ standard
  const compileFlags = [
    `-std=${cppStandard}`,
    '-O2',
    ...packageFlags.includes,
    ...vendorIncludes
  ];

  if (usesOpenMP && compiler !== 'nvcc') {
    compileFlags.push('-fopenmp');
  }

  // Build link flags
  const linkFlags = [
    ...vendorLibs,
    ...packageFlags.libs
  ];

  // Remove duplicates
  const uniqueCompileFlags = [...new Set(compileFlags)];
  const uniqueLinkFlags = [...new Set(linkFlags)];

  return {
    compiler,
    cppStandard,
    includes,
    compileFlags: uniqueCompileFlags,
    linkFlags: uniqueLinkFlags,
    suggestedPackages: packageFlags.packages,
    usesCuda,
    usesOpenMP
  };
}

/**
 * Generate the full compilation command
 * @param {string} sourceFile - Source file path
 * @param {string} outputFile - Output executable path
 * @param {Object} analysis - Result from analyzeCode
 * @returns {string[]} Command arguments
 */
function generateCompileCommand(sourceFile, outputFile, analysis) {
  const args = [
    ...analysis.compileFlags,
    '-o', outputFile,
    sourceFile,
    ...analysis.linkFlags
  ];

  return {
    compiler: analysis.compiler,
    args
  };
}

/**
 * Check for potential issues with the code
 * @param {string} code - C++ source code
 * @param {Object} cppEnv - C++ environment
 * @returns {Object} Warnings and suggestions
 */
function checkCode(code, cppEnv = null) {
  const warnings = [];
  const suggestions = [];
  const includes = parseIncludes(code);

  // Get required packages
  const { packages } = getFlagsForIncludes(includes);

  // Check if required packages are installed in the environment
  if (cppEnv && cppEnv.packages) {
    const installedPkgs = new Set(cppEnv.packages);
    for (const pkg of packages) {
      if (pkg && !installedPkgs.has(pkg)) {
        warnings.push(`Package '${pkg}' may be required but is not in the environment`);
        suggestions.push(`Add '${pkg}' to your C++ environment`);
      }
    }
  }

  // Check for common issues
  if (code.includes('using namespace std;') && code.length > 1000) {
    warnings.push("'using namespace std;' in large files can cause naming conflicts");
  }

  // Check for main function
  if (!code.includes('int main')) {
    warnings.push("No 'int main' function found - code may not be executable");
  }

  return { warnings, suggestions };
}

module.exports = {
  parseIncludes,
  analyzeCode,
  generateCompileCommand,
  checkCode
};
