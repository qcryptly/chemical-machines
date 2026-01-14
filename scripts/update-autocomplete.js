#!/usr/bin/env node

/**
 * Autocomplete Generator for Chemical Machines
 *
 * Automatically scans cm-libraries/python and cm-libraries/cpp to generate
 * autocomplete entries for CodeCell.vue
 *
 * Usage:
 *   node scripts/update-autocomplete.js [--watch]
 */

const fs = require('fs');
const path = require('path');
const chokidar = require('chokidar');

// Paths
const CM_LIBRARIES_PYTHON = path.join(__dirname, '../cm-libraries/python');
const CM_LIBRARIES_CPP = path.join(__dirname, '../cm-libraries/cpp');
const CODE_CELL_VUE = path.join(__dirname, '../cm-view/client/src/components/CodeCell.vue');

// ANSI colors for console output
const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[36m',
  red: '\x1b[31m'
};

function log(message, color = 'reset') {
  console.log(`${colors[color]}${message}${colors.reset}`);
}

/**
 * Parse Python source file to extract class methods
 */
function parseClassMethods(filePath, className) {
  try {
    const content = fs.readFileSync(filePath, 'utf-8');
    const methods = [];

    // Find class definition
    const classRegex = new RegExp(`class\\s+${className}[\\s\\(:]`, 'm');
    if (!classRegex.test(content)) {
      return methods;
    }

    // Extract method definitions (def method_name)
    // Match methods but exclude private ones (starting with _)
    const methodRegex = /^\s{4,}def\s+([a-z_][a-z0-9_]*)\s*\(/gm;
    let match;

    while ((match = methodRegex.exec(content)) !== null) {
      const methodName = match[1];
      // Skip private/magic methods
      if (!methodName.startsWith('_')) {
        methods.push(methodName);
      }
    }

    return [...new Set(methods)]; // Remove duplicates
  } catch (error) {
    return [];
  }
}

/**
 * Get methods for a specific class
 */
function getClassMethods(moduleName, className, modulePath) {
  const methods = [];

  // Try common file locations
  const possibleFiles = [
    path.join(modulePath, `${className.toLowerCase()}.py`),
    path.join(modulePath, `${moduleName}.py`),
    path.join(modulePath, 'core.py'),
    path.join(modulePath, 'atoms.py'),
    path.join(modulePath, 'molecules.py'),
    path.join(modulePath, 'hamiltonian.py'),
    path.join(modulePath, 'functions.py'),
  ];

  for (const filePath of possibleFiles) {
    if (fs.existsSync(filePath)) {
      const classMethods = parseClassMethods(filePath, className);
      methods.push(...classMethods);
    }
  }

  return [...new Set(methods)]; // Remove duplicates
}

/**
 * Parse Python __init__.py to extract exported symbols
 */
function parsePythonInit(initPath) {
  const content = fs.readFileSync(initPath, 'utf-8');
  const exports = {
    modules: [],
    classes: [],
    functions: [],
    constants: []
  };

  // Extract __all__ list
  const allMatch = content.match(/__all__\s*=\s*\[([\s\S]*?)\]/);
  if (allMatch) {
    const allItems = allMatch[1]
      .split('\n')
      .map(line => line.trim())
      .filter(line => line.length > 0 && !line.startsWith('#'))
      .join(' ')
      .split(',')
      .map(item => item.trim().replace(/['"]/g, ''))
      .filter(item => item.length > 0 && !item.startsWith('#'));

    // Categorize based on naming conventions and content analysis
    allItems.forEach(item => {
      if (item.toUpperCase() === item && item.includes('_')) {
        exports.constants.push(item);
      } else if (item[0] === item[0].toUpperCase()) {
        exports.classes.push(item);
      } else {
        exports.functions.push(item);
      }
    });
  }

  // Extract from imports if __all__ not found
  if (exports.classes.length === 0 && exports.functions.length === 0) {
    // Try to parse from imports
    const importMatches = content.matchAll(/from\s+\.\S+\s+import\s+\(([\s\S]*?)\)/g);
    for (const match of importMatches) {
      const items = match[1]
        .split(',')
        .map(item => item.trim().replace(/['"]/g, ''))
        .filter(item => item.length > 0 && !item.startsWith('#'));

      items.forEach(item => {
        if (item.toUpperCase() === item && item.includes('_')) {
          exports.constants.push(item);
        } else if (item[0] === item[0].toUpperCase()) {
          exports.classes.push(item);
        } else {
          exports.functions.push(item);
        }
      });
    }
  }

  return exports;
}

/**
 * Get module docstring for detail text
 */
function getModuleDocstring(initPath) {
  const content = fs.readFileSync(initPath, 'utf-8');
  const docMatch = content.match(/^"""([\s\S]*?)"""/);
  if (docMatch) {
    const firstLine = docMatch[1].split('\n').find(line => line.trim().length > 0);
    return firstLine ? firstLine.trim() : '';
  }
  return '';
}

/**
 * Generate autocomplete entries for a Python module
 */
function generatePythonModuleAutocomplete(moduleName, modulePath) {
  const entries = [];
  const initPath = path.join(modulePath, '__init__.py');

  if (!fs.existsSync(initPath)) {
    return entries;
  }

  const docstring = getModuleDocstring(initPath);
  const exports = parsePythonInit(initPath);

  // Add module entry
  entries.push({
    label: `cm.${moduleName}`,
    type: 'module',
    detail: docstring || `CM ${moduleName} module`
  });

  entries.push({
    label: `from cm.${moduleName} import`,
    type: 'text',
    detail: `import CM ${moduleName}`
  });

  // Add constants
  if (exports.constants.length > 0) {
    entries.push(`// CM ${moduleName} - constants`);
    exports.constants.forEach(name => {
      entries.push({
        label: name,
        type: 'constant',
        detail: `cm.${moduleName}: ${name.toLowerCase().replace(/_/g, ' ')}`
      });
    });
  }

  // Add classes with their methods
  if (exports.classes.length > 0) {
    entries.push(`// CM ${moduleName} - classes`);
    exports.classes.forEach(name => {
      entries.push({
        label: name,
        type: 'class',
        detail: `cm.${moduleName}: ${name}`
      });

      // Add class methods (for autocomplete like: atom_instance.method)
      const methods = getClassMethods(moduleName, name, modulePath);
      if (methods.length > 0) {
        methods.forEach(method => {
          entries.push({
            label: `${name}.${method}`,
            type: 'method',
            detail: `${name} method: ${method}()`
          });
        });
      }
    });
  }

  // Add functions
  if (exports.functions.length > 0) {
    entries.push(`// CM ${moduleName} - functions`);
    exports.functions.forEach(name => {
      entries.push({
        label: name,
        type: 'function',
        detail: `cm.${moduleName}: ${name}`
      });
    });
  }

  return entries;
}

/**
 * Scan all Python modules in cm-libraries/python/cm
 */
function scanPythonLibraries() {
  const cmDir = path.join(CM_LIBRARIES_PYTHON, 'cm');
  const allEntries = [];

  if (!fs.existsSync(cmDir)) {
    log('Warning: cm-libraries/python/cm not found', 'yellow');
    return allEntries;
  }

  const modules = fs.readdirSync(cmDir)
    .filter(name => {
      const modulePath = path.join(cmDir, name);
      return fs.statSync(modulePath).isDirectory() &&
             fs.existsSync(path.join(modulePath, '__init__.py'));
    })
    .sort();

  modules.forEach(moduleName => {
    const modulePath = path.join(cmDir, moduleName);
    const entries = generatePythonModuleAutocomplete(moduleName, modulePath);
    if (entries.length > 0) {
      allEntries.push(...entries);
    }
  });

  return allEntries;
}

/**
 * Format autocomplete entry for Vue template
 */
function formatEntry(entry) {
  if (typeof entry === 'string') {
    return `  ${entry}`;
  }

  const detail = entry.detail.replace(/'/g, "\\'");
  return `  { label: '${entry.label}', type: '${entry.type}', detail: '${detail}' },`;
}

/**
 * Update CodeCell.vue with new autocomplete entries
 */
function updateCodeCellVue(entries) {
  if (!fs.existsSync(CODE_CELL_VUE)) {
    log('Error: CodeCell.vue not found', 'red');
    return false;
  }

  let content = fs.readFileSync(CODE_CELL_VUE, 'utf-8');

  // Find the Python completions array
  const startMarker = 'const pythonCompletions = [';
  const endMarker = '];';

  const startIndex = content.indexOf(startMarker);
  if (startIndex === -1) {
    log('Error: Could not find pythonCompletions array', 'red');
    return false;
  }

  // Find the end of the array (we need to be careful not to match other arrays)
  let endIndex = startIndex + startMarker.length;
  let depth = 1;
  let inString = false;
  let stringChar = null;

  for (let i = endIndex; i < content.length; i++) {
    const char = content[i];
    const prevChar = i > 0 ? content[i - 1] : '';

    // Handle string literals
    if ((char === '"' || char === "'" || char === '`') && prevChar !== '\\') {
      if (!inString) {
        inString = true;
        stringChar = char;
      } else if (char === stringChar) {
        inString = false;
        stringChar = null;
      }
    }

    if (!inString) {
      if (char === '[') depth++;
      if (char === ']') depth--;

      if (depth === 0) {
        endIndex = i;
        break;
      }
    }
  }

  if (depth !== 0) {
    log('Error: Could not find end of pythonCompletions array', 'red');
    return false;
  }

  // Keep the existing non-CM entries (numpy, scipy, etc.)
  const oldArray = content.substring(startIndex + startMarker.length, endIndex).trim();
  const oldLines = oldArray.split('\n').map(line => line.trim()).filter(line => line.length > 0);

  // Find where CM library entries start
  const cmStartIndex = oldLines.findIndex(line =>
    line.includes('CM library') ||
    line.includes('cm.views') ||
    line.includes('cm.data') ||
    line.includes('cm.symbols') ||
    line.includes('cm.qm')
  );

  let preservedEntries = [];
  if (cmStartIndex > 0) {
    preservedEntries = oldLines.slice(0, cmStartIndex);
  } else {
    // If no CM entries found, preserve everything except trailing comma/bracket
    preservedEntries = oldLines.filter(line =>
      !line.startsWith(']') && line !== ','
    );
  }

  // Build new array content
  const newArrayContent = [
    ...preservedEntries,
    '  // CM library - auto-generated',
    ...entries.map(formatEntry)
  ].join('\n');

  // Replace the array content
  const newContent =
    content.substring(0, startIndex + startMarker.length) + '\n' +
    newArrayContent + '\n' +
    content.substring(endIndex);

  fs.writeFileSync(CODE_CELL_VUE, newContent, 'utf-8');
  return true;
}

/**
 * Main function to scan and update autocomplete
 */
function updateAutocomplete() {
  log('Scanning cm-libraries for autocomplete updates...', 'blue');

  const pythonEntries = scanPythonLibraries();

  if (pythonEntries.length === 0) {
    log('No Python library entries found', 'yellow');
    return false;
  }

  log(`Found ${pythonEntries.length} Python autocomplete entries`, 'green');

  if (updateCodeCellVue(pythonEntries)) {
    log('Successfully updated CodeCell.vue autocomplete', 'green');
    return true;
  } else {
    log('Failed to update CodeCell.vue', 'red');
    return false;
  }
}

/**
 * Watch mode - monitor for changes
 */
function watchMode() {
  log('Starting autocomplete watcher...', 'bright');
  log(`Watching: ${CM_LIBRARIES_PYTHON}`, 'blue');
  log('Press Ctrl+C to stop\n', 'yellow');

  // Initial update
  updateAutocomplete();

  // Watch for changes
  const watcher = chokidar.watch([
    path.join(CM_LIBRARIES_PYTHON, 'cm/*/__init__.py'),
  ], {
    persistent: true,
    ignoreInitial: true
  });

  let updateTimer = null;

  watcher.on('change', (filePath) => {
    log(`\nDetected change: ${path.basename(filePath)}`, 'yellow');

    // Debounce updates (wait 1 second after last change)
    if (updateTimer) {
      clearTimeout(updateTimer);
    }

    updateTimer = setTimeout(() => {
      updateAutocomplete();
      log('Waiting for changes...\n', 'blue');
    }, 1000);
  });

  watcher.on('error', (error) => {
    log(`Watcher error: ${error}`, 'red');
  });
}

// Main execution
if (require.main === module) {
  const args = process.argv.slice(2);

  if (args.includes('--watch') || args.includes('-w')) {
    watchMode();
  } else {
    const success = updateAutocomplete();
    process.exit(success ? 0 : 1);
  }
}

module.exports = { updateAutocomplete, scanPythonLibraries };
