# Chemical Machines Build Scripts

This directory contains automation scripts for the Chemical Machines project.

## Autocomplete Generator

**File**: `update-autocomplete.js`

Automatically generates and updates autocomplete entries for CodeCell.vue based on exports from cm-libraries modules.

### Features

- **Automatic Discovery**: Scans all Python modules in `cm-libraries/python/cm/` and extracts exported symbols from `__all__`
- **Smart Categorization**: Automatically categorizes exports into constants, classes, and functions based on naming conventions
- **Watch Mode**: Monitors `__init__.py` files for changes and automatically regenerates autocomplete
- **Preserves Existing Entries**: Keeps existing non-CM autocomplete entries (numpy, scipy, etc.) intact

### Usage

#### Manual Update
```bash
# Run once to update autocomplete
node scripts/update-autocomplete.js
```

#### Watch Mode
```bash
# Continuously watch for changes and auto-update
node scripts/update-autocomplete.js --watch
```

#### In Production
The autocomplete watcher runs automatically via supervisor as `autocomplete-watcher` and monitors for changes to any `__init__.py` files in the cm-libraries.

### How It Works

1. **Scanning**: Reads all `__init__.py` files in `cm-libraries/python/cm/`
2. **Parsing**: Extracts the `__all__` list and categorizes each export:
   - **Constants**: ALL_CAPS_WITH_UNDERSCORES
   - **Classes**: PascalCase (starts with uppercase)
   - **Functions**: snake_case (starts with lowercase)
3. **Generation**: Creates autocomplete entries with labels, types, and detail strings
4. **Updating**: Replaces the auto-generated section in CodeCell.vue while preserving manually added entries
5. **Watching** (in watch mode): Uses chokidar to detect file changes and triggers regeneration after 1 second debounce

### Configuration

The script watches these patterns:
- `cm-libraries/python/cm/*/__init__.py`

To add support for C++ libraries, extend the script to parse C++ headers and update the watch patterns.

### Output Format

Generated autocomplete entries follow this format:

```javascript
// CM library - auto-generated
{ label: 'cm.module', type: 'module', detail: 'Module description' },
{ label: 'from cm.module import', type: 'text', detail: 'import CM module' },
// CM module - constants
{ label: 'CONSTANT_NAME', type: 'constant', detail: 'cm.module: constant name' },
// CM module - classes
{ label: 'ClassName', type: 'class', detail: 'cm.module: ClassName' },
// CM module - functions
{ label: 'function_name', type: 'function', detail: 'cm.module: function_name' },
```

### Supervisor Integration

The watcher runs as a supervisor program:

```ini
[program:scripts-setup]
command=/bin/bash -c "npm install || true"
directory=/app/scripts
priority=3

[program:autocomplete-watcher]
command=node /app/scripts/update-autocomplete.js --watch
directory=/app
priority=5
depends_on=scripts-setup
```

### Logs

Watch logs in real-time:
```bash
docker compose exec chemical-machines tail -f /var/log/supervisor/autocomplete-watcher.log
```

### Dependencies

- `chokidar` (^3.5.3) - File system watcher

Install with:
```bash
cd scripts && npm install
```

### Extending the Script

To add support for additional autocomplete sources:

1. **Add Scanner Function**: Create a function like `scanCppLibraries()` to parse your source
2. **Update Main Function**: Call your scanner in `updateAutocomplete()`
3. **Update Watch Patterns**: Add file patterns to the chokidar watcher
4. **Generate Entries**: Format entries according to the CodeMirror autocomplete spec

### Troubleshooting

**Autocomplete not updating?**
- Check if the watcher is running: `docker compose exec chemical-machines ps aux | grep autocomplete`
- Check logs: `docker compose exec chemical-machines cat /var/log/supervisor/autocomplete-watcher.log`
- Manually trigger: `docker compose exec chemical-machines node /app/scripts/update-autocomplete.js`

**Missing exports?**
- Ensure exports are listed in the module's `__all__` list
- Check that `__init__.py` follows Python conventions
- Run manually to see what was detected: `node scripts/update-autocomplete.js`

**Build failing?**
- Ensure chokidar is installed: `cd scripts && npm install`
- Check Node.js version compatibility (requires Node 14+)
