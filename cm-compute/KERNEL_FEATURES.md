# Python Kernel Features

## Overview

The Python execution system now maintains persistent kernels for `.cell.py` files, providing Jupyter-like behavior where imports, variables, functions, and classes persist across cell executions.

## Features

### 1. Persistent Context (Jupyter-like behavior)

When you run cells in a `.cell.py` file:
- **One kernel per file**: Each file maintains its own isolated Python process
- **Persistent state**: Imports, variables, functions, and classes are available to all cells below
- **Execution order matters**: Context builds up as you run cells from top to bottom

Example:
```python
# %% Cell 1
import numpy as np
x = 42

# %% Cell 2
# numpy and x are available here
print(x)  # prints 42
arr = np.array([1, 2, 3])
```

### 2. Interrupt Execution

You can interrupt long-running or infinite-loop cells:

**Methods to interrupt:**
- Send `kernelAction: 'interrupt'` to stop the currently executing cell
- The kernel process receives SIGINT (equivalent to Ctrl+C)
- Context is preserved after interruption

**Behavior:**
- Interruption raises `KeyboardInterrupt` in Python
- Any code after the interruption point won't execute
- Variables defined before the interruption remain available
- The kernel stays alive and ready for the next cell

Example API call:
```javascript
{
  kernelAction: 'interrupt',
  cellInfo: {
    filePath: 'example.cell.py',
    isCellFile: true
  },
  sourceDir: ''
}
```

### 3. Execution Timeout

Set a maximum execution time for cells:

**Usage:**
```javascript
{
  code: 'import time; time.sleep(10)',
  timeout: 5000,  // 5 seconds in milliseconds
  cellInfo: {
    filePath: 'example.cell.py',
    cellIndex: 0,
    isCellFile: true
  }
}
```

**Behavior:**
- If execution exceeds the timeout, the cell is automatically interrupted
- Result includes `timedOut: true` flag
- Context is preserved after timeout

### 4. Kernel Management Actions

#### Stop Kernel
Terminates the kernel process completely:
```javascript
{
  kernelAction: 'stop',
  cellInfo: { filePath: 'example.cell.py', isCellFile: true }
}
```

#### Reset Kernel
Stops and removes the kernel (next execution will start fresh):
```javascript
{
  kernelAction: 'reset',
  cellInfo: { filePath: 'example.cell.py', isCellFile: true }
}
```

#### Check Status
Query if the kernel is currently executing:
```javascript
{
  kernelAction: 'status',
  cellInfo: { filePath: 'example.cell.py', isCellFile: true }
}
// Returns: { busy: true/false }
```

### 5. Execution Response

Execution results now include additional metadata:

```javascript
{
  output: "...",           // stdout output
  stderr: "...",           // stderr output (if any)
  interrupted: false,      // true if execution was interrupted
  timedOut: false         // true if execution exceeded timeout
}
```

## Architecture

### Kernel Lifecycle

1. **Creation**: First cell execution in a file creates a persistent Python process
2. **Reuse**: Subsequent cells in the same file use the same process
3. **Isolation**: Different files have separate kernel processes
4. **Cleanup**: Kernels can be stopped/reset manually or will be cleaned up on service restart

### Kernel Identification

Kernels are identified by: `${sourceDir}:${filePath}`

Example: `/app/workspace:example.cell.py`

### Signal Handling

The kernel uses Python's `signal` module to handle interrupts:
- SIGINT (Ctrl+C) is caught during cell execution
- Raises `KeyboardInterrupt` which can be caught by user code
- Signal handler is temporarily installed during execution and restored after

## Implementation Details

### Files Modified

- **`python-kernel.js`**: New module managing persistent Python processes
- **`execute.js`**: Updated to use persistent kernels for `.cell.py` files

### Python Wrapper

The kernel runs a Python wrapper script that:
- Maintains a persistent global namespace
- Reads cell code from stdin
- Executes code with `exec(code, globals())`
- Handles signals for interruption
- Emits delimiters to separate cell outputs

### Non-Cell Files

Regular `.py` files (not `.cell.py`) continue to use the original execution model:
- Each execution spawns a fresh Python process
- No persistent context between executions
- Ensures isolation for standalone scripts

## Testing

### Test Files

1. **`test_persistent_context.cell.py`**: Tests context persistence
   - Imports, variables, functions, classes across cells

2. **`test_interrupt.cell.py`**: Tests interrupt functionality
   - Quick execution
   - Long-running cells (can be interrupted)
   - Infinite loops (must be interrupted)
   - Context preservation after interruption
   - Timeout behavior

### Running Tests

Execute cells in the test files through your application interface:
- Run cells sequentially to verify context persistence
- Try interrupting long-running cells
- Set timeouts on slow cells
- Verify that context persists after interruptions

## Best Practices

1. **Use `.cell.py` extension** for notebook-style files that need persistent context
2. **Use regular `.py` extension** for standalone scripts that should run in isolation
3. **Set reasonable timeouts** for cells that might run indefinitely
4. **Provide interrupt UI** so users can stop long-running cells
5. **Reset kernels** when you want to start with a clean state
6. **Handle KeyboardInterrupt** in your Python code if you need cleanup after interruption

## Limitations

1. **Per-file isolation**: Cells in different files don't share context
2. **Execution order**: Running cells out of order can lead to undefined behavior
3. **Memory**: Long-running kernels accumulate state; reset if memory is a concern
4. **Restart required**: Modifying imported modules requires kernel reset
5. **SIGINT limitations**: Some operations (C extensions, system calls) may not be interruptible

## Future Enhancements

Possible improvements:
- Kernel memory usage monitoring
- Automatic kernel restart on errors
- Cell execution history/replay
- Variable inspector
- Execution time tracking
- Resource usage limits
