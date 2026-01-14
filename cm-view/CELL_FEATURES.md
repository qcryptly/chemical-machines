# Cell Features Documentation

## Overview

The workspace view now supports enhanced cell manipulation features including drag-and-drop reordering and quick cell creation.

## Features

### 1. Drag-and-Drop Cell Reordering

Cells can be reordered by dragging and dropping them to a new position.

**How to use:**
1. Click and hold on any cell (or specifically on the drag handle `⋮⋮`)
2. Drag the cell up or down
3. Drop it at the desired position
4. The cell will be inserted at the new location

**Visual feedback:**
- While dragging, the cell becomes semi-transparent (50% opacity)
- When hovering over a valid drop target, the target cell highlights with a green border
- The target cell scales up slightly (1.02x) to indicate it's ready to receive the drop
- Cursor changes to `grab` when hovering and `grabbing` when dragging

**Behavior:**
- Cell numbering automatically updates after reordering
- File is marked as unsaved after reordering
- Cell IDs are preserved during reordering
- Output and execution state are preserved

### 2. Create Cell Below

Quickly create a new cell immediately below the current cell.

**How to use:**

**Option 1: Button**
- Click the `+` button in the cell toolbar
- A new cell will be created directly below the current cell

**Option 2: Keyboard shortcut**
- Press `Alt+Enter` while editing a cell
- A new cell will be created directly below the current cell

**Behavior:**
- New cell inherits the language from the cell above
- New cell inherits the environment settings from the cell above
- Focus can be manually moved to the new cell
- File is marked as unsaved after creation

### 3. Cell Toolbar Layout

The cell toolbar now includes:

```
[⋮⋮] [1] [                    ] [+] [▶] [×]
 │    │   └─ Spacer             │   │   └─ Delete cell
 │    │                         │   └─ Run cell (Ctrl/Cmd+Enter)
 │    │                         └─ Create cell below (Alt+Enter)
 │    └─ Cell number
 └─ Drag handle
```

**Elements:**
- **Drag handle (`⋮⋮`)**: Visual indicator for drag-and-drop functionality
- **Cell number**: Shows the current position (e.g., `[1]`, `[2]`)
- **Spacer**: Flexible space that pushes action buttons to the right
- **Create below (`+`)**: Creates a new cell below the current one
- **Run (`▶`)**: Executes the cell code
- **Delete (`×`)**: Removes the cell

## Implementation Details

### CodeCell Component Changes

**New Props:**
- No new props required

**New Events:**
- `@create-below`: Emitted when user requests to create a cell below
- `@reorder`: Emitted with `{ fromIndex, toIndex }` when cell is dropped

**New State:**
- `isDragOver`: Boolean indicating if a cell is being dragged over

**Drag Handlers:**
- `handleDragStart`: Initiates drag operation, stores cell index
- `handleDragEnd`: Cleans up after drag operation
- `handleDragOver`: Handles drag hover state
- `handleDragLeave`: Clears hover state when leaving
- `handleDrop`: Completes the reorder operation

### WorkspaceView Component Changes

**New Functions:**
- `createCellBelow(index)`: Creates a new cell at position `index + 1`
- `reorderCells({ fromIndex, toIndex })`: Moves cell from one position to another

**Logic:**
- Both operations mark the file as having unsaved changes
- Cell IDs are preserved during operations
- New cells inherit properties from adjacent cells

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl/Cmd+Enter` | Run current cell |
| `Alt+Enter` | Create cell below current cell |
| `Backspace/Delete` | Delete cell (via delete button) |

### CSS Changes

**Cell styles:**
- Added `cursor: grab` for draggable cells
- Added `cursor: grabbing` when actively dragging
- Added `.drag-over` state with green border and scale effect
- Added smooth transitions for visual feedback

**New elements:**
- `.drag-handle`: Styled grip icon with hover opacity effect
- `.create-below-btn`: Plus button with border and hover state

## Browser Compatibility

The drag-and-drop functionality uses the HTML5 Drag and Drop API, which is supported in:
- Chrome/Edge: Full support
- Firefox: Full support
- Safari: Full support

## Future Enhancements

Possible improvements:
- **Multi-select**: Drag multiple cells at once
- **Copy/paste cells**: Duplicate cells with content
- **Cell folding**: Collapse cells to save screen space
- **Cell tags**: Add labels/categories to cells
- **Keyboard-only reordering**: Use keyboard shortcuts for accessibility
- **Undo/redo**: History for cell operations
- **Drag preview**: Show ghost image of cell being dragged
- **Drop zones**: Show explicit drop zones between cells
- **Touch support**: Mobile drag-and-drop

## Testing

### Manual Test Cases

**Drag-and-Drop:**
1. ✓ Drag cell from top to bottom
2. ✓ Drag cell from bottom to top
3. ✓ Drag cell to adjacent position
4. ✓ Drag cell across multiple positions
5. ✓ Cancel drag by dropping outside cells
6. ✓ Visual feedback during drag
7. ✓ Cell numbering updates correctly
8. ✓ File marked as unsaved

**Create Cell Below:**
1. ✓ Click + button creates cell below
2. ✓ Alt+Enter creates cell below
3. ✓ New cell inherits language
4. ✓ New cell inherits environment
5. ✓ Create cell below last cell
6. ✓ Create cell below first cell
7. ✓ File marked as unsaved

**Edge Cases:**
1. ✓ Drag cell onto itself (no-op)
2. ✓ Single cell in file (can't delete)
3. ✓ Rapid successive drags
4. ✓ Create multiple cells quickly
5. ✓ Reorder while cell is executing
6. ✓ Reorder cells with output

## Known Limitations

1. **No animation**: Cells snap to new position instantly
2. **No preview**: No visual preview of where cell will land
3. **Context preservation**: Cell context (imports, variables) follows execution order, not visual order
4. **Touch devices**: Limited support for touch-based drag-and-drop

## Best Practices

1. **Execution order**: Remember that cells share context based on execution order, not visual order
2. **After reordering**: Re-run cells if they depend on execution order
3. **Save frequently**: Reordering marks file as unsaved
4. **Visual organization**: Use cell titles (in `# %%` comments) for better organization
5. **Testing**: After reordering, test cell execution to ensure dependencies are maintained
