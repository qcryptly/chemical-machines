/**
 * Centralized Document Store
 *
 * Maintains a single source of truth for each file's content (keyed by path).
 * Multiple tabs can reference the same document, and edits sync automatically
 * via Vue's reactivity system.
 */

import { reactive } from 'vue'

// Map<path, Document> - reactive so Vue tracks changes
const documents = reactive(new Map())

// Reference counting to know when to clean up
const refCounts = new Map()

/**
 * Open a document (or get existing one if already open)
 * @param {string} path - File path (used as key)
 * @param {Array} cells - Parsed cells array
 * @param {Object} options - { language, useCells, isMarkdown }
 * @returns {Object} Reactive document object
 */
export function openDocument(path, cells, options = {}) {
  if (documents.has(path)) {
    // Document already open - increment refcount and return existing
    refCounts.set(path, (refCounts.get(path) || 0) + 1)
    return documents.get(path)
  }

  // Create new reactive document
  const document = reactive({
    path,
    cells,
    htmlOutputs: options.htmlOutputs || [],
    language: options.language || 'python',
    useCells: options.useCells !== false,
    isMarkdown: options.isMarkdown || false,
    isDirty: false
  })

  documents.set(path, document)
  refCounts.set(path, 1)
  return document
}

/**
 * Get a document by path (returns null if not open)
 * @param {string} path
 * @returns {Object|null}
 */
export function getDocument(path) {
  return documents.get(path) || null
}

/**
 * Check if a document is open
 * @param {string} path
 * @returns {boolean}
 */
export function hasDocument(path) {
  return documents.has(path)
}

/**
 * Update a cell's content and/or metadata
 * @param {string} path
 * @param {number} cellIndex
 * @param {Object} data - { content, language, environment, ... }
 */
export function updateCell(path, cellIndex, data) {
  const doc = documents.get(path)
  if (!doc || !doc.cells[cellIndex]) return

  const cell = doc.cells[cellIndex]

  // Update content (marks dirty if changed)
  if (data.content !== undefined && data.content !== cell.content) {
    cell.content = data.content
    doc.isDirty = true
  }

  // Update other cell properties (don't mark dirty for metadata)
  if (data.language !== undefined) cell.language = data.language
  if (data.environment !== undefined) cell.environment = data.environment
  if (data.cppEnvironment !== undefined) cell.cppEnvironment = data.cppEnvironment
  if (data.vendorEnvironment !== undefined) cell.vendorEnvironment = data.vendorEnvironment
  if (data.compiler !== undefined) cell.compiler = data.compiler
  if (data.cppStandard !== undefined) cell.cppStandard = data.cppStandard
  if (data.output !== undefined) cell.output = data.output
  if (data.status !== undefined) cell.status = data.status
  if (data.title !== undefined) cell.title = data.title
}

/**
 * Add a new cell to a document
 * @param {string} path
 * @param {number} afterIndex - Insert after this index (-1 for end)
 * @param {Object} cellData - Cell properties
 * @returns {Object|null} The new cell
 */
export function addCell(path, afterIndex = -1, cellData = {}) {
  const doc = documents.get(path)
  if (!doc) return null

  const newCell = {
    id: Date.now(),
    type: cellData.type || 'code',
    language: cellData.language || doc.language,
    environment: cellData.environment || '',
    content: cellData.content || '',
    title: cellData.title || '',
    output: null,
    status: null,
    ...cellData
  }

  const insertAt = afterIndex >= 0 ? afterIndex + 1 : doc.cells.length
  doc.cells.splice(insertAt, 0, newCell)
  // Keep htmlOutputs aligned with cells
  if (doc.htmlOutputs) {
    doc.htmlOutputs.splice(insertAt, 0, '')
  }
  doc.isDirty = true

  return newCell
}

/**
 * Delete a cell from a document
 * @param {string} path
 * @param {number} cellIndex
 */
export function deleteCell(path, cellIndex) {
  const doc = documents.get(path)
  if (!doc || !doc.cells[cellIndex]) return

  doc.cells.splice(cellIndex, 1)
  // Keep htmlOutputs aligned with cells
  if (doc.htmlOutputs) {
    doc.htmlOutputs.splice(cellIndex, 1)
  }
  doc.isDirty = true

  // Ensure at least one cell remains
  if (doc.cells.length === 0) {
    addCell(path, -1, { language: doc.language })
  }
}

/**
 * Reorder cells within a document
 * @param {string} path
 * @param {number} fromIndex
 * @param {number} toIndex
 */
export function reorderCells(path, fromIndex, toIndex) {
  const doc = documents.get(path)
  if (!doc || fromIndex === toIndex) return

  const [movedCell] = doc.cells.splice(fromIndex, 1)
  doc.cells.splice(toIndex, 0, movedCell)
  // Keep htmlOutputs aligned with cells
  if (doc.htmlOutputs && doc.htmlOutputs.length > fromIndex) {
    const [movedOutput] = doc.htmlOutputs.splice(fromIndex, 1)
    doc.htmlOutputs.splice(toIndex, 0, movedOutput)
  }
  doc.isDirty = true
}

/**
 * Set HTML outputs for a document (from .out/ file refresh)
 * @param {string} path
 * @param {Array} outputs - Array of HTML strings, one per cell
 */
export function setHtmlOutputs(path, outputs) {
  const doc = documents.get(path)
  if (doc) {
    doc.htmlOutputs = outputs || []
  }
}

/**
 * Mark a document as clean (called after save)
 * @param {string} path
 */
export function markClean(path) {
  const doc = documents.get(path)
  if (doc) doc.isDirty = false
}

/**
 * Mark a document as dirty
 * @param {string} path
 */
export function markDirty(path) {
  const doc = documents.get(path)
  if (doc) doc.isDirty = true
}

/**
 * Close a document (decrements refcount, removes when 0)
 * @param {string} path
 */
export function closeDocument(path) {
  const count = refCounts.get(path) || 0
  if (count <= 1) {
    documents.delete(path)
    refCounts.delete(path)
  } else {
    refCounts.set(path, count - 1)
  }
}

/**
 * Get the reference count for a document
 * @param {string} path
 * @returns {number}
 */
export function getRefCount(path) {
  return refCounts.get(path) || 0
}

/**
 * Get all open document paths
 * @returns {string[]}
 */
export function getOpenDocuments() {
  return Array.from(documents.keys())
}

// Export the documents map for debugging
export { documents }
