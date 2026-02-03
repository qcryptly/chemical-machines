/**
 * Layout Tree Composable
 *
 * Manages a recursive tree structure for VS Code-style split editor panes.
 * Each internal node is a split (direction + children + sizes).
 * Each leaf node is an editor group (tabs + activeTabIndex).
 */

import { ref, computed } from 'vue'

/**
 * @typedef {Object} LeafNode
 * @property {'leaf'} type
 * @property {number} id
 * @property {Array} tabs
 * @property {number} activeTabIndex
 */

/**
 * @typedef {Object} SplitNode
 * @property {'split'} type
 * @property {number} id
 * @property {'horizontal'|'vertical'} direction
 * @property {Array<LayoutNode>} children
 * @property {Array<number>} sizes - percentages, e.g. [50, 50]
 */

/**
 * @typedef {LeafNode|SplitNode} LayoutNode
 */

export function useLayoutTree() {
  let _nextId = 1

  const root = ref({
    type: 'leaf',
    id: 0,
    tabs: [],
    activeTabIndex: -1
  })

  const focusedLeafId = ref(0)

  // --- Tree traversal helpers ---

  /**
   * Recursively find a node by ID
   * @param {LayoutNode} node
   * @param {number} id
   * @returns {LayoutNode|null}
   */
  function findNode(node, id) {
    if (node.id === id) return node
    if (node.type === 'split') {
      for (const child of node.children) {
        const found = findNode(child, id)
        if (found) return found
      }
    }
    return null
  }

  /**
   * Find the parent of a node by ID
   * @param {LayoutNode} node - Current node to search
   * @param {number} id - Target node ID
   * @returns {{ parent: SplitNode, childIndex: number }|null}
   */
  function findParent(node, id) {
    if (node.type !== 'split') return null
    for (let i = 0; i < node.children.length; i++) {
      if (node.children[i].id === id) {
        return { parent: node, childIndex: i }
      }
      const found = findParent(node.children[i], id)
      if (found) return found
    }
    return null
  }

  /**
   * Get all leaf nodes in tree order (depth-first)
   * @param {LayoutNode} [node]
   * @returns {LeafNode[]}
   */
  function getAllLeaves(node) {
    if (!node) node = root.value
    if (node.type === 'leaf') return [node]
    const leaves = []
    for (const child of node.children) {
      leaves.push(...getAllLeaves(child))
    }
    return leaves
  }

  /**
   * Find a leaf node by ID
   * @param {number} id
   * @returns {LeafNode|null}
   */
  function findLeafById(id) {
    const node = findNode(root.value, id)
    return (node && node.type === 'leaf') ? node : null
  }

  /** The currently focused leaf node */
  const focusedLeaf = computed(() => findLeafById(focusedLeafId.value))

  // --- Tree mutations ---

  /**
   * Split a leaf node into a split containing [original leaf, new empty leaf]
   * @param {number} leafId - ID of the leaf to split
   * @param {'horizontal'|'vertical'} direction
   * @returns {number|null} ID of the new empty leaf, or null on failure
   */
  function splitLeaf(leafId, direction) {
    const leaf = findLeafById(leafId)
    if (!leaf) return null

    const newLeaf = {
      type: 'leaf',
      id: _nextId++,
      tabs: [],
      activeTabIndex: -1
    }

    const newSplit = {
      type: 'split',
      id: _nextId++,
      direction,
      children: [leaf, newLeaf],
      sizes: [50, 50]
    }

    const parentInfo = findParent(root.value, leafId)
    if (!parentInfo) {
      // leaf is root
      root.value = newSplit
    } else {
      parentInfo.parent.children[parentInfo.childIndex] = newSplit
    }

    return newLeaf.id
  }

  /**
   * Remove a leaf node and collapse its parent if only 1 child remains.
   * Cannot remove the last remaining leaf (root leaf).
   * @param {number} leafId
   */
  function removeLeaf(leafId) {
    // Can't remove the last pane
    if (root.value.type === 'leaf') return

    const parentInfo = findParent(root.value, leafId)
    if (!parentInfo) return

    const { parent, childIndex } = parentInfo

    // Remove the leaf
    parent.children.splice(childIndex, 1)
    parent.sizes.splice(childIndex, 1)

    // Normalize remaining sizes to sum to 100
    const total = parent.sizes.reduce((a, b) => a + b, 0)
    if (total > 0) {
      parent.sizes = parent.sizes.map(s => (s / total) * 100)
    }

    // If parent has only one child left, collapse by promoting survivor
    if (parent.children.length === 1) {
      const survivor = parent.children[0]
      const grandparentInfo = findParent(root.value, parent.id)
      if (!grandparentInfo) {
        // parent is root
        root.value = survivor
      } else {
        grandparentInfo.parent.children[grandparentInfo.childIndex] = survivor
      }
    }

    // Fix focused leaf if removed
    if (focusedLeafId.value === leafId) {
      const leaves = getAllLeaves()
      focusedLeafId.value = leaves.length > 0 ? leaves[0].id : 0
    }
  }

  /**
   * Update the sizes array of a split node
   * @param {number} splitId
   * @param {number[]} newSizes
   */
  function updateSizes(splitId, newSizes) {
    const node = findNode(root.value, splitId)
    if (node && node.type === 'split') {
      node.sizes = newSizes
    }
  }

  /**
   * Move a tab from one leaf to another.
   * Auto-removes the source leaf if it becomes empty.
   * @param {number} fromLeafId
   * @param {number} fromTabIndex
   * @param {number} toLeafId
   * @param {number} toTabIndex
   */
  function moveTab(fromLeafId, fromTabIndex, toLeafId, toTabIndex) {
    const fromLeaf = findLeafById(fromLeafId)
    const toLeaf = findLeafById(toLeafId)
    if (!fromLeaf || !toLeaf) return
    if (fromTabIndex < 0 || fromTabIndex >= fromLeaf.tabs.length) return

    // Remove tab from source
    const [tab] = fromLeaf.tabs.splice(fromTabIndex, 1)

    // Adjust source active index
    if (fromLeaf.tabs.length === 0) {
      fromLeaf.activeTabIndex = -1
    } else if (fromTabIndex <= fromLeaf.activeTabIndex) {
      fromLeaf.activeTabIndex = Math.max(0, fromLeaf.activeTabIndex - 1)
    }

    // Insert into destination
    const insertAt = Math.min(Math.max(0, toTabIndex), toLeaf.tabs.length)
    toLeaf.tabs.splice(insertAt, 0, tab)
    toLeaf.activeTabIndex = insertAt
    focusedLeafId.value = toLeafId

    // Auto-remove empty source leaf
    if (fromLeaf.tabs.length === 0) {
      removeLeaf(fromLeafId)
    }
  }

  return {
    root,
    focusedLeafId,
    focusedLeaf,
    findNode,
    findParent,
    getAllLeaves,
    findLeafById,
    splitLeaf,
    removeLeaf,
    updateSizes,
    moveTab
  }
}
