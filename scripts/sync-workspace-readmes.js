#!/usr/bin/env node

/**
 * Sync README.md to Existing Workspaces
 *
 * Copies the library reference README.md from workspaces/ to all existing
 * workspace directories that don't already have one.
 *
 * Usage:
 *   node scripts/sync-workspace-readmes.js
 */

const fs = require('fs').promises;
const path = require('path');

const WORKSPACE_DIR = process.env.WORKSPACE_DIR || path.join(__dirname, '../workspace');
const README_SOURCE = path.join(__dirname, '../workspaces/README.md');

async function syncReadmes() {
  console.log('Syncing README.md to workspaces...');

  try {
    // Check if source README exists
    await fs.access(README_SOURCE);
  } catch (error) {
    console.error('Error: README.md not found at', README_SOURCE);
    process.exit(1);
  }

  try {
    // Get all workspace directories
    const entries = await fs.readdir(WORKSPACE_DIR, { withFileTypes: true });

    let copiedCount = 0;
    let skippedCount = 0;

    for (const entry of entries) {
      if (!entry.isDirectory() || entry.name.startsWith('.')) {
        continue;
      }

      const workspaceId = entry.name;
      const workspacePath = path.join(WORKSPACE_DIR, workspaceId);
      const readmePath = path.join(workspacePath, 'README.md');

      // Check if README already exists
      try {
        await fs.access(readmePath);
        console.log(`  ✓ Workspace ${workspaceId}: README.md already exists`);
        skippedCount++;
      } catch {
        // README doesn't exist, copy it
        try {
          await fs.copyFile(README_SOURCE, readmePath);
          console.log(`  + Workspace ${workspaceId}: Copied README.md`);
          copiedCount++;
        } catch (error) {
          console.error(`  ✗ Workspace ${workspaceId}: Failed to copy README.md:`, error.message);
        }
      }
    }

    console.log('\nSync complete!');
    console.log(`  Copied: ${copiedCount}`);
    console.log(`  Skipped (already exists): ${skippedCount}`);
    console.log(`  Total workspaces: ${copiedCount + skippedCount}`);

  } catch (error) {
    console.error('Error reading workspace directory:', error);
    process.exit(1);
  }
}

// Run
syncReadmes().catch(error => {
  console.error('Fatal error:', error);
  process.exit(1);
});
