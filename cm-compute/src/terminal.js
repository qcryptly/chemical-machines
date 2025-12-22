/**
 * Terminal Session Manager
 * Manages PTY sessions for workspace terminals with chroot-like isolation
 */

const pty = require('node-pty');
const path = require('path');
const fs = require('fs');
const os = require('os');

// Active terminal sessions: sessionId -> { pty, workspaceId, emit }
const sessions = new Map();

// Workspace directory base
const WORKSPACE_DIR = process.env.WORKSPACE_DIR || '/app/workspace';

/**
 * Create a new terminal session for a workspace
 * @param {string} workspaceId - Workspace ID
 * @param {Function} emit - Function to emit output to client
 * @returns {string} Session ID
 */
function createSession(workspaceId, emit) {
  const sessionId = `term-${workspaceId}-${Date.now()}`;
  const workspaceDir = path.join(WORKSPACE_DIR, String(workspaceId));

  // Ensure workspace directory exists
  if (!fs.existsSync(workspaceDir)) {
    fs.mkdirSync(workspaceDir, { recursive: true });
  }

  // Determine shell
  const shell = process.env.SHELL || '/bin/bash';

  // Create PTY with workspace as cwd
  const ptyProcess = pty.spawn(shell, [], {
    name: 'xterm-256color',
    cols: 80,
    rows: 24,
    cwd: workspaceDir,
    env: {
      ...process.env,
      TERM: 'xterm-256color',
      HOME: workspaceDir,
      PWD: workspaceDir,
      WORKSPACE_ID: String(workspaceId),
      // Set PS1 for a nice prompt
      PS1: '\\[\\033[36m\\]workspace:\\[\\033[33m\\]\\w\\[\\033[0m\\]$ ',
      // Conda activation
      PATH: `${process.env.CONDA_PATH || '/opt/conda'}/bin:${process.env.PATH}`,
    }
  });

  // Track session
  sessions.set(sessionId, {
    pty: ptyProcess,
    workspaceId,
    emit,
    lastActivity: Date.now()
  });

  // Handle PTY output
  ptyProcess.onData((data) => {
    emit('terminal_output', { sessionId, data });
  });

  // Handle PTY exit
  ptyProcess.onExit(({ exitCode }) => {
    emit('terminal_exit', { sessionId, code: exitCode });
    sessions.delete(sessionId);
  });

  // Send initial setup commands
  // Activate conda base environment
  ptyProcess.write('source /opt/conda/etc/profile.d/conda.sh 2>/dev/null\n');
  ptyProcess.write('clear\n');

  return sessionId;
}

/**
 * Write data to a terminal session
 * @param {string} sessionId - Session ID
 * @param {string} data - Input data
 */
function writeToSession(sessionId, data) {
  const session = sessions.get(sessionId);
  if (session) {
    session.pty.write(data);
    session.lastActivity = Date.now();
  }
}

/**
 * Resize a terminal session
 * @param {string} sessionId - Session ID
 * @param {number} cols - Number of columns
 * @param {number} rows - Number of rows
 */
function resizeSession(sessionId, cols, rows) {
  const session = sessions.get(sessionId);
  if (session) {
    session.pty.resize(cols, rows);
  }
}

/**
 * Destroy a terminal session
 * @param {string} sessionId - Session ID
 */
function destroySession(sessionId) {
  const session = sessions.get(sessionId);
  if (session) {
    session.pty.kill();
    sessions.delete(sessionId);
  }
}

/**
 * Get session info
 * @param {string} sessionId - Session ID
 * @returns {Object|null} Session info or null
 */
function getSession(sessionId) {
  const session = sessions.get(sessionId);
  if (session) {
    return {
      sessionId,
      workspaceId: session.workspaceId,
      lastActivity: session.lastActivity
    };
  }
  return null;
}

/**
 * Get all active sessions for a workspace
 * @param {string} workspaceId - Workspace ID
 * @returns {string[]} Session IDs
 */
function getWorkspaceSessions(workspaceId) {
  const result = [];
  for (const [sessionId, session] of sessions) {
    if (String(session.workspaceId) === String(workspaceId)) {
      result.push(sessionId);
    }
  }
  return result;
}

/**
 * Clean up inactive sessions (older than 30 minutes)
 */
function cleanupInactiveSessions() {
  const timeout = 30 * 60 * 1000; // 30 minutes
  const now = Date.now();

  for (const [sessionId, session] of sessions) {
    if (now - session.lastActivity > timeout) {
      console.log(`Cleaning up inactive terminal session: ${sessionId}`);
      destroySession(sessionId);
    }
  }
}

// Run cleanup every 5 minutes
setInterval(cleanupInactiveSessions, 5 * 60 * 1000);

module.exports = {
  createSession,
  writeToSession,
  resizeSession,
  destroySession,
  getSession,
  getWorkspaceSessions
};
