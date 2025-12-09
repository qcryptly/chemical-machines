import { ref, onUnmounted } from 'vue';

export function useWebSocket(path = '/ws') {
  const data = ref(null);
  const status = ref('CLOSED');
  const error = ref(null);

  let socket = null;
  let reconnectTimeout = null;
  const listeners = new Map();

  function getWebSocketUrl() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${protocol}//${window.location.host}${path}`;
  }

  function connect() {
    if (socket?.readyState === WebSocket.OPEN) return;

    status.value = 'CONNECTING';
    socket = new WebSocket(getWebSocketUrl());

    socket.onopen = () => {
      status.value = 'OPEN';
      error.value = null;
    };

    socket.onmessage = (event) => {
      try {
        const parsed = JSON.parse(event.data);
        data.value = parsed;

        // Notify type-specific listeners
        if (parsed.type && listeners.has(parsed.type)) {
          listeners.get(parsed.type).forEach(cb => cb(parsed));
        }
      } catch (e) {
        data.value = event.data;
      }
    };

    socket.onerror = (e) => {
      error.value = e;
      status.value = 'ERROR';
    };

    socket.onclose = () => {
      status.value = 'CLOSED';
      // Auto-reconnect after 3 seconds
      reconnectTimeout = setTimeout(connect, 3000);
    };
  }

  function disconnect() {
    if (reconnectTimeout) {
      clearTimeout(reconnectTimeout);
      reconnectTimeout = null;
    }
    if (socket) {
      socket.close();
      socket = null;
    }
  }

  function send(message) {
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(typeof message === 'string' ? message : JSON.stringify(message));
      return true;
    }
    return false;
  }

  function on(type, callback) {
    if (!listeners.has(type)) {
      listeners.set(type, new Set());
    }
    listeners.get(type).add(callback);

    // Return unsubscribe function
    return () => listeners.get(type).delete(callback);
  }

  onUnmounted(() => {
    disconnect();
  });

  return {
    data,
    status,
    error,
    connect,
    disconnect,
    send,
    on
  };
}
