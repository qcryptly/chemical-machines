# Chemical Machines

A no-bullshit biophysics drug design platform running in Docker.

## Overview

Chemical Machines is a containerized platform for computational drug design that combines molecular simulation, database management, and an interactive web-based interface. Built for researchers who need serious compute power without the deployment headaches.

## Architecture

The platform consists of four main components orchestrated by supervisord:

### 1. **Elasticsearch**
- Caching layer for autocomplete and string search
- Fast query performance for molecular structures and annotations
- Port: 9200

### 2. **PostgreSQL**
- Primary data persistence layer
- Stores experimental results, molecular structures, and metadata
- Port: 5432

### 3. **cm-compute** (Compute Daemon)
- Express.js daemon process with Unix socket interface
- PyTorch + CUDA support for GPU-accelerated molecular dynamics
- C++ bindings for performance-critical calculations
- Queued request processing via process forking
- Socket: `/var/run/cm-compute.sock`

### 4. **cm-view** (Web Interface)
- Vue.js + Express.js interactive workspace
- Jupyter-like notebook experience with WebGL visualization
- Direct connections to PostgreSQL, Elasticsearch, and cm-compute
- Port: 3000

## Prerequisites

- Docker with GPU support (nvidia-docker2)
- CUDA-capable GPU (for compute functionality)
- 16GB+ RAM recommended
- Linux host (tested on Ubuntu 22.04)

## Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd chemical-machines

# Build and run
docker-compose up -d

# Access the web interface
open http://localhost:3000
```

## Development

### Project Structure

```
chemical-machines/
├── cm-compute/          # Compute daemon (Express + PyTorch + C++)
│   ├── package.json
│   ├── src/
│   └── lib/             # C++ native modules
├── cm-view/             # Web interface (Vue.js + Express)
│   ├── package.json
│   ├── server/          # Express backend
│   └── client/          # Vue.js frontend
├── docker/
│   ├── Dockerfile
│   └── supervisord.d/   # Supervisor configuration
├── docker-compose.yml
└── README.md
```

### Building Components

Each component can be developed independently:

```bash
# cm-compute
cd cm-compute
npm install
npm run dev

# cm-view
cd cm-view
npm install
npm run dev
```

## Configuration

Environment variables can be set in `docker-compose.yml`:

- `POSTGRES_DB`: Database name (default: chemicalmachines)
- `POSTGRES_USER`: Database user (default: cmuser)
- `POSTGRES_PASSWORD`: Database password
- `ES_JAVA_OPTS`: Elasticsearch JVM options
- `COMPUTE_SOCKET_PATH`: Unix socket path for cm-compute

## License

See dual license in LICENSE.md

## Contributing

This is a research platform. Contributions welcome, but keep it focused and performant.
