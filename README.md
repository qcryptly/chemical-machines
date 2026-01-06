# Chemical Machines

A no-bullshit biophysics drug design platform running in Docker.

## Overview

Chemical Machines is a containerized platform for computational drug design that combines molecular simulation, database management, and an interactive web-based interface. Built for researchers who need serious compute power without the deployment headaches.

## TLDR; Quickstart

    docker compose up -d --build

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

## Python Library (cm)

The `cm` Python library provides symbolic mathematics and quantum mechanics tools for use in notebooks.

### Quick Start

```python
from cm.symbols import Math

# Create variables
x = Math.var("x")
n = Math.var("n")

# Basic expressions
expr = x**2 + 2*x + 1
expr.render()              # Display LaTeX
expr.evaluate(x=3)         # = 16

# Calculus
expr.diff(x).render()      # Derivative: 2x + 2
expr.integrate(x, bounds=[0, 1]).evaluate()  # Definite integral
```

### Special Functions

Access 50+ special functions commonly used in physics:

```python
# Gamma and factorials
Math.gamma(5).evaluate()           # Γ(5) = 24
Math.factorial(5).evaluate()       # 5! = 120
Math.binomial(10, 3).evaluate()    # C(10,3) = 120

# Bessel functions
Math.besselj(0, 2.4).evaluate()    # J₀(2.4)
Math.bessely(1, x).render()        # Y₁(x)

# Orthogonal polynomials
Math.hermite(n, x).render()        # Hₙ(x) - Hermite
Math.legendre(2, 0.5).evaluate()   # P₂(0.5) = -0.125
Math.laguerre(n, x).render()       # Lₙ(x) - Laguerre
Math.assoc_laguerre(n, 2, x)       # L_n^(2)(x) - Associated Laguerre

# Spherical harmonics
theta, phi = Math.var("theta"), Math.var("phi")
Math.Ylm(2, 1, theta, phi).render()  # Y₂¹(θ,φ)

# Error functions
Math.erf(1).evaluate()             # erf(1) ≈ 0.8427

# Hypergeometric
Math.hyper2f1(a, b, c, z)          # ₂F₁(a,b;c;z)
```

**Available function categories:**
- **Gamma**: `gamma`, `loggamma`, `digamma`, `beta`, `factorial`, `factorial2`, `binomial`
- **Error**: `erf`, `erfc`, `erfi`
- **Bessel**: `besselj`, `bessely`, `besseli`, `besselk`, `jn`, `yn`, `hankel1`, `hankel2`
- **Airy**: `airyai`, `airybi`, `airyaiprime`, `airybiprime`
- **Polynomials**: `legendre`, `assoc_legendre`, `hermite`, `laguerre`, `assoc_laguerre`, `chebyshevt`, `chebyshevu`, `gegenbauer`, `jacobi`
- **Spherical**: `Ylm`, `Ylm_real`
- **Hypergeometric**: `hyper2f1`, `hyper1f1`, `hyper0f1`, `hyperpfq`
- **Elliptic**: `elliptic_k`, `elliptic_e`, `elliptic_pi`
- **Other**: `zeta`, `polylog`, `dirac`, `heaviside`, `kronecker`, `levi_civita`

### Custom Functions with Hyperparameters

Define reusable symbolic functions:

```python
from cm.symbols import Math, Scalar

# Define a function with typed hyperparameters
a, b, x = Math.var("a"), Math.var("b"), Math.var("x")
f = Math.function(a * Math.exp(b * x), hyperparams={"a": Scalar, "b": Scalar})
f.save("MyExponential")

# Retrieve and instantiate
func = Math.get_function("MyExponential")
inst = func.init(a=10, b=0.5)
inst.render()           # Shows: 10·e^(0.5x)
inst.run(x=2)           # Evaluates numerically
```

### PyTorch Compilation

Compile expressions to GPU-accelerated PyTorch functions:

```python
# Create and compile
expr = Math.sin(x) * Math.exp(-x**2)
cg = inst.run_with(x=1.0)
torch_fn = cg.compile(device='cuda')

# Evaluate on GPU
result = torch_fn(x=torch.linspace(0, 10, 1000))

# Automatic differentiation
grad_fn = torch_fn.grad()
gradients = grad_fn(x=1.0)
```

### Quantum Mechanics (cm.qm)

Tools for many-electron quantum mechanics:

```python
from cm.qm import SpinOrbital, SlaterDeterminant

# Define spin-orbitals
orbitals = [
    SpinOrbital("1s", spin="alpha"),
    SpinOrbital("1s", spin="beta"),
    SpinOrbital("2s", spin="alpha"),
]

# Create Slater determinant
det = SlaterDeterminant(orbitals)
det.render()  # Display in bra-ket notation
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
