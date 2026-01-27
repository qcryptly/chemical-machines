# Chemical Machines

An integrated computational chemistry platform combining symbolic mathematics, GPU-accelerated quantum mechanics, and molecular database integration for reproducible research in chemical physics and drug discovery.

Chemical Machines provides researchers and developers with a containerized environment for performing quantum mechanical calculations, symbolic analysis, and benchmark validation against experimental data. The platform features an interactive notebook interface, GPU acceleration via PyTorch, and integration with major molecular databases (NIST CCCBDB, PubChem, QM9).

## Key Features

- **GPU-accelerated quantum mechanics** - Hartree-Fock calculations for many-electron systems with CUDA support
- **Symbolic mathematics** - 50+ special functions including Bessel, Legendre, Hermite polynomials, and spherical harmonics
- **Interactive notebook interface** - Jupyter-like experience with persistent Python kernels and drag-and-drop cells
- **Integrated molecular databases** - 134k+ molecules from NIST CCCBDB, PubChem, and QM9 datasets
- **Real-time 3D visualization** - WebGL-powered molecular structure rendering
- **Docker-based deployment** - Single-command setup with full CUDA and database stack
- **Comprehensive test coverage** - 90+ tests across all modules ensuring scientific accuracy

## Scientific Capabilities

Chemical Machines addresses the following research problems in computational chemistry and chemical physics:

- **Atomic and molecular energy calculations** - Compute ground state energies using the Hartree-Fock approximation for atoms and small molecules
- **Electronic structure analysis** - Determine orbital energies, electron configurations, and construct Slater determinants for many-electron systems
- **Many-body quantum mechanics** - Build and evaluate multi-electron wavefunctions using Slater-Condon matrix elements
- **Symbolic mathematical analysis** - Perform analytical differentiation, integration, and manipulation of special functions commonly appearing in quantum mechanics
- **Benchmark validation** - Compare calculated molecular properties against experimental and high-level theoretical data from NIST and PubChem
- **Method development** - Implement and test new quantum mechanical approximations, basis sets, and computational approaches

### Computational Methods

The platform employs the following computational approaches:

- **Electronic structure**: Hartree-Fock approximation with Slater-type and Gaussian-type orbital basis functions
- **Special functions**: Direct numerical integration and series expansion for Bessel functions, orthogonal polynomials, and spherical harmonics
- **GPU acceleration**: PyTorch-based tensor operations enabling efficient evaluation of symbolic expressions on CUDA devices
- **Database queries**: Elasticsearch-backed full-text and property-based search across 134,000+ molecular structures

**Accuracy considerations**: Results from Hartree-Fock calculations typically achieve 85-95% of experimental correlation energies for small molecules. For publication-quality results,

 validation against experimental data or higher-level calculations (CCSD(T), full CI) is recommended.

## Quick Start

### Prerequisites

- Docker with GPU support ([nvidia-docker2](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))
- CUDA-capable GPU (optional but recommended for performance)
- 16GB+ RAM
- Linux host (tested on Ubuntu 22.04)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/chemical-machines
cd chemical-machines

# Build and start all services
docker compose up -d

# Monitor startup progress (services initialize in ~2-3 minutes)
docker compose logs -f chemical-machines
```

### Access the Platform

Once services are running:

- **Web Interface**: http://localhost:5173 (interactive notebooks)
- **Kibana**: http://localhost:5601 (Elasticsearch data browser)
- **PostgreSQL**: localhost:5432 (database access)
- **Elasticsearch**: localhost:9200 (search API)

## Python Library Overview

The `cm` Python library provides four integrated modules for computational chemistry research:

### cm.symbols

Symbolic mathematics engine with expression trees, automatic differentiation, and 50+ special functions.

- Create symbolic variables and expressions with standard arithmetic operations
- Compute analytical derivatives and integrals with respect to any variable
- Access special functions: Bessel (J, Y, I, K), orthogonal polynomials (Legendre, Hermite, Laguerre, Chebyshev), spherical harmonics (Y_l^m), Gamma, error functions, hypergeometric functions
- Render expressions as LaTeX for publication-quality output
- Compile symbolic expressions to PyTorch for GPU-accelerated evaluation

### cm.qm

Quantum mechanics toolkit for atomic and molecular calculations.

- Create atoms with automatic ground-state electron configurations (aufbau principle)
- Construct spin-orbitals and Slater determinants for many-electron wavefunctions
- Build molecular Hamiltonians with kinetic energy, nuclear attraction, and electron-electron repulsion terms
- Calculate matrix elements using Slater-Condon rules
- Define molecules with arbitrary geometries
- Support for relativistic calculations with Dirac spinors

### cm.data

Benchmark molecular database API providing programmatic access to experimental and computational reference data.

- Search across NIST CCCBDB, PubChem, and QM9 datasets by name, formula, or molecular properties
- Retrieve experimental geometries, energies, vibrational frequencies, and electronic properties
- Compare calculated values against benchmark data
- Access 134,000+ pre-computed molecular structures and properties

### cm.views

Visualization and output formatting for interactive research.

- Render 3D molecular structures with WebGL
- Display mathematical expressions with MathJax/LaTeX formatting
- Generate scatter plots and surface visualizations
- Output structured HTML for integration with notebooks

**Comprehensive documentation**: See [Library Reference](workspaces/README.md) for detailed API documentation and additional examples.

## Example Workflows

### Workflow 1: Calculating Atomic Energy

This example demonstrates computing the ground-state energy of a carbon atom using the Hartree-Fock approximation.

```python
from cm import qm

# Create carbon atom with automatic electron configuration
C = qm.atom('C')  # Automatically configures 1s² 2s² 2p²
print(C.configuration.label)  # Output: "1s² 2s² 2p²"

# Build electronic Hamiltonian with all relevant terms
H = qm.HamiltonianBuilder.electronic().build()

# Calculate ground state energy
E = C.energy(H).numerical()
print(f"Ground state energy: {E:.4f} Hartree")
# Output: -37.6886 Hartree (experimental: -37.845 Ha, within 4% agreement)
```

The Hartree-Fock energy is typically 85-95% of the exact correlation energy. The difference reflects electron correlation effects not captured in the mean-field approximation.

### Workflow 2: Symbolic Mathematics and Special Functions

This example shows symbolic manipulation of spherical harmonics, commonly used in quantum mechanics for angular momentum eigenstates.

```python
from cm import Math

# Create symbolic expression with spherical harmonics
theta, phi = Math.var("theta"), Math.var("phi")
Y_21 = Math.Ylm(2, 1, theta, phi)  # Y₂¹(θ,φ) spherical harmonic

# Render LaTeX for publication
Y_21.render()  # Displays beautiful MathJax-rendered equation

# Numerical evaluation at specific angles
result = Y_21.evaluate(theta=1.57, phi=0.785)
print(f"Y₂¹(π/2, π/4) = {result}")

# Verify orthogonality using numerical integration
from cm.qm import spherical_harmonic_orthogonality
overlap = spherical_harmonic_orthogonality(2, 1, 2, 0)  # Different m quantum number
print(f"Orthogonality check: {overlap:.6f}")  # Should be ≈ 0
```

Spherical harmonics form an orthonormal basis on the unit sphere, essential for expanding angular wavefunctions in atomic and molecular systems.

### Workflow 3: Molecular Hamiltonian and CI Matrix

This example demonstrates building a molecular Hamiltonian, generating a CI basis, and diagonalizing to find ground state energy.

```python
from cm import qm

# Create H2 molecule at experimental bond length
mol = qm.molecule([('H', 0, 0, 0), ('H', 0.74, 0, 0)])

# Build electronic Hamiltonian with kinetic, nuclear attraction, and Coulomb terms
H = (qm.HamiltonianBuilder()
     .with_kinetic()
     .with_nuclear_attraction()
     .with_coulomb()  # Includes exchange automatically
     .build())

# Bind Hamiltonian to molecule for automatic geometry handling
mol_H = qm.MolecularHamiltonian(H, mol)

# Generate CI basis (Configuration Interaction)
basis = mol.ci_basis(excitations=2)

# Build and render the Hamiltonian matrix
matrix = mol_H.matrix(basis)
matrix.render()  # Displays evaluated matrix elements

# Diagonalize to find eigenvalues and eigenvectors
eigenvalues, eigenvectors = matrix.diagonalize()
ground_state_energy = eigenvalues[0]

print(f"Ground state energy: {ground_state_energy:.6f} Hartree")
# Output: -1.353251 Hartree
```

### Workflow 4: Benchmark Data and Visualization

This example demonstrates searching benchmark databases, retrieving experimental properties, and comparing with calculated molecular geometries.

```python
from cm.data import BenchmarkAPI
from cm import qm, views
import math

# Search NIST benchmark database for water molecule
api = BenchmarkAPI()
results = api.search(name="water", source="nist")
water_experimental = results[0]

# Retrieve experimental dipole moment
dipole = water_experimental.get_property("dipole_moment")
print(f"Experimental dipole: {dipole.value} {dipole.unit}")  # 1.85 Debye

# Construct water molecule with experimental geometry
r_OH = 0.9572  # O-H bond length (Angstroms)
angle_HOH = 104.52 * math.pi / 180  # H-O-H bond angle

water_calc = qm.molecule([
    ('O', 0, 0, 0),
    ('H', r_OH, 0, 0),
    ('H', r_OH * math.cos(angle_HOH), r_OH * math.sin(angle_HOH), 0),
])

# Visualize 3D structure in interactive WebGL viewer
views.molecule(water_calc, show_bonds=True, rotate=True)
```

The NIST CCCBDB provides high-quality experimental and computational benchmark data essential for validating quantum chemical calculations.

## Architecture

Chemical Machines consists of four integrated services orchestrated within a single Docker container:

### 1. cm-compute (Compute Daemon)

Node.js/Express service managing Python kernel execution and computational jobs.

- Persistent Python kernels for Jupyter-like notebook behavior
- PyTorch integration for GPU-accelerated calculations
- C++ native modules for performance-critical operations
- Job queue with process forking for concurrent execution
- Unix socket interface (`/var/run/cm-compute.sock`)

### 2. cm-view (Web Interface)

Vue.js frontend with Express backend providing the interactive research environment.

- Cell-based code editor with syntax highlighting (Python, C++, Bash)
- WebSocket connection to cm-compute for real-time execution
- WebGL visualization panel for 3D molecular rendering
- File browser and workspace management
- PostgreSQL integration for notebook persistence

### 3. cm-libraries (Python Package)

Scientific computing libraries (cm.symbols, cm.qm, cm.data, cm.views) providing the core research capabilities described above.

### 4. Data Services

- **PostgreSQL** (port 5432): Stores workspaces, notebooks, and user data
- **Elasticsearch** (port 9200): Indexes 134k+ molecules for fast search and autocomplete
- **Kibana** (port 5601): Visual interface for exploring benchmark data

**Technology Stack**: Vue 3 (frontend), Node.js/Express (backend), Python 3.12 (scientific computing), PyTorch (GPU), PostgreSQL, Elasticsearch, Docker, Supervisord (process management)

## Installation & Deployment

### For Researchers

The recommended approach is Docker-based deployment with GPU support:

#### System Requirements

- **Operating System**: Linux (Ubuntu 22.04 LTS recommended)
- **GPU**: CUDA-capable NVIDIA GPU (compute capability ≥ 6.0)
- **Memory**: 16GB RAM minimum, 32GB recommended
- **Storage**: 20GB for Docker images and databases
- **Software**: Docker Engine 20.10+, nvidia-docker2

#### Setup

```bash
# Install nvidia-docker2 (one-time setup)
# See https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

# Clone and start
git clone <repository-url>
cd chemical-machines
docker compose up -d

# Verify GPU access
docker compose exec chemical-machines nvidia-smi

# Access web interface
open http://localhost:5173
```

#### Configuration

Edit `docker-compose.yml` to customize:

- `POSTGRES_PASSWORD`: Database password (default: changeme)
- Port mappings if conflicts exist with existing services
- GPU device selection via `NVIDIA_VISIBLE_DEVICES`

### For Developers

For local development with hot-reload:

#### Component Development

```bash
# cm-compute (with file watching)
cd cm-compute
npm install
npm run dev  # Starts nodemon for auto-restart

# cm-view server and client (concurrent mode)
cd cm-view
npm install
npm run dev  # Runs both Express server and Vite dev server

# Python library (editable install)
cd cm-libraries/python
pip install -e .
python -m pytest tests/  # Run test suite
```

#### Building C++ Modules

```bash
cd cm-compute
npm run build:cpp  # Compiles native Node.js addons
```

#### Development Dockerfile

The `Dockerfile.dev` configuration includes additional tools:

- Clang C++ compiler
- Development dependencies with `--development` flag
- `nodemon` and `concurrently` for hot-reload
- Torch environment initialized on first startup

```bash
# Build dev image
docker compose -f docker-compose.dev.yml up -d --build
```

## Testing & Validation

The `cm-libraries` Python package includes comprehensive test coverage across all scientific modules.

### Running Tests

```bash
# Run complete test suite
docker compose exec chemical-machines bash -c \
  "cd /app/cm-libraries/python && PYTHONPATH=/app/cm-libraries/python python3 -m pytest tests/ -v"

# Run specific module
docker compose exec chemical-machines bash -c \
  "cd /app/cm-libraries/python && PYTHONPATH=/app/cm-libraries/python python3 -m pytest tests/qm/ -v"

# Generate coverage report
docker compose exec chemical-machines bash -c \
  "cd /app/cm-libraries/python && PYTHONPATH=/app/cm-libraries/python python3 -m pytest tests/ --cov=cm --cov-report=term-missing"
```

### Test Coverage

**Total**: 90+ tests ensuring correctness of scientific calculations

- **symbols** (18 tests): Variable creation, arithmetic operations, derivatives, integrals, special functions, LaTeX rendering
- **qm** (55 tests): Atomic configurations, spin-orbitals, Slater determinants, Hamiltonian construction, matrix elements, molecular geometries
- **data** (17 tests): Benchmark API, property retrieval, search functionality, data validation

See [Test Documentation](cm-libraries/python/tests/README.md) for detailed test descriptions and expected outcomes.

## Benchmark Data Access

The platform provides programmatic and visual access to molecular benchmark data through Elasticsearch and Kibana.

### Accessing Kibana

```bash
# Ensure Kibana is running
docker compose up -d kibana

# Open in browser
open http://localhost:5601
```

### Using Dev Tools

Navigate to **Management → Dev Tools** to run Elasticsearch queries directly:

```
# Search for water molecule
GET benchmark_molecules/_search
{
  "query": {
    "match": { "name": "water" }
  }
}

# List all available indices
GET _cat/indices?v

# View index structure
GET benchmark_molecules/_mapping
```

### Index Structure

The `benchmark_molecules` index contains:

- **Identifiers**: `identifier`, `cas`, `smiles`, `inchi_key`
- **Properties**: `name`, `formula`, `properties` (array of computed/experimental values)
- **Geometry**: `geometry` (XYZ coordinates in Angstroms)
- **Sources**: `sources` (nist, pubchem, qm9)

**Data Sources**:
- **NIST CCCBDB**: Experimental geometries, energies, vibrational frequencies
- **PubChem**: Molecular properties, SMILES, InChI keys
- **QM9**: ~134,000 molecules with pre-computed DFT properties

Kibana provides visual exploration, filtering, and visualization of this benchmark data, essential for validation workflows.

## Project Structure

```
chemical-machines/
├── cm-compute/              # GPU compute daemon (Node.js + Python + C++)
│   ├── src/                # Express server, kernel management
│   ├── lib/                # C++ native modules for performance-critical ops
│   └── package.json
├── cm-view/                # Web interface (Vue.js + Express)
│   ├── server/             # Express API, WebSocket, database connections
│   ├── client/             # Vue 3 frontend with Vite build system
│   │   └── src/
│   │       ├── components/ # CodeCell, FileBrowser, Terminal, etc.
│   │       └── views/      # WorkspaceView, NotebookView
│   └── package.json
├── cm-libraries/           # Python scientific libraries
│   └── python/cm/
│       ├── symbols/        # Symbolic math (core.py, functions.py, special.py)
│       ├── qm/            # Quantum mechanics (atoms.py, hamiltonian.py, etc.)
│       ├── data/          # Benchmark database API (benchmark.py)
│       └── views/         # Visualization (visualization.py, output.py)
├── scripts/               # Utility scripts (autocomplete generation, etc.)
├── docker/               # Container configuration
│   ├── Dockerfile        # Production image
│   ├── Dockerfile.dev    # Development image with hot-reload
│   ├── supervisord.conf  # Production process management
│   └── supervisord.dev.conf
├── workspaces/           # Workspace template and library documentation
│   └── README.md         # Comprehensive cm library reference
├── docker-compose.yml
└── README.md
```

**C++ Integration**: The `cm-compute/lib/` directory contains Node.js native addons compiled with node-gyp, enabling performance-critical calculations (large matrix operations, numerical integration) to run at near-C++ speeds while maintaining Python API compatibility.

## Configuration

### Environment Variables

Set in `docker-compose.yml`:

- `POSTGRES_PASSWORD`: Database password (default: changeme)
- `POSTGRES_USER`: Database user (default: cmuser)
- `POSTGRES_DB`: Database name (default: chemicalmachines)
- `COMPUTE_SOCKET_PATH`: Unix socket for cm-compute communication (default: `/var/run/cm-compute.sock`)
- `NODE_ENV`: `development` or `production`
- `ES_JAVA_OPTS`: Elasticsearch JVM settings (default: `-Xms2g -Xmx2g`)
- `NVIDIA_VISIBLE_DEVICES`: GPU device selection (default: `all`)

### Database Configuration

**PostgreSQL** listens on port 5432 with credentials:
- User: `cmuser`
- Password: configurable via `POSTGRES_PASSWORD`
- Database: `chemicalmachines`

**Elasticsearch** runs on port 9200 with security disabled for local development (`xpack.security.enabled: false`).

### GPU Configuration

Specify GPUs with:

```yaml
environment:
  - NVIDIA_VISIBLE_DEVICES=0,1  # Use only GPUs 0 and 1
```

Or disable GPU:

```yaml
environment:
  - NVIDIA_VISIBLE_DEVICES=""  # CPU-only mode
```

## Contributing

Contributions to Chemical Machines are welcome. The project prioritizes scientific accuracy, performance, and maintainability.

### Development Workflow

1. **Fork the repository** and create a feature branch
2. **Set up development environment**:
   ```bash
   docker compose -f docker-compose.dev.yml up -d
   ```
3. **Make changes** with hot-reload active
4. **Run tests** to ensure correctness:
   ```bash
   docker compose exec chemical-machines bash -c \
     "cd /app/cm-libraries/python && python3 -m pytest tests/ -v"
   ```
5. **Submit a pull request** with clear description of changes

### Code Style

- **Python**: Follow PEP 8, use type hints where beneficial
- **JavaScript/Vue**: ESLint configuration in project root
- **C++**: Google C++ Style Guide for native modules
- **Commits**: Clear, descriptive messages explaining the "why" not just the "what"

### Areas for Contribution

- **Basis sets**: Add support for additional Gaussian basis sets (6-31G, cc-pVDZ, etc.)
- **Methods**: Implement post-Hartree-Fock methods (MP2, CCSD)
- **Special functions**: Expand coverage of mathematical special functions
- **Visualization**: Enhanced molecular rendering (orbitals, isosurfaces)
- **Database integration**: Additional benchmark data sources
- **Documentation**: Examples, tutorials, API documentation

## License & Citation

### License

See [LICENSE.md](LICENSE.md) for dual licensing terms.

### Citing Chemical Machines

If you use Chemical Machines in published research, please cite:

```bibtex
@software{chemical_machines,
  title = {Chemical Machines: An Integrated Platform for Computational Chemistry},
  author = {[Author Names]},
  year = {2025},
  url = {https://github.com/[username]/chemical-machines},
  note = {GPU-accelerated quantum mechanics and symbolic mathematics platform}
}
```

### Publications

Research using Chemical Machines:

- [Publications will be listed here as they become available]

## Additional Resources

- **Comprehensive Library Documentation**: [workspaces/README.md](workspaces/README.md) - Complete API reference with detailed examples for all `cm` modules
- **Test Documentation**: [cm-libraries/python/tests/README.md](cm-libraries/python/tests/README.md) - Detailed test descriptions and validation procedures
- **Docker GPU Setup**: [NVIDIA Container Toolkit Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### Troubleshooting

**GPU not detected**:
```bash
docker compose exec chemical-machines nvidia-smi
# If this fails, verify nvidia-docker2 installation
```

**Services not starting**:
```bash
docker compose logs -f chemical-machines
# Check supervisor logs for specific service failures
```

**Test failures**:
- Ensure Python environment is correctly initialized
- Check PYTHONPATH is set correctly
- Verify all dependencies are installed

For additional support, please open an issue on the project repository.

---

**Version**: 1.0.0
**Last Updated**: January 2025
