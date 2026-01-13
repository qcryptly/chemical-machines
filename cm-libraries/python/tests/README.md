# Chemical Machines Test Suite

Comprehensive unit and integration tests for the Chemical Machines Python libraries.

## Running Tests

### Run all tests
```bash
docker compose exec chemical-machines bash -c "cd /app/cm-libraries/python && PYTHONPATH=/app/cm-libraries/python python3 -m pytest tests/"
```

### Run specific test modules
```bash
# Symbols tests
docker compose exec chemical-machines bash -c "cd /app/cm-libraries/python && PYTHONPATH=/app/cm-libraries/python python3 -m pytest tests/symbols/"

# QM tests
docker compose exec chemical-machines bash -c "cd /app/cm-libraries/python && PYTHONPATH=/app/cm-libraries/python python3 -m pytest tests/qm/"

# Benchmark tests
docker compose exec chemical-machines bash -c "cd /app/cm-libraries/python && PYTHONPATH=/app/cm-libraries/python python3 -m pytest tests/data/"
```

### Run with coverage
```bash
docker compose exec chemical-machines bash -c "cd /app/cm-libraries/python && PYTHONPATH=/app/cm-libraries/python python3 -m pytest tests/ --cov=cm --cov-report=html"
```

### Run specific test classes or functions
```bash
# Run a specific test class
docker compose exec chemical-machines bash -c "cd /app/cm-libraries/python && PYTHONPATH=/app/cm-libraries/python python3 -m pytest tests/symbols/test_core.py::TestVar"

# Run a specific test function
docker compose exec chemical-machines bash -c "cd /app/cm-libraries/python && PYTHONPATH=/app/cm-libraries/python python3 -m pytest tests/symbols/test_core.py::TestVar::test_create_var"
```

### Run with markers
```bash
# Run only unit tests
docker compose exec chemical-machines bash -c "cd /app/cm-libraries/python && PYTHONPATH=/app/cm-libraries/python python3 -m pytest tests/ -m unit"

# Run only integration tests (requires running services)
docker compose exec chemical-machines bash -c "cd /app/cm-libraries/python && PYTHONPATH=/app/cm-libraries/python python3 -m pytest tests/ -m integration"

# Skip slow tests
docker compose exec chemical-machines bash -c "cd /app/cm-libraries/python && PYTHONPATH=/app/cm-libraries/python python3 -m pytest tests/ -m 'not slow'"
```

## Test Structure

```
tests/
├── conftest.py           # Shared fixtures and configuration
├── symbols/              # Tests for cm.symbols module
│   └── test_core.py      # Core symbols (Var, Const, Sum, Product)
├── qm/                   # Tests for cm.qm module
│   ├── test_hamiltonian.py   # Hamiltonian builder and matrix elements
│   ├── test_spinorbitals.py  # Spin-orbitals and Slater determinants
│   └── test_molecules.py     # Atoms and molecules
├── data/                 # Tests for cm.data module
│   └── test_benchmark.py # Benchmark database API
└── views/                # Tests for cm.views module
    └── test_output.py    # Output and visualization
```

## Test Markers

- `@pytest.mark.unit` - Fast unit tests
- `@pytest.mark.integration` - Integration tests (require running services)
- `@pytest.mark.slow` - Slow-running tests
- `@pytest.mark.symbols` - Tests for symbols module
- `@pytest.mark.qm` - Tests for QM module
- `@pytest.mark.data` - Tests for data module

## Coverage

Target coverage: >80% for all modules

Run coverage report:
```bash
docker compose exec chemical-machines bash -c "cd /app/cm-libraries/python && PYTHONPATH=/app/cm-libraries/python python3 -m pytest tests/ --cov=cm --cov-report=term-missing"
```

## Writing Tests

### Test Naming Conventions

- Test files: `test_*.py`
- Test classes: `Test*`
- Test functions: `test_*`

### Example Test

```python
import pytest
from cm.symbols import Var, Const

class TestMyFeature:
    """Tests for my feature."""

    def test_basic_functionality(self):
        """Test basic functionality."""
        x = Var("x")
        c = Const(5)
        result = x + c
        assert result is not None

    @pytest.mark.unit
    def test_edge_case(self):
        """Test edge case."""
        # Test implementation
        pass

    @pytest.mark.slow
    def test_performance(self):
        """Test performance."""
        # Slow test implementation
        pass
```

### Fixtures

Common fixtures are defined in `conftest.py`:

- `sample_molecule_h2` - Simple H2 molecule
- `sample_molecule_water` - Water molecule with symbolic geometry
- `simple_hamiltonian` - Electronic Hamiltonian

Use fixtures in tests:
```python
def test_with_molecule(sample_molecule_h2):
    """Test using H2 molecule fixture."""
    mol = sample_molecule_h2
    assert mol.n_atoms == 2
```

## CI/CD Integration

Tests should be run in CI/CD pipelines before deployment:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    docker compose exec -T chemical-machines bash -c "cd /app/cm-libraries/python && PYTHONPATH=/app/cm-libraries/python python3 -m pytest tests/"
```

## Troubleshooting

### Import Errors

If you get import errors, make sure `PYTHONPATH` is set correctly:
```bash
export PYTHONPATH=/app/cm-libraries/python
```

### Service Dependencies

Some integration tests require running Chemical Machines services (cm-view, cm-compute, Elasticsearch). These tests are marked with `@pytest.mark.skip` or `@pytest.mark.integration`.

To run integration tests, ensure all services are running:
```bash
docker compose up -d
```

### Debugging Tests

Run with verbose output and don't capture stdout:
```bash
docker compose exec chemical-machines bash -c "cd /app/cm-libraries/python && PYTHONPATH=/app/cm-libraries/python python3 -m pytest tests/ -vv -s"
```

Run with Python debugger:
```bash
docker compose exec chemical-machines bash -c "cd /app/cm-libraries/python && PYTHONPATH=/app/cm-libraries/python python3 -m pytest tests/ --pdb"
```
