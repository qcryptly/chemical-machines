"""
Benchmark Database Interface

Provides Python API for accessing and comparing molecular benchmark data
from NIST CCCBDB, PubChem, and QM9 datasets.
"""

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import urllib.request
import urllib.parse
import urllib.error
import socket


# API base URL - connects to cm-view server
API_BASE = os.environ.get('CM_API_URL', 'http://localhost:3000')


class BenchmarkError(Exception):
    """Base exception for benchmark API errors."""
    pass


class ServiceUnavailableError(BenchmarkError):
    """Raised when the Chemical Machines services are not available."""

    def __init__(self, service: str, details: str = ''):
        self.service = service
        self.details = details
        message = f"{service} is not available. "

        if service == 'cm-view':
            message += (
                "The cm-view server is not running.\n\n"
                "To start the services, run:\n"
                "  docker compose up -d\n\n"
                "Then access the benchmark API from within a Chemical Machines notebook."
            )
        elif service == 'cm-compute':
            message += (
                "The cm-compute daemon is not responding.\n\n"
                "Check that cm-compute is running:\n"
                "  docker compose logs cm-compute\n\n"
                "Or restart the services:\n"
                "  docker compose restart"
            )
        elif service == 'elasticsearch':
            message += (
                "Elasticsearch is not available.\n\n"
                "Check that Elasticsearch is running:\n"
                "  curl http://localhost:9200\n\n"
                "The benchmark index may not exist yet. Run:\n"
                "  benchmark.sync(['qm9'])"
            )

        if details:
            message += f"\n\nDetails: {details}"

        super().__init__(message)


class APIError(BenchmarkError):
    """Raised when the API returns an error response."""

    def __init__(self, status_code: int, message: str, endpoint: str = ''):
        self.status_code = status_code
        self.endpoint = endpoint
        error_msg = f"API error {status_code}"
        if endpoint:
            error_msg += f" on {endpoint}"
        error_msg += f": {message}"

        # Add helpful context for common errors
        if status_code == 500:
            error_msg += (
                "\n\nThis may indicate:\n"
                "  - Elasticsearch index doesn't exist (run benchmark.sync() first)\n"
                "  - cm-compute job handler error\n"
                "  - Database connection issue"
            )
        elif status_code == 404:
            error_msg += "\n\nThe requested resource was not found."

        super().__init__(error_msg)


class JobError(BenchmarkError):
    """Raised when a background job fails."""

    def __init__(self, job_id: str, error: str):
        self.job_id = job_id
        self.error = error
        super().__init__(f"Job {job_id} failed: {error}")


class IndexingInProgressError(BenchmarkError):
    """Raised when the benchmark database is currently being indexed."""

    def __init__(self, sources: List[str] = None, progress: float = None, job_id: str = None):
        self.sources = sources or []
        self.progress = progress
        self.job_id = job_id

        message = "Benchmark database is currently being indexed"
        if sources:
            message += f" (sources: {', '.join(sources)})"
        if progress is not None:
            message += f" - {progress:.0f}% complete"
        message += ".\n\nPlease wait for indexing to complete, then retry."
        if job_id:
            message += f"\nJob ID: {job_id}"

        super().__init__(message)


class MoleculeNotFoundError(BenchmarkError):
    """Raised when a molecule is not found in the benchmark database."""

    def __init__(self, identifier: str, can_fetch: bool = True):
        self.identifier = identifier
        self.can_fetch = can_fetch

        message = f'Molecule "{identifier}" was not found in the benchmark database.'
        if can_fetch:
            message += (
                "\n\nYou can fetch it from external sources using:\n"
                f'  mol = benchmark.get("{identifier}")\n'
                "This will query PubChem, NIST, and QM9."
            )
        else:
            message += "\n\nThis molecule is not available in any of the configured sources."

        super().__init__(message)


class NoIndexError(BenchmarkError):
    """Raised when the benchmark index has not been created yet."""

    def __init__(self):
        super().__init__(
            "Benchmark database has not been initialized.\n\n"
            "Run the following to download and index benchmark data:\n"
            "  from cm.data import benchmark\n"
            "  benchmark.sync(['qm9'])  # Downloads ~134k molecules\n\n"
            "For faster initial testing, fetch individual molecules:\n"
            "  mol = benchmark.get('water')  # Fetches from PubChem"
        )


@dataclass
class BenchmarkProperty:
    """A single molecular property from benchmark data."""
    name: str
    value: float
    unit: str = ''
    source: str = ''
    method: str = ''
    uncertainty: Optional[float] = None

    def __repr__(self) -> str:
        unit_str = f' {self.unit}' if self.unit else ''
        return f'{self.name}: {self.value}{unit_str} ({self.source})'


@dataclass
class BenchmarkMolecule:
    """Molecular data from benchmark databases."""
    identifier: str
    name: str = ''
    formula: str = ''
    cas: str = ''
    smiles: str = ''
    inchi: str = ''
    inchi_key: str = ''
    cid: Optional[int] = None
    molecular_weight: Optional[float] = None
    charge: int = 0
    multiplicity: int = 1
    sources: List[str] = field(default_factory=list)
    properties: List[BenchmarkProperty] = field(default_factory=list)
    geometry: Optional[Dict[str, Any]] = None
    file_paths: Dict[str, str] = field(default_factory=dict)
    cached_at: str = ''

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkMolecule':
        """Create BenchmarkMolecule from API response dictionary."""
        props = []
        for p in data.get('properties', []):
            props.append(BenchmarkProperty(
                name=p.get('name', ''),
                value=p.get('value', 0.0),
                unit=p.get('unit', ''),
                source=p.get('source', ''),
                method=p.get('method', ''),
                uncertainty=p.get('uncertainty')
            ))

        return cls(
            identifier=data.get('identifier', ''),
            name=data.get('name', ''),
            formula=data.get('formula', ''),
            cas=data.get('cas', ''),
            smiles=data.get('smiles', ''),
            inchi=data.get('inchi', ''),
            inchi_key=data.get('inchi_key', ''),
            cid=data.get('cid'),
            molecular_weight=data.get('molecular_weight'),
            charge=data.get('charge', 0),
            multiplicity=data.get('multiplicity', 1),
            sources=data.get('sources', []),
            properties=props,
            geometry=data.get('geometry'),
            file_paths=data.get('file_paths', {}),
            cached_at=data.get('cached_at', '')
        )

    def get_property(self, name: str, source: str = None) -> Optional[BenchmarkProperty]:
        """Get a specific property by name, optionally filtered by source."""
        for prop in self.properties:
            if prop.name == name or name in prop.name:
                if source is None or prop.source == source:
                    return prop
        return None

    def get_xyz(self) -> Optional[str]:
        """Get geometry as XYZ format string."""
        if not self.geometry or 'atoms' not in self.geometry:
            return None

        atoms = self.geometry['atoms']
        lines = [str(len(atoms)), f'{self.name} - from benchmark']

        for atom in atoms:
            lines.append(
                f"{atom['element']:2s}  {atom['x']:12.6f}  {atom['y']:12.6f}  {atom['z']:12.6f}"
            )

        return '\n'.join(lines)

    def render(self) -> None:
        """Display molecule data in notebook."""
        try:
            from cm.views import html
        except ImportError:
            print(self)
            return

        rows = []
        rows.append(f'<h3>{self.name or self.identifier}</h3>')
        rows.append('<table class="benchmark-table">')
        rows.append('<tr><th>Property</th><th>Value</th><th>Unit</th><th>Source</th></tr>')

        for prop in self.properties:
            rows.append(
                f'<tr><td>{prop.name}</td><td>{prop.value:.6g}</td>'
                f'<td>{prop.unit}</td><td>{prop.source}</td></tr>'
            )

        rows.append('</table>')

        if self.geometry:
            rows.append(f'<p><strong>Geometry:</strong> {len(self.geometry.get("atoms", []))} atoms</p>')

        rows.append(f'<p><em>Sources: {", ".join(self.sources)}</em></p>')

        html('\n'.join(rows))


@dataclass
class PropertyComparison:
    """Comparison of a single property between computed and benchmark."""
    property: str
    computed: float
    benchmark: Optional[float]
    difference: Optional[float] = None
    percent_difference: Optional[float] = None
    unit: str = ''
    benchmark_source: str = ''
    benchmark_method: str = ''
    note: str = ''


@dataclass
class ComparisonResult:
    """Result of comparing computed values with benchmark data."""
    identifier: str
    benchmark_sources: List[str]
    comparisons: List[PropertyComparison]
    properties_compared: int = 0
    properties_missing: int = 0
    avg_percent_diff: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ComparisonResult':
        """Create ComparisonResult from API response dictionary."""
        comparisons = []
        for c in data.get('comparisons', []):
            comparisons.append(PropertyComparison(
                property=c.get('property', ''),
                computed=c.get('computed', 0.0),
                benchmark=c.get('benchmark'),
                difference=c.get('difference'),
                percent_difference=c.get('percent_difference'),
                unit=c.get('unit', ''),
                benchmark_source=c.get('benchmark_source', ''),
                benchmark_method=c.get('benchmark_method', ''),
                note=c.get('note', '')
            ))

        summary = data.get('summary', {})

        return cls(
            identifier=data.get('identifier', ''),
            benchmark_sources=data.get('benchmark_sources', []),
            comparisons=comparisons,
            properties_compared=summary.get('properties_compared', 0),
            properties_missing=summary.get('properties_missing', 0),
            avg_percent_diff=summary.get('avg_percent_diff')
        )

    def render(self) -> None:
        """Display comparison table in notebook."""
        try:
            from cm.views import html
        except ImportError:
            print(self)
            return

        rows = []
        rows.append(f'<h3>Comparison: {self.identifier}</h3>')
        rows.append(f'<p>Benchmark sources: {", ".join(self.benchmark_sources)}</p>')

        rows.append('<table class="comparison-table">')
        rows.append(
            '<tr><th>Property</th><th>Computed</th><th>Benchmark</th>'
            '<th>Difference</th><th>% Diff</th><th>Method</th></tr>'
        )

        for comp in self.comparisons:
            diff_class = ''
            if comp.percent_difference is not None:
                abs_diff = abs(comp.percent_difference)
                if abs_diff < 1:
                    diff_class = 'style="color: #4caf50"'
                elif abs_diff < 5:
                    diff_class = 'style="color: #8bc34a"'
                elif abs_diff < 10:
                    diff_class = 'style="color: #ff9800"'
                else:
                    diff_class = 'style="color: #f44336"'

            bench_val = f'{comp.benchmark:.6g}' if comp.benchmark is not None else '-'
            diff_val = f'{comp.difference:.6g}' if comp.difference is not None else '-'
            pct_val = f'{comp.percent_difference:.2f}%' if comp.percent_difference is not None else '-'

            rows.append(
                f'<tr><td>{comp.property}</td><td>{comp.computed:.6g}</td>'
                f'<td>{bench_val}</td><td {diff_class}>{diff_val}</td>'
                f'<td {diff_class}>{pct_val}</td><td>{comp.benchmark_method or comp.note or "-"}</td></tr>'
            )

        rows.append('</table>')

        if self.avg_percent_diff is not None:
            rows.append(f'<p><strong>Average percent difference:</strong> {self.avg_percent_diff:.2f}%</p>')

        html('\n'.join(rows))


def _api_request(endpoint: str, method: str = 'GET', data: Dict = None, timeout: float = 30) -> Dict:
    """Make API request to cm-view server."""
    url = f'{API_BASE}{endpoint}'

    if method == 'GET' and data:
        query = urllib.parse.urlencode(data)
        url = f'{url}?{query}'
        req = urllib.request.Request(url)
    elif method == 'POST':
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode('utf-8') if data else None,
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
    else:
        req = urllib.request.Request(url)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode('utf-8'))

    except urllib.error.URLError as e:
        # Connection errors
        if isinstance(e.reason, socket.timeout):
            raise ServiceUnavailableError(
                'cm-view',
                f"Connection timed out after {timeout}s"
            )
        elif isinstance(e.reason, ConnectionRefusedError):
            raise ServiceUnavailableError(
                'cm-view',
                f"Connection refused to {API_BASE}"
            )
        elif isinstance(e.reason, socket.gaierror):
            raise ServiceUnavailableError(
                'cm-view',
                f"Could not resolve host: {API_BASE}"
            )
        else:
            raise ServiceUnavailableError(
                'cm-view',
                str(e.reason)
            )

    except urllib.error.HTTPError as e:
        # HTTP errors with status codes
        try:
            error_body = e.read().decode('utf-8')
            error_data = json.loads(error_body)
            error_message = error_data.get('error', error_body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            error_message = e.reason or str(e)

        # Detect specific service issues from error messages
        if e.code == 500:
            error_lower = error_message.lower()
            if 'elasticsearch' in error_lower or 'econnrefused' in error_lower:
                raise ServiceUnavailableError('elasticsearch', error_message)
            elif 'compute' in error_lower or 'socket' in error_lower:
                raise ServiceUnavailableError('cm-compute', error_message)

        raise APIError(e.code, error_message, endpoint)

    except socket.timeout:
        raise ServiceUnavailableError(
            'cm-view',
            f"Request timed out after {timeout}s"
        )


def _wait_for_job(job_id: str, timeout: float = 60) -> Dict:
    """Poll for job completion."""
    start = time.time()
    poll_count = 0

    while time.time() - start < timeout:
        try:
            result = _api_request(f'/api/compute/{job_id}')
        except APIError as e:
            if e.status_code == 404:
                # Job might not be registered yet, wait and retry
                if poll_count < 3:
                    poll_count += 1
                    time.sleep(0.5)
                    continue
                raise JobError(job_id, "Job not found - it may have been cancelled or never started")
            raise

        status = result.get('status')

        if status == 'completed':
            return result.get('result', {})
        elif status == 'failed':
            error = result.get('error', 'Unknown error')
            raise JobError(job_id, error)
        elif status == 'cancelled':
            raise JobError(job_id, 'Job was cancelled')

        poll_count += 1
        time.sleep(0.5)

    raise TimeoutError(
        f"Job {job_id} did not complete within {timeout} seconds.\n"
        f"The job may still be running. Check status with:\n"
        f"  curl {API_BASE}/api/compute/{job_id}"
    )


def search(
    query: str = None,
    *,
    name: str = None,
    formula: str = None,
    cas: str = None,
    smiles: str = None,
    sources: List[str] = None,
    limit: int = 20
) -> List[BenchmarkMolecule]:
    """
    Search benchmark databases for molecules.

    Args:
        query: General search query (name, formula, CAS, SMILES)
        name: Search by name
        formula: Search by molecular formula
        cas: Search by CAS number
        smiles: Search by SMILES
        sources: List of sources to search (pubchem, nist, qm9)
        limit: Maximum results to return

    Returns:
        List of matching BenchmarkMolecule objects

    Raises:
        NoIndexError: If the benchmark index doesn't exist yet
        IndexingInProgressError: If the database is currently being indexed
        ServiceUnavailableError: If services are not running

    Example:
        >>> results = search("water")
        >>> results = search(formula="H2O")
        >>> results = search(cas="7732-18-5")
    """
    # Build search query
    q = query or name or formula or cas or smiles
    if not q:
        raise ValueError("Must provide a search query")

    params = {'q': q, 'limit': str(limit)}
    if sources:
        params['sources'] = ','.join(sources)

    try:
        result = _api_request('/api/benchmark/search', data=params)
    except APIError as e:
        # Check if this is an index-related error
        if e.status_code == 500 and 'index' in str(e).lower():
            # Check indexing status for better error message
            _check_indexing_status()
            # If we get here, index doesn't exist
            raise NoIndexError()
        raise

    # Check for indexing status in response
    if result.get('status') == 'indexing':
        raise IndexingInProgressError(
            sources=result.get('indexing_sources', []),
            progress=result.get('indexing_progress')
        )

    if result.get('status') == 'searching' and result.get('jobId'):
        job_result = _wait_for_job(result['jobId'])
        results = job_result.get('results', [])
    else:
        results = result.get('results', [])

    return [BenchmarkMolecule.from_dict(r) for r in results]


def get(
    identifier: str,
    sources: List[str] = None,
    workspace_id: str = '1',
    fetch_if_missing: bool = True
) -> BenchmarkMolecule:
    """
    Get detailed benchmark data for a specific molecule.

    Args:
        identifier: CAS number, PubChem CID, SMILES, or InChIKey
        sources: Sources to query (pubchem, nist, qm9)
        workspace_id: Workspace for storing downloaded files
        fetch_if_missing: If True, fetch from external sources when not cached

    Returns:
        BenchmarkMolecule with full property data

    Raises:
        NoIndexError: If the benchmark index doesn't exist yet
        IndexingInProgressError: If the database is currently being indexed
        MoleculeNotFoundError: If molecule not found and fetch_if_missing=False
        ServiceUnavailableError: If services are not running

    Example:
        >>> mol = get("7732-18-5")  # Water by CAS
        >>> mol = get("962")        # PubChem CID
        >>> mol.properties
    """
    params = {'workspaceId': workspace_id}
    if sources:
        params['sources'] = ','.join(sources)
    if not fetch_if_missing:
        params['cacheOnly'] = 'true'

    try:
        result = _api_request(f'/api/benchmark/molecule/{urllib.parse.quote(identifier)}', data=params)
    except APIError as e:
        # Check if this is an index-related error
        if e.status_code == 500 and 'index' in str(e).lower():
            _check_indexing_status()
            raise NoIndexError()
        elif e.status_code == 404:
            # Molecule not found - check if we should suggest fetching
            if fetch_if_missing:
                raise MoleculeNotFoundError(identifier, can_fetch=True)
            else:
                raise MoleculeNotFoundError(identifier, can_fetch=False)
        raise

    # Check for indexing status in response
    if result.get('status') == 'indexing':
        raise IndexingInProgressError(
            sources=result.get('indexing_sources', []),
            progress=result.get('indexing_progress')
        )

    # Check for not found response
    if result.get('status') == 'not_found':
        raise MoleculeNotFoundError(identifier, can_fetch=fetch_if_missing)

    if result.get('status') == 'fetching' and result.get('jobId'):
        job_result = _wait_for_job(result['jobId'], timeout=120)
        data = job_result.get('data', job_result)
    elif result.get('status') == 'cached':
        data = result.get('data', {})
    else:
        data = result

    return BenchmarkMolecule.from_dict(data)


def compare(
    computed: Union[Dict[str, float], Any],
    identifier: str
) -> ComparisonResult:
    """
    Compare computed molecular properties with benchmark data.

    Args:
        computed: Dictionary of computed properties, or an HFResult/DFTResult
                  object from cm.qm.integrals
        identifier: Benchmark molecule identifier

    Returns:
        ComparisonResult with property-by-property comparison

    Example:
        >>> comparison = compare(
        ...     {"total_energy": -76.026, "dipole_moment": 1.85},
        ...     "7732-18-5"
        ... )
        >>> comparison.render()

        >>> # Or pass an HFResult directly:
        >>> hf = hartree_fock(atoms, basis='STO-3G')
        >>> comparison = compare(hf, "7732-18-5")
    """
    # Handle HFResult/DFTResult objects (duck typing via density matrix)
    if hasattr(computed, 'energy') and hasattr(computed, 'density'):
        props = {}
        try:
            props['total_energy'] = float(computed.energy)
        except Exception:
            pass
        # Compute dipole moment if possible
        try:
            from cm.qm.integrals import dipole_moment as _dipole_moment
            dipole = _dipole_moment(computed)
            props['dipole_moment'] = dipole.magnitude
        except Exception:
            pass
        computed = props

    if not isinstance(computed, dict):
        raise TypeError("computed must be a dict or Molecule object")

    result = _api_request('/api/benchmark/compare', method='POST', data={
        'computed': computed,
        'identifier': identifier
    })

    if result.get('status') == 'comparing' and result.get('jobId'):
        job_result = _wait_for_job(result['jobId'])
        return ComparisonResult.from_dict(job_result)

    return ComparisonResult.from_dict(result)


def sync(sources: List[str] = None, workspace_id: str = '1') -> Dict[str, Any]:
    """
    Trigger synchronization of benchmark databases.

    Downloads and indexes data from specified sources.
    For QM9, this downloads the full dataset (~134k molecules).

    Args:
        sources: Sources to sync (default: ['qm9'])
        workspace_id: Workspace for storing downloads

    Returns:
        Sync job status

    Example:
        >>> sync(['qm9'])  # Download and index QM9 dataset
    """
    result = _api_request('/api/benchmark/sync', method='POST', data={
        'sources': sources or ['qm9'],
        'workspaceId': workspace_id
    })

    return result


def stats() -> Dict[str, Any]:
    """
    Get statistics about indexed benchmark data.

    Returns:
        Dictionary with molecule counts per source, plus indexing status

    Example:
        >>> stats()
        {'sources': {'pubchem': 150, 'nist': 42, 'qm9': 133885}, 'total_molecules': 134077}
    """
    return _api_request('/api/benchmark/stats')


def sync_status() -> Dict[str, Any]:
    """
    Get current sync status with detailed progress information.

    Returns:
        Dictionary with sync status including:
        - is_syncing: bool - Whether a sync is currently in progress
        - jobs: List of active indexing jobs with progress details
        - sources: Dict of molecule counts per source

    Example:
        >>> sync_status()
        {
            'is_syncing': True,
            'jobs': [{
                'source': 'qm9',
                'progress': 45,
                'phase': 'indexing',
                'details': {'indexed': 60000, 'total': 133885}
            }],
            'sources': {'qm9': 60000}
        }
    """
    st = _api_request('/api/benchmark/stats')

    # Check for active indexing jobs
    indexing = st.get('indexing', {})
    jobs = []

    for source, info in indexing.items():
        if isinstance(info, dict):
            jobs.append({
                'source': source,
                'job_id': info.get('jobId', ''),
                'progress': info.get('progress', 0),
                'started_at': info.get('startedAt', ''),
                'status': info.get('status', 'indexing')
            })

    return {
        'is_syncing': len(jobs) > 0,
        'jobs': jobs,
        'sources': st.get('sources', {}),
        'total_molecules': st.get('total_molecules', 0),
        'index_exists': st.get('index_exists', True)
    }


def wait_for_sync(
    poll_interval: float = 2.0,
    timeout: float = 3600,
    show_progress: bool = True
) -> Dict[str, Any]:
    """
    Wait for sync to complete, optionally showing progress.

    Args:
        poll_interval: Seconds between status checks (default: 2.0)
        timeout: Maximum seconds to wait (default: 3600 = 1 hour)
        show_progress: If True, print progress updates

    Returns:
        Final stats when sync completes

    Raises:
        TimeoutError: If sync doesn't complete within timeout

    Example:
        >>> sync(['qm9'])
        >>> wait_for_sync(show_progress=True)
        Syncing qm9: 45% (phase: indexing, indexed: 60000)
        Syncing qm9: 78% (phase: indexing, indexed: 105000)
        Syncing qm9: 100% (phase: complete, indexed: 133885)
        Sync complete!
    """
    start = time.time()

    while time.time() - start < timeout:
        try:
            ss = sync_status()

            if not ss['is_syncing']:
                if show_progress:
                    print("Sync complete!")
                    st = stats()
                    print(f"Indexed molecules: {st.get('total_molecules', 0)}")
                return stats()

            if show_progress and ss['jobs']:
                for job in ss['jobs']:
                    source = job.get('source', 'unknown')
                    progress = job.get('progress', 0)
                    msg = f"Syncing {source}: {progress:.0f}%"

                    # Add phase and details if available
                    st = status('*')  # Get detailed status
                    for ij in st.indexing_jobs:
                        if ij.source == source:
                            if ij.phase:
                                msg += f" (phase: {ij.phase}"
                                if ij.phase == 'download' and ij.downloaded:
                                    msg += f", {ij.downloaded}/{ij.total}"
                                elif ij.phase == 'indexing' and ij.indexed:
                                    msg += f", indexed: {ij.indexed}"
                                msg += ")"
                            break

                    print(msg)

        except (ServiceUnavailableError, APIError) as e:
            if show_progress:
                print(f"Warning: {e}")

        time.sleep(poll_interval)

    raise TimeoutError(f"Sync did not complete within {timeout} seconds")


@dataclass
class IndexingJobInfo:
    """Information about an active indexing job."""
    source: str
    job_id: str
    progress: float
    phase: str = ''
    details: Dict[str, Any] = field(default_factory=dict)
    started_at: str = ''

    @property
    def downloaded(self) -> str:
        """Get downloaded size string from details."""
        return self.details.get('downloaded', '')

    @property
    def total(self) -> str:
        """Get total size string from details."""
        return self.details.get('total', '')

    @property
    def indexed(self) -> int:
        """Get number of indexed molecules from details."""
        return self.details.get('indexed', 0)


@dataclass
class MoleculeStatus:
    """Status of a molecule in the benchmark database."""
    identifier: str
    status: str  # 'indexed', 'indexing', 'not_found', 'no_index'
    exists: bool
    sources: List[str] = field(default_factory=list)
    cached_at: str = ''
    indexing_sources: List[str] = field(default_factory=list)
    indexing_progress: Optional[float] = None
    indexing_jobs: List[IndexingJobInfo] = field(default_factory=list)
    message: str = ''

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MoleculeStatus':
        # Parse indexing_jobs array from server response
        raw_jobs = data.get('indexing_jobs', [])
        indexing_jobs = []
        indexing_sources = []

        for j in raw_jobs:
            source = j.get('source', '')
            indexing_sources.append(source)
            indexing_jobs.append(IndexingJobInfo(
                source=source,
                job_id=j.get('jobId', ''),
                progress=j.get('progress', 0) or 0,
                phase=j.get('phase', ''),
                details=j.get('details', {}),
                started_at=j.get('startedAt', '')
            ))

        # Calculate average progress from all jobs
        indexing_progress = None
        if indexing_jobs:
            progresses = [j.progress for j in indexing_jobs if j.progress is not None]
            if progresses:
                indexing_progress = sum(progresses) / len(progresses)

        # Handle molecule data when status is 'indexed'
        molecule = data.get('molecule', {})

        return cls(
            identifier=molecule.get('identifier', data.get('identifier', '')),
            status=data.get('status', 'not_found'),
            exists=data.get('exists', False),
            sources=molecule.get('sources', data.get('sources', [])),
            cached_at=molecule.get('cached_at', data.get('cached_at', '')),
            indexing_sources=indexing_sources,
            indexing_progress=indexing_progress,
            indexing_jobs=indexing_jobs,
            message=data.get('message', '')
        )


def status(identifier: str) -> MoleculeStatus:
    """
    Check the status of a molecule in the benchmark database.

    This function checks whether a molecule is indexed, currently being indexed,
    or not found in the database.

    Args:
        identifier: CAS number, PubChem CID, SMILES, or InChIKey

    Returns:
        MoleculeStatus with status information

    Status values:
        - 'indexed': Molecule data is available
        - 'indexing': Database is currently being indexed
        - 'not_found': Molecule not in database (can fetch from external sources)
        - 'no_index': Index doesn't exist yet (run sync() first)

    Example:
        >>> st = status("7732-18-5")
        >>> if st.status == 'indexed':
        ...     mol = get("7732-18-5")
        >>> elif st.status == 'indexing':
        ...     progress = f"{st.indexing_progress:.0f}%" if st.indexing_progress else "in progress"
        ...     print(f"Please wait, indexing {progress}")
    """
    result = _api_request(f'/api/benchmark/status/{urllib.parse.quote(identifier)}')
    return MoleculeStatus.from_dict(result)


def _check_indexing_status() -> None:
    """
    Check if indexing is in progress and raise appropriate exception.

    This helper checks the global indexing status and raises
    IndexingInProgressError if any sources are being indexed.
    """
    try:
        st = _api_request('/api/benchmark/stats')
        indexing = st.get('indexing', {})

        if indexing.get('is_indexing'):
            active_sources = indexing.get('active_sources', [])
            progress = indexing.get('progress')
            raise IndexingInProgressError(
                sources=active_sources,
                progress=progress
            )
    except APIError:
        # Stats endpoint might fail if index doesn't exist
        pass


def _handle_status_errors(identifier: str, check_indexing: bool = True) -> None:
    """
    Check status endpoint and raise appropriate exceptions.

    Args:
        identifier: Molecule identifier being looked up
        check_indexing: Whether to check for indexing status

    Raises:
        NoIndexError: If the index doesn't exist
        IndexingInProgressError: If database is being indexed
        MoleculeNotFoundError: If molecule not found (only if check_indexing=False)
    """
    try:
        st = status(identifier)

        if st.status == 'no_index':
            raise NoIndexError()
        elif st.status == 'indexing':
            raise IndexingInProgressError(
                sources=st.indexing_sources,
                progress=st.indexing_progress
            )
        elif st.status == 'not_found' and not check_indexing:
            # Only raise not found if we're not going to fetch
            raise MoleculeNotFoundError(identifier, can_fetch=True)

    except APIError as e:
        # Status endpoint might not be available, continue
        if e.status_code == 404:
            pass
        else:
            raise
