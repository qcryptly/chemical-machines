"""
Dynamic autocomplete using jedi library for Python code completion.

This module provides intelligent code completion by analyzing Python code context
and variable types. It integrates with the jedi library to provide:
- Instance method/attribute completion (e.g., atom_instance.energy)
- Type-aware suggestions
- Docstring information
- Function signatures

Usage:
    from cm.autocomplete import get_completions

    code = "from cm.qm import atom\\nC = atom('C')\\nC."
    completions = get_completions(code, line=3, column=2)
    # Returns list of dicts with completion information
"""

import jedi
from typing import List, Dict, Optional, Any


def get_completions(
    code: str,
    line: int,
    column: int,
    namespace: Optional[Dict[str, Any]] = None
) -> List[Dict[str, str]]:
    """
    Get code completions using jedi.

    Args:
        code: Full code context to analyze
        line: Cursor line number (1-indexed)
        column: Cursor column number (0-indexed)
        namespace: Optional current kernel namespace for additional context

    Returns:
        List of completion dictionaries with keys:
        - name: Completion text
        - type: Type of completion ('function', 'class', 'instance', 'module', etc.)
        - signature: Function signature if available
        - docstring: First 200 chars of docstring if available

    Example:
        >>> code = "from cm.qm import atom\\nC = atom('C')\\nC."
        >>> completions = get_completions(code, line=3, column=2)
        >>> [c['name'] for c in completions]
        ['energy', 'configuration', 'z', 'symbol', ...]
    """
    try:
        # Create jedi Script object for analysis
        script = jedi.Script(code, path='<stdin>')

        # Get completions at the specified position
        jedi_completions = script.complete(line, column)

        # Convert jedi completions to our format
        results = []
        for completion in jedi_completions[:50]:  # Limit to top 50 results
            try:
                # Get signature if available
                signature = None
                signatures = completion.get_signatures()
                if signatures:
                    try:
                        signature = signatures[0].to_string()
                    except Exception:
                        signature = None

                # Get docstring (truncate to 200 chars)
                docstring = None
                try:
                    doc = completion.docstring(raw=True)
                    if doc:
                        docstring = doc[:200] + '...' if len(doc) > 200 else doc
                except Exception:
                    docstring = None

                # Build completion entry
                results.append({
                    'name': completion.name,
                    'type': completion.type,  # 'function', 'class', 'instance', 'module', etc.
                    'signature': signature,
                    'docstring': docstring
                })
            except Exception as e:
                # Skip problematic completions
                continue

        return results

    except Exception as e:
        # Return empty list on any error - graceful degradation
        import traceback
        print(f"Autocomplete error: {e}")
        traceback.print_exc()
        return []


def get_completions_json(code: str, line: int, column: int) -> str:
    """
    Get completions and return as JSON string.

    Convenience wrapper for use in kernel execution where we need
    to return results as JSON output.

    Args:
        code: Full code context
        line: Cursor line (1-indexed)
        column: Cursor column (0-indexed)

    Returns:
        JSON string of completions list
    """
    import json
    completions = get_completions(code, line, column)
    return json.dumps(completions)


# Example usage and testing
if __name__ == '__main__':
    # Test 1: Class method completion
    test_code_1 = """from cm.qm import atom
C = atom('C')
C."""

    print("Test 1: Instance method completion")
    print("-" * 50)
    completions = get_completions(test_code_1, line=3, column=2)
    for comp in completions[:10]:
        print(f"  {comp['name']:20s} [{comp['type']}]")

    # Test 2: Module completion
    test_code_2 = """from cm.symbols import Math
Math."""

    print("\nTest 2: Class method completion")
    print("-" * 50)
    completions = get_completions(test_code_2, line=2, column=5)
    for comp in completions[:10]:
        print(f"  {comp['name']:20s} [{comp['type']}]")

    # Test 3: Import completion
    test_code_3 = """from cm.qm import """

    print("\nTest 3: Import completion")
    print("-" * 50)
    completions = get_completions(test_code_3, line=1, column=18)
    for comp in completions[:10]:
        print(f"  {comp['name']:20s} [{comp['type']}]")
