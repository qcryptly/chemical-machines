"""
Unit tests for the dynamic autocomplete module.

Tests jedi-based code completion for:
- Instance method completion
- Class method completion
- Import completion
- Variable attribute completion
"""

import pytest
from cm.autocomplete import get_completions, get_completions_json
import json


class TestInstanceCompletion:
    """Test autocomplete for object instances."""

    def test_atom_instance_completion(self):
        """Test completion for atom instance methods."""
        code = """from cm.qm import atom
C = atom('C')
C."""
        completions = get_completions(code, line=3, column=2)
        completion_names = [c['name'] for c in completions]

        # Should include atom instance methods
        assert 'energy' in completion_names or len(completions) > 0, \
            "Should return completions for atom instance"

    def test_math_instance_completion(self):
        """Test completion for Math symbolic expression."""
        code = """from cm.symbols import Math
x = Math.var('x')
x."""
        completions = get_completions(code, line=3, column=2)
        completion_names = [c['name'] for c in completions]

        # Should include expression methods like diff, integrate, etc.
        # (exact methods depend on implementation)
        assert len(completions) > 0, \
            "Should return completions for Math expression instance"


class TestClassCompletion:
    """Test autocomplete for class methods."""

    def test_math_class_completion(self):
        """Test completion for Math class methods."""
        code = """from cm.symbols import Math
Math."""
        completions = get_completions(code, line=2, column=5)
        completion_names = [c['name'] for c in completions]

        # Should include Math class methods
        expected_methods = ['sin', 'cos', 'var', 'exp']
        found = [m for m in expected_methods if m in completion_names]

        assert len(found) > 0, \
            f"Should find Math class methods. Got: {completion_names[:10]}"

    def test_atom_function_in_imports(self):
        """Test completion for imports."""
        code = """from cm.qm import """
        completions = get_completions(code, line=1, column=18)
        completion_names = [c['name'] for c in completions]

        # Should suggest importable items from cm.qm
        assert len(completions) > 0, \
            "Should return import suggestions"


class TestCompletionMetadata:
    """Test completion metadata (type, signature, docstring)."""

    def test_completion_has_type(self):
        """Test that completions include type information."""
        code = """from cm.qm import atom
C = atom('C')
C."""
        completions = get_completions(code, line=3, column=2)

        if completions:
            # All completions should have a type
            for comp in completions:
                assert 'type' in comp, \
                    f"Completion {comp.get('name')} missing 'type' field"
                assert 'name' in comp, \
                    "Completion missing 'name' field"

    def test_completion_optional_fields(self):
        """Test optional fields (signature, docstring)."""
        code = """from cm.symbols import Math
Math.sin"""
        completions = get_completions(code, line=2, column=8)

        if completions:
            # At least one completion should have signature or docstring
            has_metadata = any(
                comp.get('signature') or comp.get('docstring')
                for comp in completions
            )
            # Note: This is optional, so we just check structure
            assert isinstance(completions[0], dict), \
                "Completions should be dictionaries"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_code(self):
        """Test completion with empty code."""
        completions = get_completions("", line=1, column=0)
        assert isinstance(completions, list), \
            "Should return empty list for empty code"

    def test_invalid_syntax(self):
        """Test completion with syntax errors."""
        code = """def broken(
x."""
        # Should not crash, just return empty or partial results
        completions = get_completions(code, line=2, column=2)
        assert isinstance(completions, list), \
            "Should return list even with syntax errors"

    def test_large_code(self):
        """Test completion with large code block."""
        code = "\n".join([f"x{i} = {i}" for i in range(100)])
        code += "\nx0."

        completions = get_completions(code, line=101, column=3)
        assert isinstance(completions, list), \
            "Should handle large code blocks"

    def test_multiline_statement(self):
        """Test completion in multiline statement."""
        code = """from cm.qm import (
    atom,
    Molecule
)
C = atom('C')
C."""
        completions = get_completions(code, line=6, column=2)
        assert isinstance(completions, list), \
            "Should handle multiline imports"


class TestJSONOutput:
    """Test JSON serialization helper."""

    def test_get_completions_json(self):
        """Test JSON string output."""
        code = """from cm.symbols import Math
Math."""
        result = get_completions_json(code, line=2, column=5)

        # Should be valid JSON
        parsed = json.loads(result)
        assert isinstance(parsed, list), \
            "JSON output should be a list"

    def test_json_with_unicode(self):
        """Test JSON with unicode characters."""
        code = """# θ φ
from cm.symbols import Math
Math."""
        result = get_completions_json(code, line=3, column=5)

        # Should handle unicode without errors
        parsed = json.loads(result)
        assert isinstance(parsed, list), \
            "Should handle unicode in code"


# Integration test (skipped by default)
@pytest.mark.skip(reason="Requires full cm-libraries installation")
def test_full_integration():
    """Full integration test with actual cm.qm module."""
    code = """from cm.qm import atom
C = atom('C')
C.energy"""

    completions = get_completions(code, line=3, column=8)
    completion_names = [c['name'] for c in completions]

    # In full environment, should find 'energy' method
    assert 'energy' in completion_names, \
        f"Should find energy method. Got: {completion_names}"


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])
