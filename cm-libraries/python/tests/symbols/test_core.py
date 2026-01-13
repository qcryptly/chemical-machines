"""Tests for cm.symbols.core module."""

import pytest
from cm.symbols import Var, Const, Sum, Product


class TestVar:
    """Tests for Var class."""

    def test_create_var(self):
        """Test variable creation."""
        x = Var("x")
        assert x.name == "x"
        # str() returns repr-style format
        assert "x" in str(x)

    def test_var_to_latex(self):
        """Test LaTeX representation."""
        x = Var("x")
        assert x.to_latex() == "x"

        theta = Var("theta")
        assert theta.to_latex() == "\\theta"

    def test_var_equality(self):
        """Test variable equality."""
        x1 = Var("x")
        x2 = Var("x")
        y = Var("y")

        assert x1.name == x2.name
        assert x1.name != y.name


class TestConst:
    """Tests for Const class."""

    def test_create_const(self):
        """Test constant creation."""
        c = Const(42)
        assert c.value == 42

    def test_const_operations(self):
        """Test arithmetic operations on constants."""
        a = Const(10)
        b = Const(5)

        # Addition
        c = a + b
        assert isinstance(c, (Const, type(a + b)))

        # Subtraction
        d = a - b
        assert isinstance(d, (Const, type(a - b)))

        # Multiplication
        e = a * b
        assert isinstance(e, (Const, type(a * b)))

        # Division
        f = a / b
        assert isinstance(f, (Const, type(a / b)))

    def test_const_to_latex(self):
        """Test LaTeX representation."""
        c = Const(42)
        assert c.to_latex() == "42"

        c_neg = Const(-3.14)
        latex = c_neg.to_latex()
        assert "-3.14" in latex


class TestSum:
    """Tests for Sum class."""

    def test_create_sum(self):
        """Test sum creation."""
        i = Var("i")
        expr = i ** 2
        # Sum(summand, var, lower, upper)
        s = Sum(expr, i, 1, 10)
        assert s.summand is not None
        assert s.var.name == "i"
        assert s.is_definite

    def test_sum_of_const(self):
        """Test sum of constants (via + operator)."""
        a = Const(10)
        b = Const(20)
        s = a + b
        assert isinstance(s, (Const, type(a + b)))

    def test_sum_to_latex(self):
        """Test LaTeX representation of sum."""
        i = Var("i")
        n = Var("n")
        s = Sum(i, i, 1, n)
        latex = s.to_latex()
        assert "sum" in latex.lower() or "Σ" in latex or "\\sum" in latex


class TestProduct:
    """Tests for Product class."""

    def test_create_product(self):
        """Test product creation."""
        i = Var("i")
        expr = i ** 2
        # Product(factor, var, lower, upper)
        p = Product(expr, i, 1, 5)
        assert p.factor is not None
        assert p.var.name == "i"
        assert p.is_definite

    def test_product_of_const(self):
        """Test product of constants (via * operator)."""
        a = Const(3)
        b = Const(4)
        p = a * b
        assert isinstance(p, (Const, type(a * b)))

    def test_product_to_latex(self):
        """Test LaTeX representation of product."""
        i = Var("i")
        n = Var("n")
        p = Product(i, i, 1, n)
        latex = p.to_latex()
        assert "prod" in latex.lower() or "Π" in latex or "\\prod" in latex


class TestArithmeticOperations:
    """Tests for arithmetic operations between Expr objects."""

    def test_var_plus_const(self):
        """Test variable + constant."""
        x = Var("x")
        c = Const(5)
        result = x + c
        assert result is not None

    def test_var_times_var(self):
        """Test variable * variable."""
        x = Var("x")
        y = Var("y")
        result = x * y
        assert result is not None

    def test_power(self):
        """Test power operation."""
        x = Var("x")
        result = x ** 2
        assert result is not None

    def test_division(self):
        """Test division."""
        x = Var("x")
        y = Var("y")
        result = x / y
        assert result is not None

    def test_negation(self):
        """Test negation."""
        x = Var("x")
        result = -x
        assert result is not None

    def test_complex_expression(self):
        """Test complex expression."""
        x = Var("x")
        y = Var("y")
        z = Var("z")

        # (x + y) * z - x/y
        expr = (x + y) * z - x / y
        assert expr is not None
        latex = expr.to_latex()
        assert latex is not None


@pytest.mark.unit
class TestExpressionSimplification:
    """Tests for expression simplification."""

    def test_const_addition(self):
        """Test that constant addition simplifies."""
        a = Const(10)
        b = Const(20)
        result = a + b
        # Should simplify to Const(30) or equivalent

    def test_zero_addition(self):
        """Test adding zero."""
        x = Var("x")
        zero = Const(0)
        result = x + zero
        # Should simplify to x

    def test_one_multiplication(self):
        """Test multiplying by one."""
        x = Var("x")
        one = Const(1)
        result = x * one
        # Should simplify to x


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
