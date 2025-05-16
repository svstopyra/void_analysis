import sympy as sp
from sympy import Matrix, symbols, simplify, trace, eye, det

# Symbols
x, y, z = symbols('x y z')
G, rho_bar, pi = sp.symbols('G rho_bar pi')

# Create symbolic matrices for ∇q Ψ^(n), n = 1 to 5
Psi = [Matrix(3, 3, lambda i, j: sp.Function(f'Psi^{(n)}_{i}{j}')(x, y, z)) for n in range(1, 6)]


#-------------------------------------------------------------------------------

import sympy as sp
import jupyprint as jp
from sympy import MatrixSymbol, symbols, Trace, expand, collect, latex, Mul
from collections import defaultdict

# Setup
epsilon = symbols('epsilon')
N = 5
Psi_n = [MatrixSymbol(f"Psi{n}", 3, 3) for n in range(1, N+1)]
Psi_full = sum((epsilon**n * Psi_n[n-1] for n in range(1, N+1)), start=sp.ZeroMatrix(3, 3))


def canonical_trace(expr):
    """Return Trace expression with factors reordered to canonical cyclic form."""
    if not isinstance(expr, Trace):
        return expr
    arg = expr.args[0]
    if not isinstance(arg, sp.MatMul):
        return expr
    factors = list(arg.args)
    cyclic_perms = [tuple(factors[i:] + factors[:i]) for i in range(len(factors))]
    canonical = min(cyclic_perms, key=lambda tup: tuple(str(f) for f in tup))
    return Trace(sp.MatMul(*canonical, evaluate=False))

def simplify_trace_terms(terms):
    """
    terms: list of (order, term) where term is ε^n * Trace(...)
    returns: dict of order -> simplified expression
    """
    grouped = defaultdict(list)
    for order, term in terms:
        coeff, trace_part = term.as_coeff_Mul()
        trace_expr = list(term.atoms(Trace))[0]  # get the Trace object
        canon = canonical_trace(trace_expr)
        grouped[order].append((coeff, canon))
    simplified = {}
    for order, termlist in grouped.items():
        term_dict = defaultdict(int)
        for coeff, trace in termlist:
            term_dict[trace] += coeff
        simplified[order] = sp.Add(*[c * t for t, c in term_dict.items() if c != 0])
        # Final simplification: combine repeated matrix products
        simplified[order] = sp.Add(*[
            coeff * simplify_matrix_powers(trace)
            for trace, coeff in term_dict.items() if coeff != 0
        ])
    return dict(sorted(simplified.items()))


def expand_matrix_power(Psi_list, power=2, max_order=None):
    from itertools import product
    terms = []
    indices = list(range(1, len(Psi_list) + 1))
    for idx_tuple in product(indices, repeat=power):
        eps_power = sum(idx_tuple)
        if max_order is not None and eps_power > max_order:
            continue  # skip higher-order terms
        mats = [Psi_list[i - 1] for i in idx_tuple]
        term = (epsilon ** eps_power) * Trace(sp.MatMul(*mats, evaluate=False))
        terms.append((eps_power, term))
    return terms

def simplify_matrix_powers(trace_expr):
    """Replace repeated products like A*A with A**2 inside Trace."""
    if not isinstance(trace_expr, Trace):
        return trace_expr
    arg = trace_expr.args[0]
    if not isinstance(arg, sp.MatMul):
        return trace_expr
    new_factors = []
    i = 0
    while i < len(arg.args):
        current = arg.args[i]
        count = 1
        while i + count < len(arg.args) and arg.args[i + count] == current:
            count += 1
        if count > 1:
            new_factors.append(current ** count)
        else:
            new_factors.append(current)
        i += count
    return Trace(sp.MatMul(*new_factors, evaluate=False))

def display_grouped_terms(terms):
    from collections import defaultdict
    grouped = defaultdict(list)
    for order, term in terms:
        grouped[order].append(term)
    for order in sorted(grouped):
        expr = sp.Add(*grouped[order])
        print(f"\\mathcal{{O}}(\\epsilon^{order}) :\n" + latex(expr) + "\\\\")

def display_simplified_terms(terms):
    simplified = simplify_trace_terms(terms)
    for order in sorted(simplified):
        print(f"\\mathcal{{O}}(\\epsilon^{order}) :\n" + latex(simplified[order]) + "\\\\")

max_eps_order = 5

for power in [2, 3]:
    raw_terms = expand_matrix_power(Psi_n, power=power, max_order=max_eps_order)
    print(f"\\textbf{{Trace}}\\left(\\Psi^{power}\\right) \\text{{ expansion up to }} \\mathcal{{O}}(\\epsilon^{max_eps_order}):\\\\")
    display_simplified_terms(raw_terms)


for power in [2, 3]:
    terms = expand_matrix_power(Psi_n, power=power,max_order=5)
    display_grouped_terms(terms)

#-------------------------------------------------------------------------------
# Trace Products:

def expand_trace_series(Psi_list, trace_power, max_order):
    """Returns list of (order, Trace(...)) for Tr(Psi^trace_power)"""
    raw_terms = expand_matrix_power(Psi_list, power=trace_power, max_order=max_order)
    return simplify_trace_terms(raw_terms)

def combine_trace_products(trace1, trace2, max_order):
    """
    trace1, trace2: dicts from order -> expression
    Returns: dict from order -> combined simplified expression
    """
    from collections import defaultdict
    result = defaultdict(lambda: 0)
    for o1, e1 in trace1.items():
        for o2, e2 in trace2.items():
            total_order = o1 + o2
            if total_order > max_order:
                continue
            result[total_order] += sp.expand(e1 * e2)
    return dict(sorted(result.items()))


# Expand Tr(Psi) and Tr(Psi^2)
trace1_terms = expand_trace_series(Psi_n, trace_power=1, max_order=5)
trace2_terms = expand_trace_series(Psi_n, trace_power=2, max_order=5)

# Multiply them term by term
combined = combine_trace_products(trace1_terms, trace2_terms, max_order=5)

# Display result
for order, expr in combined.items():
    print(f"\\operatorname{{Tr}}(\\Psi) \\cdot \\operatorname{{Tr}}(\\Psi^2) ~ \\mathcal{{O}}(\\epsilon^{order}):\n" + latex(expr) + "\\\\")

#-------------------------------------------------------------------------------
# Combined trace parser:

from itertools import product

def expand_trace_expression(expr, Psi_components, max_order=6):
    """
    Expand a symbolic expression involving Trace(Psi), Trace(Psi**n), etc.,
    where Psi = sum εⁿ Psi⁽ⁿ⁾, and return a dict mapping ε^order to simplified terms.
    
    Parameters:
        expr            -- symbolic expression using Trace(MatrixSymbol(...))
        Psi_components  -- list of Psi^(n) MatrixSymbols: [Psi1, Psi2, ...]
        max_order       -- maximum ε^n order to retain
    
    Returns:
        dict {order: simplified sympy.Expr}
    """
    epsilon = sp.symbols('epsilon')
    from collections import defaultdict
    def is_trace_power(term):
        return isinstance(term, Trace)
    def expand_term(term):
        """Expand a single Trace or product of Traces."""
        if isinstance(term, Trace):
            # Example: Tr(Psi**2)
            arg = term.args[0]
            if isinstance(arg, sp.MatPow):
                power = arg.exp
            elif isinstance(arg, sp.MatMul):
                power = len(arg.args)
            elif isinstance(arg, sp.MatrixSymbol):
                power = 1
            else:
                raise ValueError(f"Unsupported trace argument: {arg}")
            return expand_trace_series(Psi_components, power, max_order)
        elif isinstance(term, sp.Mul):
            # Example: Tr(Psi^2) * Tr(Psi^3)
            trace_factors = [expand_term(f) for f in term.args if isinstance(f, Trace)]
            non_trace_factors = [f for f in term.args if not isinstance(f, Trace)]
            # Start combining trace expansions
            if len(trace_factors) == 2:
                result = combine_trace_products(trace_factors[0], trace_factors[1], max_order)
            else:
                raise NotImplementedError("Only supports products of two traces currently.")
            # Multiply non-trace factors in after
            if non_trace_factors:
                for order in result:
                    result[order] = sp.Mul(sp.Mul(*non_trace_factors), result[order])
            return result
        elif isinstance(term, sp.Add):
            result = defaultdict(lambda: 0)
            for sub in term.args:
                for k, v in expand_term(sub).items():
                    result[k] += v
            return dict(result)
        else:
            # Constant (not a Trace or Trace product)
            return {0: term}
    # Fully expand expression into sum of additive terms
    total = defaultdict(lambda: 0)
    expanded_expr = sp.expand(expr)
    # Recursively expand each term
    expanded_result = expand_term(expanded_expr)
    # Merge results up to max_order
    for k, v in expanded_result.items():
        if k <= max_order:
            total[k] += v
    # Final simplification (optional)
    return {k: sp.simplify(v) for k, v in sorted(total.items())}

def expand_trace_expression(expr, Psi_components, max_order=6):
    from collections import defaultdict
    epsilon = sp.symbols('epsilon')
    def expand_term(term):
        if isinstance(term, Trace):
            arg = term.args[0]
            if isinstance(arg, sp.MatPow):
                power = arg.exp
            elif isinstance(arg, sp.MatMul):
                power = len(arg.args)
            elif isinstance(arg, sp.MatrixSymbol):
                power = 1
            else:
                raise ValueError(f"Unsupported trace argument: {arg}")
            return expand_trace_series(Psi_components, power, max_order)
        elif isinstance(term, sp.Mul):
            trace_factors = [expand_term(f) for f in term.args if isinstance(f, Trace)]
            non_trace_factors = [f for f in term.args if not isinstance(f, Trace)]
            # Start combining trace expansions (pairwise only for now)
            if len(trace_factors) == 2:
                result = combine_trace_products(trace_factors[0], trace_factors[1], max_order)
            elif len(trace_factors) == 1:
                result = trace_factors[0]
            else:
                raise NotImplementedError("More than 2 trace factors not supported yet.")
            if non_trace_factors:
                for k in result:
                    result[k] = sp.Mul(*non_trace_factors) * result[k]
            return result
        elif isinstance(term, sp.Add):
            result = defaultdict(lambda: 0)
            for arg in term.args:
                sub = expand_term(arg)
                for k, v in sub.items():
                    result[k] += v
            return dict(result)
        elif isinstance(term, sp.Number):
            return {0: term}
        else:
            return {0: term}
    # Expand the input expression symbolically
    expr = sp.sympify(expr).expand()
    result = expand_term(expr)
    # Final cleanup
    return {k: sp.simplify(v) for k, v in sorted(result.items()) if k <= max_order}


def expand_trace_expression(expr, Psi_components, max_order=6):
    from collections import defaultdict
    from itertools import product
    epsilon = sp.symbols('epsilon')
    # Cache trace expansions
    trace_cache = {}
    def expand_single_trace(trace_expr):
        arg = trace_expr.args[0]
        if isinstance(arg, sp.MatPow):
            power = arg.exp
        elif isinstance(arg, sp.MatMul):
            power = len(arg.args)
        elif isinstance(arg, sp.MatrixSymbol):
            power = 1
        else:
            raise ValueError(f"Unsupported trace argument: {arg}")
        key = ('Trace', power)
        if key in trace_cache:
            return trace_cache[key]
        terms = expand_trace_series(Psi_components, power, max_order)
        trace_cache[key] = terms
        return terms
    def recursive_expand(expr):
        if isinstance(expr, Trace):
            return expand_single_trace(expr)
        elif isinstance(expr, sp.Number):
            return {0: expr}
        elif isinstance(expr, sp.Symbol):
            return {0: expr}
        elif isinstance(expr, sp.Add):
            result = defaultdict(lambda: 0)
            for arg in expr.args:
                sub = recursive_expand(arg)
                for k, v in sub.items():
                    result[k] += v
            return dict(result)
        elif isinstance(expr, sp.Mul):
            parts = [recursive_expand(arg) for arg in expr.args]
            result = defaultdict(lambda: 0)
            for combo in product(*[list(p.items()) for p in parts]):
                total_order = sum(o for o, _ in combo)
                if total_order <= max_order:
                    value = sp.Mul(*[v for _, v in combo])
                    result[total_order] += value
            return dict(result)
        else:
            return {0: expr}
    # Expand everything structurally
    expr = sp.sympify(expr).expand()
    result = recursive_expand(expr)
    # Simplify each term
    return {k: sp.simplify(v) for k, v in sorted(result.items()) if k <= max_order}


def expand_trace_expression(expr, Psi_components, max_order=6):
    """
    Expand any expression involving Trace(Psi), Trace(Psi^n), etc., where
    Psi = εΨ1 + ε²Ψ2 + ..., and collect terms by ε^order.
    """
    from collections import defaultdict
    from itertools import product
    epsilon = sp.symbols('epsilon')
    trace_cache = {}
    # --- Expand a single trace like Trace(Psi^2)
    def expand_single_trace(trace_expr):
        arg = trace_expr.args[0]
        if isinstance(arg, sp.MatPow):
            power = arg.exp
        elif isinstance(arg, sp.MatMul):
            power = len(arg.args)
        elif isinstance(arg, sp.MatrixSymbol):
            power = 1
        else:
            raise ValueError(f"Unsupported trace argument: {arg}")
        key = ('Trace', power)
        if key in trace_cache:
            return trace_cache[key]
        terms = expand_trace_series(Psi_components, power, max_order)
        trace_cache[key] = terms
        return terms
    # --- Recursive expansion of a single term
    def recursive_expand(term):
        if isinstance(term, Trace):
            return expand_single_trace(term)
        elif isinstance(term, sp.Number):
            return {0: term}
        elif isinstance(term, sp.Symbol):
            return {0: term}
        elif isinstance(term, sp.Add):
            result = defaultdict(lambda: 0)
            for arg in term.args:
                sub = recursive_expand(arg)
                for k, v in sub.items():
                    result[k] += v
            return dict(result)
        elif isinstance(term, sp.Pow):
            base, exp = term.args
            # Carefully unroll powers of traces only when exp is a concrete integer > 1
            if isinstance(base, Trace) and isinstance(exp, sp.Integer) and exp.is_positive:
                # Avoid infinite recursion by directly creating the multiplication object
                prod = sp.Mul.fromiter([base] * exp, evaluate=False)
                return recursive_expand(prod)
            else:
                return {0: term}
        elif isinstance(term, sp.Mul):
            parts = [recursive_expand(arg) for arg in term.args]
            result = defaultdict(lambda: 0)
            for combo in product(*[p.items() for p in parts]):
                order = sum(o for o, _ in combo)
                if order > max_order:
                    continue
                value = sp.Mul(*[v for _, v in combo])
                result[order] += value
            return dict(result)
        else:
            return {0: term}
    # --- Step 1: flatten top-level sum into additive terms
    expr = sp.sympify(expr).expand()
    additive_terms = expr.as_ordered_terms()  # ensures clean term-by-term
    # --- Step 2: expand each term and collect results
    final_result = defaultdict(lambda: 0)
    term_expansion_list = []
    for term in additive_terms:
        term_expansion = recursive_expand(term)
        term_expansion_list.append(term_expansion)
        for order, value in term_expansion.items():
            final_result[order] += value
    # --- Step 3: cleanup and return
    return {k: sp.simplify(v) for k, v in sorted(final_result.items()) if k <= max_order}






epsilon = symbols('epsilon')
Psi = MatrixSymbol("Psi", 3, 3)
Psi_n = [MatrixSymbol(f"Psi{n}", 3, 3) for n in range(1, 6)]

# Expression involving Ψ (which we symbolically replace with Ψ = εΨ₁ + ε²Ψ₂ + ...)
#expr = Trace(Psi**2) + Trace(Psi)**2 + Trace(Psi**3)
#expr = (Trace(Psi)**2 - Trace(Psi**2))/2
expr = Trace(Psi)*Trace(Psi**2)*Trace(Psi**3)


expr = Trace(Psi) + (Trace(Psi)**2 - Trace(Psi**2))/2 + (Trace(Psi)**3 - 3*Trace(Psi)*Trace(Psi**2) + 2*Trace(Psi**3))/6

#expr1 = Trace(Psi**2)
#expr2 = Trace(Psi**3)
#expr3 = Trace(Psi**2) + Trace(Psi**3)
#expr4 = Trace(Psi)**2



# Expand!
result = expand_trace_expression(expr, Psi_components=Psi_n, max_order=5)
#result1 = expand_trace_expression(expr1, Psi_components=Psi_n, max_order=5)
#result2 = expand_trace_expression(expr2, Psi_components=Psi_n, max_order=5)
#result3 = expand_trace_expression(expr3, Psi_components=Psi_n, max_order=5)
#result4 = expand_trace_expression(expr4, Psi_components=Psi_n, max_order=5)

def split_latex_expression(expr_latex, max_line_len=80):
    """
    Break a long LaTeX expression into multiple lines.
    - Inserts '\\' at top-level '+' or '-' boundaries
    - Respects bracketed expressions
    - Good for align* or equation environments
    """
    breaks = []
    depth = 0
    last_break = 0
    i = 0
    while i < len(expr_latex):
        c = expr_latex[i]
        # Update depth (for \left( and \right) also)
        if c in ['{', '[', '(']:
            depth += 1
        elif c in ['}', ']', ')']:
            depth -= 1
        # Heuristic break point at top-level + or -
        if depth == 0 and c in ['+', '-'] and (i - last_break) > max_line_len:
            breaks.append(i)
            last_break = i
        i += 1
    # If no breaks found, return as is
    if not breaks:
        return expr_latex
    # Build new string with breaks
    lines = []
    prev = 0
    for b in breaks:
        lines.append(expr_latex[prev:b].rstrip())
        prev = b
    lines.append(expr_latex[prev:].rstrip())
    return ' \\\\\n'.join(lines)


def display_result(result,split=False,max_line_len=80):
    for order, val in result.items():
        latex_text = latex(val)
        if split:
            latex_test = split_latex_expression(
                latex_text,max_line_len=max_line_len
            )
        print(f"\\mathcal{{O}}(\\epsilon^{order}):\n" + latex(val) + "\\\\")

# Step 4: Display
display_result(result,split=True)


#-------------------------------------------------------------------------------
# Cofactor matrix expansion:

import sympy as sp
from collections import defaultdict

# Setup
epsilon = sp.symbols('epsilon')
x, y, z = sp.symbols('x y z')
coords = [x, y, z]
d = 3
N = 5  # Maximum perturbation order

# Step 1: Define Psi^(n) matrices with function entries
Psi_orders = []
for n in range(1, N + 1):
    Psi_n = sp.Matrix(d, d, lambda i, j: sp.Function(f"psi{n}_{i}{j}")(*coords))
    Psi_orders.append(Psi_n)

# Step 2: Construct full Psi gradient and F = I + grad Psi
Psi_total = sum((epsilon**n * Psi_orders[n - 1] for n in range(1, N + 1)), start=sp.zeros(d, d))
F = sp.eye(d) + Psi_total

# Step 3: Compute cofactor matrix: cof(F) = adj(F).T
C = F.adjugate().T  # Note: adjugate is cof(F)^T

# Step 4: Expand and collect by powers of epsilon
def expand_cofactor_by_order(C, max_order=5):
    collected = defaultdict(lambda: sp.zeros(d, d))
    for i in range(d):
        for j in range(d):
            expanded = sp.expand(C[i, j])
            grouped = sp.collect(expanded, epsilon, evaluate=False)
            for key, val in grouped.items():
                if key == 1:
                    order = 0
                elif isinstance(key, sp.Pow) and key.base == epsilon:
                    order = int(key.exp)
                elif key == epsilon:
                    order = 1
                else:
                    continue
                if order <= max_order:
                    collected[order][i, j] += val
    return dict(sorted(collected.items()))

cofactors_by_order = expand_cofactor_by_order(C, max_order=N)

for order, matrix in cofactors_by_order.items():
    print(f"\\text{{Cofactor matrix }} C^{{({order})}} ~ \\mathcal{{O}}(\\epsilon^{order}):\n" + sp.latex(matrix) + "\\\\")




