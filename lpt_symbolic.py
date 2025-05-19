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

def expand_matrix_power_general(matrix_order_map, power=2, max_order=None):
    """
    Generalized matrix power expansion.
    - matrix_order_map: dict {MatrixSymbol: order in ε}
    - power: number of matrices in product (e.g. 2 for Tr(AB))
    - Returns: list of (ε-order, Trace(...))
    """
    from itertools import product
    symbols = list(matrix_order_map.keys())
    terms = []
    for combo in product(symbols, repeat=power):
        eps_order = sum(matrix_order_map[M] for M in combo)
        if max_order is not None and eps_order > max_order:
            continue
        expr = Trace(sp.MatMul(*combo, evaluate=False))
        terms.append((eps_order, epsilon**eps_order * expr))
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

def expand_trace_series_general(matrix_order_map, trace_power, max_order):
    raw_terms = expand_matrix_power_general(matrix_order_map, trace_power, max_order)
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

#def trace_of_cof_times_B(A, B):
#    return (1/2) * (Trace(A)**2 - Trace(A @ A)) * Trace(B) - Trace(A) * Trace(A @ B) + Trace(A @ A @ B)

def trace_of_cof_times_B(A, B):
    """
    Compute Tr[cof(A) · B] using Cayley–Hamilton identity:
    Tr[cof(A) · B] = ½(Tr(A)^2 - Tr(A²)) Tr(B) - Tr(A) Tr(A B) + Tr(A² B)
    
    A and B should be symbolic matrix expressions (not summed yet).
    Returns: symbolic expression involving only Trace(Mul(...)) terms
    """
    A_tr = Trace(A)
    A_sq = sp.MatMul(A, A, evaluate=False)
    AB = sp.MatMul(A, B, evaluate=False)
    A2B = sp.MatMul(A_sq, B, evaluate=False)
    # Structure: don't square Trace(A), multiply them as separate Tr(A) * Tr(A)
    A_tr_squared = sp.Mul(A_tr, A_tr, evaluate=False)
    term1 = sp.Mul(1/2, sp.Add(A_tr_squared, sp.Mul(-1, Trace(A_sq)), evaluate=False), Trace(B), evaluate=False)
    term2 = sp.Mul(-1, A_tr, Trace(AB), evaluate=False)
    term3 = Trace(A2B)
    return sp.Add(term1, term2, term3, evaluate=False)

import sympy as sp
from sympy import symbols, MatrixSymbol, Trace, MatMul, simplify, latex
from collections import defaultdict
from itertools import product
import jupyprint as jp

# --- Setup
epsilon = symbols('epsilon')
d = 3
N = 5  # Max order

# Psi components
Psi_n = [MatrixSymbol(f"Psi{n}", d, d) for n in range(1, N+1)]
D2_Psi_n = [MatrixSymbol(f"D2_Psi{n}", d, d) for n in range(1, N+1)]  # Laplacian-like terms (symbolic)

# Full symbolic fields
Psi_full = sum(((epsilon**(n+1)) * Psi_n[n] for n in range(N)),start=sp.ZeroMatrix(3, 3),evaluate=False)
D2_Psi_full = sum(((epsilon**(n+1)) * D2_Psi_n[n] for n in range(N)),start=sp.ZeroMatrix(3, 3),evaluate=False)

# --- Define the trace identity:
#def trace_of_cof_times_B(A, B):
#    """Implements Tr[cof(A) · B] using Cayley–Hamilton / Newton identity"""
#    A2 = MatMul(A, A, evaluate=False)
#    AB = MatMul(A, B, evaluate=False)
#    AAB = MatMul(A2, B, evaluate=False)
#    return (1/2)*(Trace(A)**2 - Trace(A2))*Trace(B) - Trace(A)*Trace(AB) + Trace(AAB)



from collections import defaultdict

def expand_trace_expression(expr, matrix_order_map, max_order=5):
    """Expands a symbolic trace expression in powers of epsilon using Psi = sum ε^n Psi⁽ⁿ⁾"""
    trace_cache = {}
    def expand_single_trace(trace_expr):
        arg = trace_expr.args[0]
        # If argument is a MatrixSymbol, treat as-is
        if isinstance(arg, sp.MatrixSymbol):
            order = matrix_order_map.get(arg, 0)
            return {order: epsilon**order * trace_expr}
        # If it's a MatMul or composite, recursively expand
        elif isinstance(arg, sp.MatMul):
            factors = arg.args
            parts = [recursive_expand(f) for f in factors]
            result = defaultdict(lambda: 0)
            for combo in product(*[p.items() for p in parts]):
                order = sum(o for o, _ in combo)
                if order > max_order:
                    continue
                mul = sp.MatMul(*[v for _, v in combo], evaluate=False)
                result[order] += epsilon**order * Trace(mul)
            return dict(result)
        # If it's a sum or unknown expression, fallback (not ideal)
        else:
            return {0: trace_expr}
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
                for k, v in recursive_expand(arg).items():
                    result[k] += v
            return dict(result)
        elif isinstance(term, sp.Mul):
            parts = [recursive_expand(arg) for arg in term.args]
            result = defaultdict(lambda: 0)
            for combo in product(*[list(p.items()) for p in parts]):
                order = sum(o for o, _ in combo)
                if order <= max_order:
                    value = sp.Mul(*[v for _, v in combo])
                    result[order] += value
            return dict(result)
        elif isinstance(term, sp.Pow):
            base, exp = term.args
            if isinstance(base, Trace) and isinstance(exp, sp.Integer) and exp.is_positive:
                repeated = sp.Mul.fromiter([base] * exp, evaluate=False)
                return recursive_expand(repeated)
            else:
                return {0: term}
        else:
            return {0: term}
    expr = sp.sympify(expr).expand()
    additive_terms = expr.as_ordered_terms()
    total = defaultdict(lambda: 0)
    for term in additive_terms:
        for k, v in recursive_expand(term).items():
            total[k] += v
    return {k: sp.simplify(v) for k, v in sorted(total.items()) if k <= max_order}

def expand_trace_additivity(expr):
    """
    Expand arguments inside Trace(...) using linearity and distributivity.
    Ensures all Trace(...) wrap simple matrix products.
    """
    def expand_trace(trace_expr):
        arg = trace_expr.args[0]
        expanded_arg = sp.expand(arg, deep=True, multinomial=True)
        if isinstance(expanded_arg, sp.Add):
            return sp.Add(*[Trace(term) for term in expanded_arg.args])
        else:
            return Trace(expanded_arg)
    if not hasattr(expr, 'args') or isinstance(expr, (sp.Number, sp.Symbol)):
        return expr
    if isinstance(expr, Trace):
        return expand_trace(expr)
    return expr.func(*[expand_trace_additivity(arg) for arg in expr.args])

def extract_epsilon_order(expr):
    """
    Given a Mul like ε^n * A * ε^m * B, extract total ε-order and the symbolic product.
    Returns: (epsilon_order, matrix_expr)
    """
    if not isinstance(expr, sp.MatMul):
        return (0, expr)
    eps_order = 0
    non_eps_factors = []
    for factor in expr.args:
        if factor.has(epsilon):
            coeff = factor.as_coeff_exponent(epsilon)
            if coeff[0] == 1 and coeff[1] != 0:
                eps_order += coeff[1]
            else:
                non_eps_factors.append(factor)
        else:
            non_eps_factors.append(factor)
    return (int(eps_order), sp.MatMul(*non_eps_factors, evaluate=False))

def factor_epsilon_in_traces(expr):
    """
    Replace Trace(epsilon^n * A * epsilon^m * B ...) with epsilon^k * Trace(A B ...)
    """
    if isinstance(expr, Trace):
        arg = expr.args[0]
        eps_order, matrix_expr = extract_epsilon_order(arg)
        return epsilon**eps_order * Trace(matrix_expr)
    if not hasattr(expr, 'args') or isinstance(expr, (sp.Number, sp.Symbol)):
        return expr
    return expr.func(*[factor_epsilon_in_traces(arg) for arg in expr.args])


# Construct symbolic expression: Tr[cof(∇Ψ) · D2Ψ]
cof_trace_expr = trace_of_cof_times_B(Psi_full, D2_Psi_full)
additive_expanded = expand_trace_additivity(cof_trace_expr)
fully_expanded = sp.expand(additive_expanded, deep=True, multinomial=True)
factored = factor_epsilon_in_traces(fully_expanded)
result = expand_trace_expression(factored, matrix_order_map, max_order=5)

# Expand it
#all_components = Psi_n + D2_Psi_n

# Build matrix-order map
matrix_order_map = {Psi_n[i]: i+1 for i in range(N)}
matrix_order_map.update({D2_Psi_n[i]: i+1 for i in range(N)})

# Use general version
#expanded = expand_trace_expression(expr, matrix_order_map, max_order=N)



expanded = expand_trace_expression(fully_expanded_expr, matrix_order_map, max_order=N)

# Display each order
for order, term in expanded.items():
    print(f"\\mathcal{{O}}(\\epsilon^{order}):\n" + latex(term) + "\\\\")


#-------------------------------------------------------------------------
# More cohesive pipeline:

# Patch: epsilon defined inside the function
# Updated version of epsilon_expand that correctly tracks epsilon powers in full terms

import sympy as sp
from sympy import MatrixSymbol, Trace
from collections import defaultdict
from itertools import product
from functools import reduce
import operator

def epsilon_expand_with_true_orders(expr, expand_symbols, max_order=5, start_orders=None):
    from sympy import MatrixSymbol, Symbol, expand, Trace
    from functools import reduce
    from collections import defaultdict
    epsilon = Symbol('epsilon')
    if start_orders is None:
        start_orders = {s: 1 for s in expand_symbols}
    # Step 1: Expand each matrix symbol into its series
    symbol_map = {}
    name_to_order = {}
    for name in expand_symbols:
        start = start_orders.get(name, 1)
        mats = []
        for i in range(start, max_order + 1):
            mat = MatrixSymbol(f"{name}{i}", 3, 3)
            mats.append(epsilon**i * mat)
            name_to_order[mat] = i
        expansion = reduce(lambda x, y: x + y, mats)
        symbol_map[name] = expansion
    substituted_expr = expr
    for name, expansion in symbol_map.items():
        base = MatrixSymbol(name, 3, 3)
        substituted_expr = substituted_expr.subs(base, expansion)
    # Step 2: Expand the full expression symbolically
    expanded_expr = expand(substituted_expr, deep=True, multinomial=True)
    # Step 3: Expand traces over sums
    def distribute_trace(expr):
        if isinstance(expr, Trace):
            inner = expand(expr.args[0], deep=True, multinomial=True)
            if isinstance(inner, sp.Add):
                return sp.Add(*[Trace(term) for term in inner.args])
            return Trace(inner)
        elif isinstance(expr, sp.Add):
            return sp.Add(*[distribute_trace(arg) for arg in expr.args])
        elif isinstance(expr, sp.Mul):
            return sp.Mul(*[distribute_trace(arg) for arg in expr.args], evaluate=False)
        return expr
    additive_expr = distribute_trace(expanded_expr)
    terms = additive_expr.args if isinstance(additive_expr, sp.Add) else [additive_expr]
    # Step 4: Classify terms by symbolic epsilon order from matrix labels
    def compute_true_order(term):
        if isinstance(term, Trace):
            return compute_true_order(term.args[0])
        elif isinstance(term, sp.Mul):
            return sum(compute_true_order(arg) for arg in term.args)
        elif isinstance(term, sp.Add):
            return max(compute_true_order(arg) for arg in term.args)
        elif isinstance(term, sp.Pow) and isinstance(term.base, sp.MatrixSymbol):
            base_order = name_to_order.get(term.base, 0)
            return base_order * int(term.exp)
        elif isinstance(term, sp.MatrixSymbol):
            return name_to_order.get(term, 0)
        return 0
    grouped = defaultdict(lambda: 0)
    for term in terms:
        order = compute_true_order(term)
        if order <= max_order:
            grouped[order] += term
    return dict(sorted(grouped.items()))

def extract_scalars_from_traces(expr, scalars):
    """Pull scalar factors like ε or λ outside trace expressions."""
    if isinstance(expr, Trace):
        arg = expr.args[0]
        if isinstance(arg, sp.Mul):
            scalar_factors = []
            matrix_factors = []
            for factor in arg.args:
                if any(factor.has(s) for s in scalars):
                    scalar_factors.append(factor)
                else:
                    matrix_factors.append(factor)
            return sp.Mul(*scalar_factors) * Trace(sp.Mul(*matrix_factors, evaluate=False))
        return expr
    elif isinstance(expr, sp.Add):
        return sp.Add(*[extract_scalars_from_traces(arg, scalars) for arg in expr.args])
    elif isinstance(expr, sp.Mul):
        return sp.Mul(*[extract_scalars_from_traces(arg, scalars) for arg in expr.args], evaluate=False)
    return expr

def multinomial_trace_expand(symbol_orders, structure, max_order=5):
    """
    Expand Tr(A^n B^m C^k ...) given structure = ["A", "A", "B", "B", "C", "C", "C"]
    and a dictionary of series: {"A": [A1, A2, ...], "B": [B1, ...]}
    Returns a dict {order: expression} for each ε^order term.
    """
    epsilon = sp.Symbol("epsilon")  # ensure epsilon is always defined
    grouped = defaultdict(lambda: 0)
    # Define index ranges for each element of the trace structure
    index_ranges = [range(1, len(symbol_orders[sym]) + 1) for sym in structure]
    for idx_combo in product(*index_ranges):
        eps_power = sum(idx_combo)
        if eps_power > max_order:
            continue
        matrices = [symbol_orders[sym][idx - 1] for sym, idx in zip(structure, idx_combo)]
        trace_term = Trace(sp.MatMul(*matrices, evaluate=False))
        grouped[eps_power] += epsilon**eps_power * trace_term
    return dict(sorted(grouped.items()))


# Universal dispatcher to expand any expression in epsilon, handling traces, products, sums
def epsilon_expand(expr, expand_symbols, max_order=5, start_orders=None):
    epsilon = sp.Symbol("epsilon")
    if start_orders is None:
        start_orders = {s: 1 for s in expand_symbols}
    # Step 1: Create matrix series for each symbol
    symbol_orders = {}
    for name in expand_symbols:
        start = start_orders.get(name, 1)
        matrices = [MatrixSymbol(f"{name}{i}", 3, 3) for i in range(start, max_order + 1)]
        symbol_orders[name] = matrices
    def expand_recursive(term):
        # Case: single trace of a product (e.g., Tr(A*A*B))
        if isinstance(term, Trace):
            inner = term.args[0]
            if isinstance(inner, sp.MatMul):
                structure = []
                for arg in inner.args:
                    if isinstance(arg, MatrixSymbol):
                        for base_name, series in symbol_orders.items():
                            if arg.name.startswith(base_name):
                                structure.append(base_name)
                                break
                if structure:
                    return multinomial_trace_expand(symbol_orders, structure, max_order=max_order)
            elif isinstance(inner, MatrixSymbol):
                return multinomial_trace_expand(symbol_orders, [inner.name[0]], max_order=max_order)
            elif isinstance(inner, sp.Pow):
                base = inner.base
                exp = inner.exp
                if isinstance(base, MatrixSymbol) and isinstance(exp, int):
                    return multinomial_trace_expand(symbol_orders, [base.name[0]] * exp, max_order=max_order)
            return {0: term}
        # Case: Add
        elif isinstance(term, sp.Add):
            result = defaultdict(lambda: 0)
            for arg in term.args:
                sub = expand_recursive(arg)
                for k, v in sub.items():
                    result[k] += v
            return dict(result)
        # Case: Mul
        elif isinstance(term, sp.Mul):
            parts = [expand_recursive(arg) for arg in term.args]
            result = defaultdict(lambda: 0)
            for combo in product(*[list(p.items()) for p in parts]):
                order = sum(o for o, _ in combo)
                if order <= max_order:
                    value = sp.Mul(*[v for _, v in combo])
                    result[order] += value
            return dict(result)
        # Case: scalar or unknown
        return {0: term}
    # Run expansion
    expanded = expand_recursive(sp.sympify(expr))
    # Final scalar cleanup
    return {
        k: extract_scalars_from_traces(sp.simplify(v), scalars=[epsilon])
        for k, v in sorted(expanded.items())
    }

# ---------------- Test All Cases ----------------
# Reconstruct matrix symbols
A_series = [MatrixSymbol(f"A{i}", 3, 3) for i in range(1, 5)]
B_series = [MatrixSymbol(f"B{i}", 3, 3) for i in range(1, 5)]
C_series = [MatrixSymbol(f"C{i}", 3, 3) for i in range(1, 5)]

# Setup base symbols
A, B, C = MatrixSymbol("A", 3, 3), MatrixSymbol("B", 3, 3), MatrixSymbol("C", 3, 3)
lam = sp.Symbol("λ")

# Test expressions
tests = {
    "Case 1: Tr(AB)": Trace(A * B),
    "Case 2: Tr(A^2)": Trace(A * A),
    "Case 3: Tr(A^2) + Tr(AB)": Trace(A * A) + Trace(A * B),
    "Case 4: Tr(A^2 B)": Trace(A * A * B),
    "Case 5: Tr(A) Tr(B)": Trace(A) * Trace(B),
    "Case 6: Tr(A^2) Tr(AB)": Trace(A * A) * Trace(A * B),
    "Case 7: Tr(BA)": Trace(B * A),
    "Case 8: Tr(ABA)": Trace(A * B * A),
    "Case 9: Tr(A)^2": Trace(A)**2,
    "Case 10: 3/2 Tr(AB) − 5 Tr(A^2)": (3/2) * Trace(A * B) - 5 * Trace(A * A),
    "Case 11: Tr(A+B)": Trace(A + B),
    "Case 12: λ Tr(AB)": lam * Trace(A * B),
    "Case 13: Tr(A^2 B^2 C^3)": Trace(A * A * B * B * C * C * C)
}

# Run expansions and display
test_outputs = {}
for name, expr in tests.items():
    expanded = epsilon_expand(expr, expand_symbols=["A", "B", "C"], max_order=5)
    test_outputs[name] = expanded

# Display all results in a single DataFrame
from pandas import DataFrame
rows = []
for test_name, result in test_outputs.items():
    for order, val in result.items():
        rows.append((test_name, f"ε^{order}", val))







# Define scalar and matrices
lam = sp.Symbol("λ")
A = MatrixSymbol("A", 3, 3)
B = MatrixSymbol("B", 3, 3)

# Test expression
expr = Trace(lam * A * B)

# Run the epsilon expansion with scalar extraction
intermediate_result = epsilon_expand_with_true_orders(expr, expand_symbols=["A", "B"], max_order=4)

# Clean up traces by extracting λ and epsilon if needed
final_result = {order: extract_scalars_from_traces(val, scalars=[lam, sp.Symbol("epsilon")])
                for order, val in intermediate_result.items()}

# Declare matrix symbols
A_series = [MatrixSymbol(f"A{i}", 3, 3) for i in range(1, 5)]
B_series = [MatrixSymbol(f"B{i}", 3, 3) for i in range(1, 5)]
C_series = [MatrixSymbol(f"C{i}", 3, 3) for i in range(1, 5)]

symbol_orders = {"A": A_series, "B": B_series, "C": C_series}
structure = ["A", "A", "B", "B", "C", "C", "C"]

result = multinomial_trace_expand(symbol_orders, structure, max_order=9)

# Optional: clean up epsilon in traces
epsilon = sp.Symbol("epsilon")
cleaned = {
    order: extract_scalars_from_traces(val, scalars=[epsilon])
    for order, val in result.items()
}

for order, expr in cleaned.items():
    print(f"Order ε^{order}: {expr}")





A = MatrixSymbol("A", 3, 3)
B = MatrixSymbol("B", 3, 3)
expr = Trace(A @ B)

result = epsilon_expand_with_true_orders(expr, expand_symbols=["A", "B"], max_order=4)

df = pd.DataFrame.from_dict(test_result, orient='index', columns=["Corrected ε-order Expansion"])
tools.display_dataframe_to_user("Corrected Epsilon Expansion of Tr(AB)", df)


# All test cases:

# Re-run test suite
lam = sp.Symbol("λ")
A = MatrixSymbol("A", 3, 3)
B = MatrixSymbol("B", 3, 3)

test_cases = {
    "Case 1: Tr(AB)": Trace(A * B),
    "Case 2: Tr(A^2)": Trace(A * A),
    "Case 3: Tr(A^2) + Tr(AB)": Trace(A * A) + Trace(A * B),
    "Case 4: Tr(A^2 B)": Trace(A * A * B),
    "Case 5: Tr(A) Tr(B)": Trace(A) * Trace(B),
    "Case 6: Tr(A^2) Tr(AB)": Trace(A * A) * Trace(A * B),
    "Case 7: Tr(BA)": Trace(B * A),
    "Case 8: Tr(ABA)": Trace(A * B * A),
    "Case 9: Tr(A)^2": Trace(A)**2,
    "Case 10: 3/2 Tr(AB) - 5 Tr(A^2)": (3/2) * Trace(A * B) - 5 * Trace(A * A),
    "Case 11: Tr(A + B)": Trace(A + B),
    "Case 12: Tr(λ AB)": Trace(lam * A * B),
}

all_results = {}
for label, expr in test_cases.items():
    intermediate = epsilon_expand_with_true_orders(expr, expand_symbols=["A", "B"], max_order=4)
    cleaned = {order: extract_scalars_from_traces(val, scalars=[lam, sp.Symbol("epsilon")])
               for order, val in intermediate.items()}
    all_results[label] = cleaned

flat_rows = []
for label, result in all_results.items():
    for order, expr in result.items():
        flat_rows.append((label, order, expr))


#------------------------------------------------------------------------
# Modular structure:

# Step 1: Function to expand a power series like (εA₁ + ε²A₂ + ...)^n using multinomial theorem
def expand_single_matrix_power(symbol, order_list, power, max_order):
    from itertools import product
    from math import factorial
    epsilon = sp.Symbol("epsilon")
    terms = defaultdict(lambda: 0)
    n = power
    k = len(order_list)
    for alpha in product(range(n + 1), repeat=k):
        if sum(alpha) != n:
            continue
        # epsilon power is weighted by index * alpha_i
        eps_power = sum((i + 1) * a for i, a in enumerate(alpha))
        if eps_power > max_order:
            continue
        # multinomial coefficient
        coeff = factorial(n)
        for a in alpha:
            coeff //= factorial(a)
        # construct matrix product term
        mat_factors = []
        for i, a in enumerate(alpha):
            mat_factors.extend([MatrixSymbol(f"{symbol}{i + 1}", 3, 3)] * a)
        term = coeff * Trace(sp.MatMul(*mat_factors, evaluate=False))
        terms[eps_power] += term
    return dict(terms)

# Step 2: Process a single Trace(A^n B^m ...) using known power expansions
def process_trace_product_structure(symbol_structure, symbol_orders, max_order):
    from itertools import product
    # Expand each block separately
    power_blocks = defaultdict(int)
    for sym in symbol_structure:
        power_blocks[sym] += 1
    expanded_blocks = {}
    for sym, pow in power_blocks.items():
        expanded_blocks[sym] = expand_single_matrix_power(sym, symbol_orders[sym], pow, max_order)
    # Multiply out all terms from each symbol's expansion
    result = defaultdict(lambda: 0)
    block_items = [list(v.items()) for v in expanded_blocks.values()]
    for combo in product(*block_items):
        total_order = sum(c[0] for c in combo)
        if total_order > max_order:
            continue
        product_term = sp.Mul(*[c[1] for c in combo], evaluate=False)
        result[total_order] += product_term
    return result

def expand_matrix_symbol(symbol)

def process_term(factor,**kwargs):
    if isinstance(factor,sp.MatrixSymbol):
        return expand_matrix_symbol(factor,**kwargs)
    elif isinstance(factor,sp.MatPow):
        base, exp = factor.args
        return expand_matrix_power(base,exp,**kwargs)
    elif isinstance(factor,sp.MatMul):
        # Recursively process each term in the product:
        all_exprs = [process_factor(arg,**kwargs) for arg in factor.args]
        expr = all_exprs[0]
        for k in range(1,len(all_exprs)):
            expr = product_expansion(expr,all_exprs[k],**kwargs)
        return expr
    elif isinstance(factor,sp.Add):
        # Recursively process each term in the sum:
        all_exprs = [process_factor(arg,**kwargs) for arg in factor.args]
        expr = all_exprs[0]
        for k in range(1,len(all_exprs)):
            expr = expr + all_exprs[k]
        return expr
    else:
        # If not sure what to do with it, just return it unchanged:
        return factor

def process_matrix_product(trace_expression):
    if not isinstance(trace_expression, sp.MatMul):
        raise Exception("Expression is not a matrix product.")
    # Get the individual products:
    product_list = trace_expression.args
    expansion_list = []
    for arg in product_list:
        if isinstance(arg,sp.MatrixSymbol):
            # Return an expression here for the expansion of the symbol:
        elif isinstance(arg,sp.MatPow):
            base, exp = arg.args

# Step 3: Recursively split Trace(A + B) to Trace(A) + Trace(B)
def split_linear_trace(expr):
    if isinstance(expr, Trace):
        arg = expr.args[0]
        if isinstance(arg, sp.Add):
            return sum(split_linear_trace(Trace(term)) for term in arg.args)
        return expr
    elif isinstance(expr, sp.Add):
        return sp.Add(*[split_linear_trace(arg) for arg in expr.args])
    else:
        return expr

def detect_structure_from_trace_argument(arg, symbol_orders):
    """
    Detect the matrix structure (like ['A', 'A', 'B']) from a Trace argument.
    Returns a list of symbol names like ['A', 'A', 'B'] or None if not matched.
    """
    structure = []
    def extract_symbol_name(msymbol):
        for sym in symbol_orders:
            if msymbol.name.startswith(sym):
                return sym
        return None
    if isinstance(arg, sp.MatMul):
        for term in arg.args:
            if isinstance(term, MatrixSymbol):
                sym = extract_symbol_name(term)
                if sym:
                    structure.append(sym)
    elif isinstance(arg, sp.MatPow):
        base, exp = arg.args
        if isinstance(base, MatrixSymbol):
            sym = extract_symbol_name(base)
            if sym:
                structure.extend([sym] * exp)
    elif isinstance(arg, MatrixSymbol):
        sym = extract_symbol_name(arg)
        if sym:
            structure.append(sym)
    return structure if structure else None

# Patch into expand_trace_argument to use the new architecture
def expand_trace_argument(expr, symbol_orders, max_order):
    if isinstance(expr, Trace):
        arg = expr.args[0]
        # split Trace(A + B) -> Trace(A) + Trace(B)
        linear = split_linear_trace(expr)
        if isinstance(linear, sp.Add):
            result = defaultdict(lambda: 0)
            for subtrace in linear.args:
                structure = detect_structure_from_trace_argument(subtrace.args[0], symbol_orders)
                if structure:
                    expanded = process_trace_product_structure(structure, symbol_orders, max_order)
                    for k, v in expanded.items():
                        result[k] += v
                else:
                    result[0] += subtrace
            return dict(result)
        else:
            structure = detect_structure_from_trace_argument(linear.args[0], symbol_orders)
            if structure:
                return process_trace_product_structure(structure, symbol_orders, max_order)
            return {0: linear}
    return {0: expr}


# Test expand_single_matrix_power for A^2
test1 = expand_single_matrix_power("A", [1, 2, 3], 2, max_order=5)

# Test process_trace_product_structure for Tr(A^2 B)
test2 = process_trace_product_structure(["A", "A", "B"], {"A": [1, 2, 3], "B": [1, 2, 3]}, max_order=5)

# Test split_linear_trace for Trace(A + B)
A = MatrixSymbol("A", 3, 3)
B = MatrixSymbol("B", 3, 3)
test3 = split_linear_trace(Trace(A + B))

# Convert dictionary outputs to rows for display
rows1 = [(f"A^2", f"ε^{k}", v) for k, v in test1.items()]
rows2 = [(f"A^2B", f"ε^{k}", v) for k, v in test2.items()]
rows3 = [("Tr(A + B)", "", test3)]



# Rerun test suite using final decomposed architecture
final_outputs_refactored = {}
for name, expr in tests.items():
    expanded = epsilon_expand(expr, expand_symbols=["A", "B", "C"], max_order=5)
    final_outputs_refactored[name] = {
        k: safe_simplify(v, scalars=[epsilon]) for k, v in expanded.items()
    }

# Display cleaned expansion
rows = []
for test_name, result in final_outputs_refactored.items():
    for order, val in result.items():
        rows.append((test_name, f"ε^{order}", val))



# Generate the two test cases for detect_structure_from_trace_argument
A1, A2, A3 = [MatrixSymbol(f"A{i}", 3, 3) for i in range(1, 4)]
A = MatrixSymbol("A", 3, 3)

epsilon = sp.Symbol("epsilon")

# Case 1: Trace of A^2
case1_expr = Trace(A**2)
case1_structure = detect_structure_from_trace_argument(case1_expr.args[0], {"A": [1, 2, 3]})

# Case 2: Trace((ε A1 + ε² A2 + ε³ A3)^2)
A_series = epsilon*A1 + epsilon**2*A2 + epsilon**3*A3
case2_expr = Trace(A_series**2)
case2_structure = detect_structure_from_trace_argument(case2_expr.args[0], {"A": [1, 2, 3]})

(case1_expr, case1_structure), (case2_expr, case2_structure)

#-------------------------------------------------------------------

# Starting from scratch:

def add_expressions(expr_list):
    return sp.Add(*expr_list)

def multiply_expressions(expr_list):
    return sp.Mul(*expr_list, evaluate=False)

def ismatrix_symbol(symbol):
    if isinstance(symbol,sp.MatrixSymbol):
        return True
    if isinstance(symbol,sp.MatMul):
        return True
    if isinstance(symbol,sp.MatPow):
        return True
    if isinstance(symbol,sp.Add):
        args = symbol.args
        for arg in args:
            if ismatrix_symbol(symbol):
                return True
        return False
    return False
            

def isscalar_symbol(symbol):
    return not ismatrix_symbol(symbol)

# Apply trace linarity:
def split_trace(trace_expr,extract_scalars=True):
    if not isinstance(trace_expr,sp.Trace):
        raise Exception("Not a trace term!")
    arg = trace_expr.args[0]
    if isinstance(arg,sp.Add):
        all_exprs = arg.args
        # Apply linearity, recursively processing the trace on each term:
        traced_exprs = [Trace(expr) for expr in all_exprs]
        if extract_scalars:
            traced_exprs = [extract_all_scalars(expr) for expr in traced_exprs]
        return add_expressions(traced_exprs)
    else:
        if extract_scalars:
            return extract_all_scalars(trace_expr)
        else:
            return trace_expr

def extract_scalars_from_traces(expr, scalars):
    """Pull scalar factors like ε or λ outside trace expressions."""
    if isinstance(expr, Trace):
        arg = expr.args[0]
        if isinstance(arg, sp.Mul):
            scalar_factors = []
            matrix_factors = []
            for factor in arg.args:
                if any(factor.has(s) for s in scalars):
                    scalar_factors.append(factor)
                else:
                    matrix_factors.append(factor)
            return sp.Mul(*scalar_factors) * Trace(sp.Mul(*matrix_factors, evaluate=False))
        return expr
    elif isinstance(expr, sp.Add):
        return sp.Add(*[extract_scalars_from_traces(arg, scalars) for arg in expr.args])
    elif isinstance(expr, sp.Mul):
        return sp.Mul(*[extract_scalars_from_traces(arg, scalars) for arg in expr.args], evaluate=False)
    return expr

def classify_product_factors(arg):
    if not isinstance(arg,sp.Mul):
        raise Exception("Not a product")
    numbers = []
    scalars = []
    matrices = []
    for factor in arg.args:
        if isinstance(factor,sp.Number):
            numbers.append(factor)
        elif isinstance(factor,sp.MatrixSymbol):
            matrices.append(factor)
        elif isinstance(factor,sp.Symbol) or isinstance(factor,sp.Trace):
            scalars.append(factor)
        elif isinstance(factor,sp.MatMul):
            nums, scalr, mats = classify_product_factors(factor)
            numbers += nums
            scalars += scals
            matrices += mats
        elif isinstance(factor,sp.MatPow):
            matrices.append(factor)
        elif isinstance(factor,sp.Pow):
            base, exp = factor.args
            if isinstance(base,sp.Number):
                numbers.append(factor)
            else:
                scalars.append(factor)
        else:
            matrices.append(factor)
    return numbers, scalars, matrices

def classify_power_factors(arg):
    if not (isinstance(arg,sp.Pow) or isinstance(arg,sp.MatPow)):
        raise Exception("Not a power")
    base, exp = arg.args
    numbers = []
    scalars = []
    matrices = []
    if isinstance(base,sp.Mul):
        nums, scals, mats = classify_product_factors(base)
        numbers = [x**exp for x in nums]
        scalars = [x**exp for x in scals]
        matrices = [sp.Mul(*mats,evaluate=False)**exp]
    elif isinstance(base,sp.MatrixSymbol):
        matrices.append(arg)
    elif isinstance(base,sp.Symbol) or isinstance(factor,sp.Trace):
        scalars.append(arg)
    elif isinstance(base,sp.Number):
        numbers.append(arg)
    else:
        matrices.append(matrices)
    return numbers, scalars, matrices

def is_power_type(expr):
    return (isinstance(expr,sp.Pow) or isinstance(expr,sp.MatPow))

def is_symbol_type(expr):
    return isinstance(expr,sp.Symbol) or isinstance(expr,sp.MatrixSymbol)

def get_factors(expr):
    is_power = is_power_type(expr)
    is_symbol = is_symbol_type(expr)
    is_number = isinstance(expr,sp.Number)
    is_product = isinstance(expr,sp.Mul)
    if not (is_product or is_power or is_symbol or is_number):
        raise Exception("Not a valid product")
    if isinstance(expr,sp.Mul):
        factors = []
        for arg in expr.args:
            if is_symbol_type(arg):
                # Only want raw symbols as products:
                factors.append(arg)
            else:
                # In case one of the terms in the product is itself a 
                # product, or a power:
                factors.extend(get_factors(arg))
    elif is_power:
        base, exp = expr.args
        factors = get_factors(base)*exp
    elif is_symbol or is_number:
        return [expr]
    return factors

def cyclical_shift(input_list,left=True):
    if left:
        input_list.append(input_list.pop(0))
        return input_list
    else:
        input_list.insert(0,input_list.pop(-1))
        return input_list

def cyclically_equivalent(trace1,trace2):
    """Return True if the given trace expressions are cyclically equivalent"""
    if (not isinstance(trace1,sp.Trace) or (not isinstance(trace2,sp.Trace))):
        raise Exception("Both arguments must be traces")
    if trace1 == trace2:
        # Trivial case
        return True
    arg1 = trace1.args[0]
    arg2 = trace2.args[0]
    factors1 = get_factors(arg1)
    factors2 = get_factors(arg2)
    n1 = len(factors1)
    n2 = len(factors2)
    if n1 != n2:
        return False
    for k in range(n1):
    # Check all cyclical shifts:
        if factors1 == factors2:
            return True
        factors2 = cyclical_shift(factors2)
    return False

def get_matching_trace(trace,trace_cache):
    if not isinstance(trace,sp.Trace):
        raise Exception("Not a trace")
    for tr in trace_cache:
        if cyclically_equivalent(trace,tr):
            return tr
    return None

def find_all_traces(expr,trace_cache):
    if isinstance(expr,sp.Trace):
        trace_cache.append(expr)
    for arg in expr.args:
        trace_cache = find_all_traces(arg,trace_cache)
    return trace_cache

def gather_equivalent_traces(expr,simplify=True):
    # Parse the expression and search for traces,
    # replacing them with traces already found if they are cyclically
    # equivalent
    # We should usually always make the traces as simple as possible first:
    if simplify:
        expr = simplify_trace_expression(expr)
    trace_cache = find_all_traces(expr,[])
    unique_traces = []
    trace_dictionary = {}
    for tr in trace_cache:
        match = get_matching_trace(tr,unique_traces)
        if match is None:
            unique_traces.append(tr)
            trace_dictionary[tr] = tr
        else:
            trace_dictionary[tr] = match
    # Now parse the expression and replace cyclically equivalent terms:
    for tr in trace_cache:
        expr = expr.subs(tr,trace_dictionary[tr])
    return expr


def simplify_trace_expression(expr):
    # Apply linearity and extract scalars:
    if isinstance(expr,sp.Add):
        return sp.Add(*[simplify_trace_expression(arg) for arg in expr.args])
    if isinstance(expr,sp.Mul):
        factors = [arg for arg in expr.args]
        simplified_factors = [simplify_trace_expression(arg) for arg in factors]
        return sp.Mul(*simplified_factors,evaluate=False)
    if isinstance(expr,sp.Pow) or isinstance(expr,sp.MatPow):
        base, exp = expr.args
        simplified_base = simplify_trace_expression(base)
        return simplified_base**exp
    if isinstance(expr,sp.Trace):
        arg = expr.args[0]
        # Fully expand the argument:
        expanded_arg = expand_matrix_expression(expr.args[0])
        factored_arg = [factor_out_all_scalars(arg) for arg in expanded_arg]
        traces = [Trace(arg) for arg in factored_arg]
        if len(expanded_arg) > 1:
            simplified = [simplify_trace_expression(tr) for tr in traces]
            return sp.Add(*[tr for tr in simplified])
        else:
            if isinstance(factored_arg[0],sp.Mul):
                numbers, scalars, matrices = classify_product_factors(
                    factored_arg[0]
                )
            elif (isinstance(factored_arg[0],sp.Pow) 
                  or isinstance(factored_arg[0],sp.MatPow)):
                numbers, scalars, matrices = classify_power_factors(
                    factored_arg[0]
                )
            else:
                numbers = []
                scalars = []
                matrices = [factored_arg[0]]
            numprod = sp.Mul(*numbers,evaluate=False)
            scalprod = sp.Mul(*scalars,evaluate=False)
            matprod = sp.Mul(*matrices,evaluate=False)
            return numprod*scalprod*Trace(matprod)
    else:
        return expr


def extract_all_scalars(trace_expr):
    if not isinstance(trace_expr,sp.Trace):
        raise Exception("Not a trace term!")
    arg = trace_expr.args[0]
    factored = factor_out_all_scalars(arg)
    all_scalars = get_scalar_variables(arg)
    powers = [get_highest_extractable_power(arg, var) for var in all_scalars]
    rescaled_arg = factored
    for p, var in zip(powers, all_scalars):
        rescaled_arg = matrix_divide_scalar(rescaled_arg,var**p,
                                            factorise=False)
    return sp.Mul(*[var**p for p, var in zip(powers, all_scalars)],
                  evaluate=False)*Trace(rescaled_arg)

def process_trace(trace_expr,symbol_dictionaries,extract_scalars=True):
    if not isinstance(trace_expr,sp.Trace):
        raise Exception("Not a trace term!")
    arg = trace_expr.args[0]
    processed_args = process_term(arg,symbol_dictionaries)
    if extract_scalars:
        processed_args = factor_out_all_scalars(processed_args)
    return split_trace(Trace(processed_args),extract_scalars=extract_scalars)

def expand_matrix_symbol(symbol,symbol_dictionaries):
    if not isinstance(symbol,sp.Expr):
        raise Exception("Not a sympy expression.")
    if isinstance(symbol,sp.Symbol) or isinstance(symbol,sp.MatrixSymbol):
        name = symbol.name
        epsilon = symbols('epsilon')
        if name in symbol_dictionaries:
            all_exprs = [
                            sym*(epsilon**pow) for sym, pow in 
                            zip(symbol_dictionaries[name]["matrices"],
                                symbol_dictionaries[name]["orders"])
            ]
            return add_expressions(all_exprs)
        else:
            # If no specified rule, safer to leave it as it is
            return symbol
    else:
        # Assume we want to substitute all terms in the dictionary:
        new_symbol = symbol
        for sym in symbol_dictionaries:
            symvar = symbol_dictionaries[sym]["symbol"]
            substitution = expand_matrix_symbol(symvar,symbol_dictionaries)
            new_symbol = new_symbol.subs(symvar,substitution)
        return new_symbol

def get_power(factor,variable):
    if isinstance(factor,sp.Symbol) or isinstance(factor,sp.MatrixSymbol):
        if factor == variable:
            return 1
        else:
            return 0
    if isinstance(factor,sp.Pow) or isinstance(factor,sp.MatPow):
        base, exp = factor.args
        base_power = get_power(base,variable)
        if base_power is not None:
            return base_power*exp
        else:
            return None
    if isinstance(factor,sp.Mul):
        subfactors = factor.args
        all_powers = [get_power(sub,variable) for sub in subfactors]
        if any(pow is None for pow in all_powers):
            return None
        return sum(all_powers)
    if isinstance(factor,sp.Add):
        subterms = factor.args
        all_powers = [get_power(sub,variable) for sub in subterms]
        first_power = all_powers[0] 
        if all(pow == first_power for pow in all_powers):
            return first_power
        else:
            return None
    if isinstance(factor,sp.Trace):
        if isinstance(variable,sp.Trace):
            if factor == variable:
                return 1
            else:
                return 0
        else:
            return 0
    else:
        return 0


def get_highest_extractable_power(expr,var):
    power = get_power(expr,var)
    if power is not None:
        return power
    else:
        if isinstance(expr,sp.Add):
            powers = [get_highest_extractable_power(arg,var) for arg in expr.args]
            return min(powers)
        elif isinstance(expr,sp.Mul):
            powers = [get_highest_extractable_power(arg,var) for arg in expr.args]
            return sum(powers)
        elif isinstance(expr,sp.Pow) or isinstance(expr,sp.MatPow):
            base, exp = expr.args
            base_power = get_highest_extractable_power(base,var)
            return base_power*exp
        else:
            # If all else fails, assume we just cant extract any powers:
            return 0

def get_scalar_variables(expr):
    scalar_vars = []
    if is_symbol_type(expr):
        if isscalar_symbol(expr):
            scalar_vars.append(expr)
        return scalar_vars
    for arg in expr.args:
        if (isinstance(arg,sp.Symbol) or isinstance(arg,sp.Trace)):
            if arg not in scalar_vars:
                scalar_vars.append(arg)
        elif isinstance(arg,sp.MatrixSymbol):
            continue
        else:
            new_vars = get_scalar_variables(arg)
            for var in new_vars:
                if var not in scalar_vars:
                    scalar_vars.append(var)
    return scalar_vars

def factor_out_all_scalars(expr):
    if not isinstance(expr,sp.Expr):
        raise Exception("Not a sympy expression.")
    all_scalars = get_scalar_variables(expr)
    new_expr = expr
    for scalar in all_scalars:
        new_expr = factor_out_variable(new_expr,scalar)
    return new_expr

def matrix_divide_scalar(expr,scalar,factorise=True):
    # Simplify division of a matrix expression by a scalar, 
    # cancelling anything which can be cancelled.
    if not isinstance(expr,sp.Expr):
        raise Exception("Not a sympy expression.")
    if isinstance(expr,sp.Add):
        all_divisions = [
            matrix_divide_scalar(arg,scalar) for arg in expr.args
        ]
        return sp.Add(*all_divisions)
    else:
        if factorise:
            return factor_out_all_scalars(expr)/scalar
        else:
            return expr/scalar

def factor_out_variable(expr,variable):
    if not (isinstance(variable,sp.Symbol) or isinstance(variable,sp.Trace)):
        raise Exception("Invalid variable for extraction.")
    # Returns the expression with all powers of variable
    # out front, multiplying the rest of the expression, if this is possible.
    if isinstance(expr,sp.Mul):
        factors = expr.args
        no_var = []
        powers = []
        for factor in factors:
            if factor.has(variable):
                factored = factor_out_variable(factor,variable)
                power = get_highest_extractable_power(factored,variable)
                if power != 0:
                    if isinstance(factored,sp.Add):
                        no_var.append(
                            matrix_divide_scalar(factored,variable**power)
                        )
                    else:
                        no_var.append(factored/variable**power)
                else:
                    no_var.append(factor)
                powers.append(power)
            else:
                no_var.append(factor)
        other = sp.Mul(*no_var,evaluate=False)
        total_pow = sum(powers)
        var_factor = variable**(total_pow)
        return sp.Mul(var_factor,other,evaluate=False)
    elif isinstance(expr,sp.Add):
        # Check that factoring is actually possible:
        power = get_highest_extractable_power(expr,variable)
        return sp.Add(*[factor_out_variable(arg,variable) for arg in expr.args])
    elif isinstance(expr,sp.Pow) or isinstance(expr,sp.MatPow):
        base, exp = expr.args
        factored_base = factor_out_variable(base,variable)
        power = get_highest_extractable_power(factored_base,variable)
        if power != 0:
            if isinstance(factored_base,sp.Add):
                new_base = matrix_divide_scalar(factored_base,variable**power)
            else:
                new_base = factored_base / variable**power
        else:
            new_base = base
        var_power = exp*power
        return sp.Mul(variable**var_power,new_base**exp,evaluate=False)
    elif isinstance(expr,sp.Trace):
        return extract_all_scalars(expr)
    else:
        # If unsure, do nothing.
        return expr


def get_expression_list(expr):
    """Convert an expression to a list of it's additive terms."""
    if isinstance(expr,list):
        return expr
    if isinstance(expr,sp.Add):
        return [arg for arg in expr.args]
    else:
        return [expr]

def expand_product_pair_to_list(expr1,expr2):
    """Manually expand a product of two expressions to a list of the terms 
    in the expanded expression"""
    expr1_list = get_expression_list(expr1)
    expr2_list = get_expression_list(expr2)
    return [sp.Mul(x,y,evaluate=False) for x in expr1_list for y in expr2_list]

def expand_products_to_list(product_expression):
    """Recursively expand a product of n variables"""
    if not isinstance(product_expression,sp.Mul):
        raise Exception("Argument is not a product.")
    factors = product_expression.args
    factor_lists = [expand_matrix_expression(factor) for factor in factors]
    term_list = expand_product_pair_to_list(factor_lists[0],factor_lists[1])
    for k in range(2,len(factor_lists)):
        term_list = expand_product_pair_to_list(term_list,factor_lists[k])
    return term_list
            

from math import factorial
def get_multinomial_term(all_terms,alpha):
    if len(all_terms) != len(alpha):
        raise Exception("Invalid indices")
    mat_factors = []
    for i, a in enumerate(alpha):
        mat_factors.extend([all_terms[i]] * a)
    # Coefficient:
    n = sum(alpha)
    coeff = factorial(n)
    for a in alpha:
        coeff //= factorial(a)
    mat = sp.Mul(*mat_factors,evaluate=False)
    return mat, coeff

def expand_power_multinomial(expr):
    base, exp = expr.args
    expanded_base = expand_matrix_expression(base)
    nterms = len(expanded_base)
    indices = product(range(exp+1),repeat=nterms)
    valid_indices = [x for x in indices if sum(x) == exp]
    factors_and_coeffs = [
        get_multinomial_term(expanded_base,alpha)
        for alpha in valid_indices
    ]
    expansion_terms = [term[1]*term[0] for term in factors_and_coeffs]
    return expansion_terms

def expand_power(expr):
    # Expand a matrix power, taking account of non-commutativity:
    base, exp = expr.args
    expanded_base = expand_matrix_expression(base)
    if exp == 0:
        return sp.ZeroMatrix(expanded_base[0].shape[0],
                             expanded_base[0].shape[1])
    elif exp == 1:
        return sp.Add(*expanded_base)
    elif exp > 1:
        expansion_terms = expand_product_pair_to_list(expanded_base,
                                                      expanded_base)
        for k in range(2,exp):
            expansion_terms = expand_product_pair_to_list(expansion_terms,
                                                          expanded_base)
    return expansion_terms


def expand_trace(expr):
    if not isinstance(expr,sp.Trace):
        raise Exception("Not a trace expression")
    arg = expr.args[0]
    expanded = expand_matrix_expression(arg)
    return [split_trace(Trace(term)) for term in expanded]

def has_traces(expr):
    return any(isinstance(arg,sp.Trace) for arg in expr.args)

# Recursively decompose an addition into all the terms that are added:
def expand_matrix_expression(expr):
    if isinstance(expr,sp.Add):
        expr_list = []
        for arg in expr.args:
            expr_list += expand_matrix_expression(arg)
    elif (isinstance(expr,sp.MatMul) 
            or (isinstance(expr,sp.Mul) and has_traces(expr))):
        expr_list = expand_products_to_list(expr)
    elif isinstance(expr,sp.Symbol) or isinstance(expr,sp.MatrixSymbol):
        expr_list = [expr]
    elif (isinstance(expr,sp.MatPow) 
            or (isinstance(expr,sp.Pow) and has_traces(expr))):
        expr_list = expand_power(expr)
    elif isinstance(expr,sp.Trace):
        expr_list = expand_trace(expr)
    elif isinstance(expr,sp.Mul) or isinstance(expr,sp.Pow):
        # Scalars only, so just do expand:
        expanded = sp.expand(expr)
        if isinstance(expanded,sp.Add):
            expr_list = [arg for arg in expanded.args]
        else:
            # No further expansions possible, so just add this term:
            expr_list = [expanded]
    else:
        expr_list = [expr]
    return expr_list

# Gather terms by their powers of some variable:
def gather_terms(expr, variable):
    term_list = expand_matrix_expression(expr)
    all_powers = [get_highest_extractable_power(arg,variable) for arg in term_list]
    pmin = min(all_powers)
    pmax = max(all_powers)
    power_list = range(pmin,pmax+1)
    expr_lists = [[] for p in power_list]
    for arg, power in zip(term_list,all_powers):
        ind = power - pmin
        expr_lists[ind].append(factor_out_variable(arg,variable)/variable**power)
    power_dictionary = defaultdict(int)
    for p, expr_list in zip(power_list,expr_lists):
        power_dictionary[p] = sp.Add(*expr_list)
    return power_dictionary

def process_term(term,symbol_dictionaries,extract_scalars=True,max_order=None):
    epsilon = sp.Symbol("epsilon")
    if isinstance(term,sp.Trace):
        new_term = process_trace(
            term,symbol_dictionaries,extract_scalars=extract_scalars
        )
    else:
        expanded = expand_matrix_symbol(term,symbol_dictionaries)
        if extract_scalars:
            new_term = factor_out_all_scalars(
                sp.Add(*expand_matrix_expression(expanded))
            )
        else:
            new_term = sp.Add(*expand_matrix_expression(expanded))
    if max_order is not None:
        order_range = range(0,max_order+1)
        all_orders = gather_terms(new_term,epsilon)
        new_dict = defaultdict(int)
        for p in order_range:
            if all_orders[p] != 0: 
                new_dict[p] = all_orders[p]
        new_term = sp.Add(*[epsilon**p*new_dict[p] for p in order_range])
    return new_term


def get_order_symbols(name,max_order,min_order = 1):
    start = min_order
    orders = range(start,max_order + 1)
    matrices = [MatrixSymbol(f"{name}{i}", 3, 3) for i in orders]
    return orders, matrices

max_order = 5
min_order = 1
expand_symbol_names = ["A","B"]
symbol_dictionaries = {name:{} for name in expand_symbol_names}
for name in expand_symbol_names:
    symbol_dictionaries[name]["min_order"] = min_order
    symbol_dictionaries[name]["symbol"] = MatrixSymbol(name,3,3)
    orders, matrices = get_order_symbols(name,max_order,symbol_dictionaries[name]["min_order"])
    symbol_dictionaries[name]["orders"] = orders
    symbol_dictionaries[name]["matrices"] = matrices


#-------------------------------------------------------------------------------
# Symbolic LPT

# Gradient of the Displacement field:
# Variables:
DPsi = MatrixSymbol("DPsi",3,3) # gradient of the full displacement field
DtDPsi = MatrixSymbol("DtDPsi",3,3)#  - Time derivative term
G = sp.Symbols("G") # Newton's constant
Gfactor = 4*sp.pi*G # Prefactor to RHS
epsilon = sp.Symbol("epsilon") # LPT Perturbation parameter (dummy variable)


max_order = 5
min_order = 1
expand_symbol_names = ["DPsi","DtDPsi"]
symbol_dictionaries = {name:{} for name in expand_symbol_names}
for name in expand_symbol_names:
    symbol_dictionaries[name]["min_order"] = min_order
    symbol_dictionaries[name]["symbol"] = MatrixSymbol(name,3,3)
    orders, matrices = get_order_symbols(name,max_order,symbol_dictionaries[name]["min_order"])
    symbol_dictionaries[name]["orders"] = orders
    symbol_dictionaries[name]["matrices"] = matrices

LDPsi = sp.Rational(1,2)*(Trace(DPsi)**2 - Trace(DPsi**2))
detDPsi = sp.Rational(1,6)*(Trace(DPsi)**3 - 3*Trace(DPsi)*Trace(DPsi**2)
                             + 2*Trace(DPsi**3))

# RHS of the Evolution equation:
RHS = Gfactor*(Trace(DPsi) + LDPsi + detDPsi)
RHS_expanded = gather_terms(
    simplify_trace_expression(
    process_term(RHS,symbol_dictionaries,max_order=max_order)
    ),epsilon
)

# LHS of the Evolution Equation:
I = sp.matrices.expressions.Identity(3)# Identity matrix
C = (DPsi**2 - Trace(DPsi)*DPsi + 
    sp.Rational(1,2)*(Trace(DPsi)**2 - Trace(DPsi**2))*I)
LHS = Trace((I + Trace(DPsi)*I - DPsi + C)*DtDPsi)
LHS_expanded = gather_terms(
    simplify_trace_expression(
    process_term(LHS,symbol_dictionaries,max_order=max_order)
    ),epsilon
)

# Solutions of each order:
DtDPsi1, DtDPsi2, DtDPsi3, DtDPsi4, DtDPsi5 = symbol_dictionaries["DtDPsi"]["matrices"]
DPsi1, DPsi2, DPsi3, DPsi4, DPsi5 = symbol_dictionaries["DPsi"]["matrices"]
lpt_1 = (LHS_expanded[1] - RHS_expanded[1])
lpt_2 = simplify_trace_expression(
    (LHS_expanded[2] - RHS_expanded[2]).subs(DtDPsi1,Gfactor*DPsi1)
)



