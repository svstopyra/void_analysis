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
    """Pull scalar factors (like λ or ε) outside of traces where possible."""
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



