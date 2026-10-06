# =====================================================================
#  Κεφάλαιο 8 — ΜΔΕ πρώτης τάξης, ταξινόμηση και κανονικές μορφές με SymPy
#  Συνοδευτικός κώδικας του βιβλίου «Μετασχηματισμοί Laplace και Fourier,
#  Μερικές Διαφορικές Εξισώσεις», Ν. Ματζάκος.
#  Τρέχει όπως είναι σε Python 3 (sympy, numpy, matplotlib)
#  ή απευθείας στη σελίδα του βιβλίου (Pyodide).
# =====================================================================
import sympy as sp
from sympy.solvers.pde import pdsolve, checkpdesol, classify_pde
import numpy as np
import matplotlib.pyplot as plt

x, y = sp.symbols('x y', real=True)
u = sp.Function('u')(x, y)

# ---------- Α. ΜΔΕ πρώτης τάξης με την pdsolve ----------
print("Α. Εξισώσεις πρώτης τάξης")
for eq in [sp.Eq(3*u.diff(x) + u.diff(y), 0),            # σταθεροί συντελεστές
           sp.Eq(u.diff(x) - 3*u.diff(y), 2*x),           # με όρο πηγής
           sp.Eq(u.diff(x) + 4*x*u.diff(y), 0),           # μεταβλητοί συντελεστές
           sp.Eq(x*u.diff(x) + y*u.diff(y), 2*u)]:        # ομογενείς συναρτήσεις
    sol = pdsolve(eq)
    print("  ", eq.lhs, "=", eq.rhs, "  ->  ", sol.rhs, "   έλεγχος:", checkpdesol(eq, sol)[0])

# η SymPy δίνει άλλη (ισοδύναμη) μορφή: η διαφορά από το x² + F(y+3x) είναι συνάρτηση του y+3x
s = sp.symbols('s')
diff_ = sp.expand(sp.Rational(19, 100)*x**2 - sp.Rational(27, 50)*x*y - sp.Rational(9, 100)*y**2 - x**2)
print("   διαφορά:", diff_, " =", sp.factor(diff_.subs(y, s - 3*x)), " με s = y + 3x")

# ---------- Β. Ταξινόμηση: Δ = b² - ac (προσοχή: ο συντελεστής του u_xy είναι 2b) ----------
def typos(a, b2, c):
    D = sp.Rational(b2, 2)**2 - a*c
    return D, ("υπερβολική" if D > 0 else "παραβολική" if D == 0 else "ελλειπτική")

print("\nΒ. Ταξινόμηση")
for coeffs in [(2, 7, 3), (4, -4, 1), (1, 4, 13), (1, 3, 3), (2, -5, 3)]:
    print("  ", coeffs, "->", typos(*coeffs))

# ---------- Γ. Οι νέοι συντελεστές μετά την αλλαγή μεταβλητών ξ(x,y), η(x,y) ----------
def transform(a, b, c, d, e, xi, eta):
    D = lambda f, *v: sp.diff(f, *v)
    A = a*D(xi, x)**2 + 2*b*D(xi, x)*D(xi, y) + c*D(xi, y)**2
    B = a*D(xi, x)*D(eta, x) + b*(D(xi, x)*D(eta, y) + D(xi, y)*D(eta, x)) + c*D(xi, y)*D(eta, y)
    C = a*D(eta, x)**2 + 2*b*D(eta, x)*D(eta, y) + c*D(eta, y)**2
    Dn = a*D(xi, x, 2) + 2*b*D(xi, x, y) + c*D(xi, y, 2) + d*D(xi, x) + e*D(xi, y)
    En = a*D(eta, x, 2) + 2*b*D(eta, x, y) + c*D(eta, y, 2) + d*D(eta, x) + e*D(eta, y)
    return [sp.simplify(t) for t in (A, B, C, Dn, En)]

print("\nΓ. Κανονικές μορφές [Ā, B̄, C̄, D̄, Ē]")
print("   u_xx+u_xy-2u_yy+3u_x+6u_y, ξ=y-2x, η=y+x  :", transform(1, sp.Rational(1, 2), -2, 3, 6, y - 2*x, y + x))
print("   4u_xx-4u_xy+u_yy-2u_y,      ξ=x+2y, η=x    :", transform(4, -2, 1, 0, -2, x + 2*y, x))
print("   u_xx+4u_xy+13u_yy,          ξ=y-2x, η=3x   :", transform(1, 2, 13, 0, 0, y - 2*x, 3*x))
print("   2u_xx-5u_xy+3u_yy+u_x+u_y,  ξ=x+y,  η=3x+2y:", transform(2, sp.Rational(-5, 2), 3, 1, 1, x + y, 3*x + 2*y))
print("   x²u_xx-y²u_yy,              ξ=xy,   η=y/x  :", transform(x**2, 0, -y**2, 0, 0, x*y, y/x))

# ---------- Δ. Επαλήθευση της γενικής λύσης με άγνωστες F, G ----------
F, G = sp.Function('F'), sp.Function('G')
w = F(y - 3*x) + G(2*y - x)
print("\nΔ. 2u_xx+7u_xy+3u_yy για u = F(y-3x)+G(2y-x):",
      sp.simplify(2*w.diff(x, 2) + 7*w.diff(x, y) + 3*w.diff(y, 2)))

# ---------- Ε. Οι χαρακτηριστικές μεταφέρουν την πληροφορία ----------
# u_x + 4x u_y = 0, u(0,y) = 1/(1+y²): η λύση είναι σταθερή πάνω στις παραβολές y - 2x² = C
X, Y = np.meshgrid(np.linspace(-1.5, 1.5, 301), np.linspace(-2, 6, 301))
U = 1/(1 + (Y - 2*X**2)**2)
plt.figure(figsize=(6, 4))
plt.contourf(X, Y, U, 20, cmap="viridis")
for C in (-2, 0, 2, 4):
    xs = np.linspace(-1.5, 1.5, 200); plt.plot(xs, 2*xs**2 + C, "w--", lw=.8)
plt.colorbar(label="u"); plt.xlabel("x"); plt.ylabel("y")
plt.title("u = 1/(1+(y-2x²)²) και χαρακτηριστικές y = 2x² + C"); plt.show()
