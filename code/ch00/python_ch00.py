# =====================================================================
#  Κεφάλαιο 0 — Μαθηματικό Υπόβαθρο: επαλήθευση με SymPy και SciPy
#  Συνοδευτικός κώδικας του βιβλίου «Μετασχηματισμοί Laplace και Fourier,
#  Μερικές Διαφορικές Εξισώσεις», Ν. Ματζάκος.
#  Τρέχει όπως είναι σε Python 3 (sympy, numpy, scipy, matplotlib)
#  ή απευθείας στη σελίδα του βιβλίου (Pyodide).
# =====================================================================
import sympy as sp
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

s, t, x = sp.symbols('s t x')

# ---------- Α. Ανάλυση σε απλά κλάσματα (Ενότητα 0.2) ----------
print("Α. Απλά κλάσματα")
print("  32/(s^4-16)            =", sp.apart(32/(s**4 - 16), s))
print("  (s+7)/((s-1)(s^2+2s+5)) =", sp.apart((s + 7)/((s - 1)*(s**2 + 2*s + 5)), s))
# Ο παράγοντας s^2+2s+5 είναι ανάγωγος (ρίζες -1±2i): η apart τον αφήνει ανέπαφο.
print("  1/(s^2+2s+5)           =", sp.apart(1/(s**2 + 2*s + 5), s))

# ---------- Β. Π.Α.Τ. της Λυμένης Άσκησης 0.5 ----------
print("\nΒ. y'' + 2y' + 5y = 10cos t, y(0) = y'(0) = 0")
y = sp.Function('y')
ode = sp.Eq(y(t).diff(t, 2) + 2*y(t).diff(t) + 5*y(t), 10*sp.cos(t))
sol = sp.dsolve(ode, ics={y(0): 0, y(t).diff(t).subs(t, 0): 0})
print("  y(t) =", sp.simplify(sol.rhs))

# ---------- Γ. Ολοκληρώματα ----------
print("\nΓ. Ολοκληρώματα")
n = sp.symbols('n', integer=True, positive=True)
In = sp.integrate((x**2 + x)*sp.cos(n*x), (x, -sp.pi, sp.pi))
print("  I_n = ∫(x^2+x)cos(nx)dx στο [-π,π] =", sp.simplify(In))      # Παράδειγμα 0.10(β)
val, err = quad(lambda u: np.exp(-u**2), -np.inf, np.inf)              # ολοκλήρωμα Gauss
print(f"  ∫e^(-x^2)dx ≈ {val:.12f}   (√π = {np.sqrt(np.pi):.12f}, σφάλμα ≈ {err:.1e})")

# ---------- Δ. Απόσβεση: χρόνος αποκατάστασης 1% (Σχήμα 0.3) ----------
# y'' + 2ζy' + y = 0, y(0) = 1, y'(0) = 0  (ω0 = 1)
print("\nΔ. Χρόνος μετά τον οποίο |y(t)| < 1% οριστικά")
tt = np.linspace(0, 20, 200001)
def response(z, tt):
    if z < 1:
        wd = np.sqrt(1 - z**2)
        return np.exp(-z*tt)*(np.cos(wd*tt) + z/wd*np.sin(wd*tt))
    if z == 1:
        return (1 + tt)*np.exp(-tt)
    r1, r2 = -z + np.sqrt(z**2 - 1), -z - np.sqrt(z**2 - 1)
    return (r2*np.exp(r1*tt) - r1*np.exp(r2*tt))/(r2 - r1)

plt.figure(figsize=(7, 3.6))
for z in (0.7, 0.85, 1.0):
    yy = response(z, tt)
    ts = tt[np.nonzero(np.abs(yy) >= 0.01)[0][-1]]
    print(f"  ζ = {z:<4}: t_1% ≈ {ts:.2f}")
    plt.plot(tt, yy, label=f"ζ = {z}")
plt.axhspan(-0.01, 0.01, color='0.85')
plt.xlim(0, 12); plt.xlabel("t"); plt.ylabel("y(t)"); plt.legend(); plt.grid(alpha=.3)
plt.title("Αποκρίσεις και ζώνη ±1%")
plt.show()
