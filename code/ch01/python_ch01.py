# =====================================================================
#  Κεφάλαιο 1 — Ο Μετασχηματισμός Laplace με SymPy και αριθμητικός έλεγχος
#  Συνοδευτικός κώδικας του βιβλίου «Μετασχηματισμοί Laplace και Fourier,
#  Μερικές Διαφορικές Εξισώσεις», Ν. Ματζάκος.
#  Τρέχει όπως είναι σε Python 3 (sympy, numpy, scipy, matplotlib)
#  ή απευθείας στη σελίδα του βιβλίου (Pyodide).
# =====================================================================
import sympy as sp
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

t, s = sp.symbols('t s', positive=True)

def L(f):
    """Μετασχηματισμός Laplace (χωρίς τις συνθήκες σύγκλισης), απλοποιημένος."""
    return sp.simplify(sp.laplace_transform(f, t, s, noconds=True))

# ---------- Α. Λυμένες Ασκήσεις 1.3–1.5 ----------
print("Α. Μετασχηματισμοί")
print("  L{4t²e^(-t) + 3cosh2t - 5e^(2t)sin3t} =", L(4*t**2*sp.exp(-t) + 3*sp.cosh(2*t) - 5*sp.exp(2*t)*sp.sin(3*t)))
print("  L{t e^(-2t) sin t}                  =", sp.factor(L(t*sp.exp(-2*t)*sp.sin(t))))
print("  L{sin²t / t}                        =", L(sp.sin(t)**2/t))
print("  L{√t}                               =", L(sp.sqrt(t)))

# ---------- Β. Αριθμητικός έλεγχος της Εφαρμογής 1.3(α) ----------
print("\nΒ. ∫_0^∞ e^(-2t) t cos t dt")
val, err = quad(lambda u: np.exp(-2*u)*u*np.cos(u), 0, np.inf)
print(f"  αριθμητικά ≈ {val:.12f},  ακριβής τιμή 3/25 = {3/25:.12f}")

# ---------- Γ. Πότε «σβήνει» η ουρά της e^(-st) e^(0.5t); (Παράδειγμα 1.3) ----------
tt = np.linspace(0, 12, 600)
plt.figure(figsize=(7, 3.6))
for sv in (0.6, 1.0, 2.0):
    plt.plot(tt, np.exp(-(sv - 0.5)*tt), label=f"s = {sv}")
plt.xlabel("t"); plt.ylabel("e^(-st) e^(0.5t)")
plt.title("Η ολοκληρωτέα του ορισμού για f(t) = e^(0.5t)")
plt.legend(); plt.grid(alpha=.3)
plt.show()
print("\nΓ. Για s > 0.5 η ολοκληρωτέα φθίνει εκθετικά· όσο πιο κοντά στο 0.5, τόσο πιο αργά.")

# ---------- Δ. (Προαιρετικό) θεώρημα τελικής τιμής για το Παράδειγμα 1.19 ----------
# y'' + 3y' + 2y = 0, y(0) = 1, y'(0) = 0  ->  Y(s) = (s+3)/((s+1)(s+2))
Y = (s + 3)/((s + 1)*(s + 2))
y = sp.inverse_laplace_transform(Y, s, t)
print("\nΔ. y(t) =", sp.simplify(y))
print("   lim_{s→0} sY(s) =", sp.limit(s*Y, s, 0), "  και  y(20) ≈", float(y.subs(t, 20)))
