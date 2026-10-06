# =====================================================================
#  Κεφάλαιο 4 — Π.Α.Τ. με SymPy και αποκρίσεις με SciPy
#  Συνοδευτικός κώδικας του βιβλίου «Μετασχηματισμοί Laplace και Fourier,
#  Μερικές Διαφορικές Εξισώσεις», Ν. Ματζάκος.
#  Τρέχει όπως είναι σε Python 3 (sympy, numpy, scipy, matplotlib)
#  ή απευθείας στη σελίδα του βιβλίου (Pyodide).
# =====================================================================
import sympy as sp
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

t, s = sp.symbols('t s', positive=True)
y = sp.Function('y')

# ---------- Α. Παράδειγμα 4.2: y'' + 4y' + 3y = 8e^t, y(0) = 1, y'(0) = -1 ----------
ode = sp.Eq(y(t).diff(t, 2) + 4*y(t).diff(t) + 3*y(t), 8*sp.exp(t))
sol = sp.dsolve(ode, y(t), ics={y(0): 1, y(t).diff(t).subs(t, 0): -1})
print("Α. dsolve:          ", sol.rhs)
# Βήμα–βήμα: (s²Y - s + 1) + 4(sY - 1) + 3Y = 8/(s-1)
Y = sp.symbols('Y')
Ys = sp.solve(sp.Eq((s**2*Y - s + 1) + 4*(s*Y - 1) + 3*Y, 8/(s - 1)), Y)[0]
print("   Y(s) =", sp.factor(Ys), " =", sp.apart(Ys, s))
# (όρο προς όρο μετά την apart: ασφαλές και με την παλαιότερη SymPy του browser)
print("   y(t) =", sp.expand(sum(sp.inverse_laplace_transform(q, s, t) for q in sp.Add.make_args(sp.apart(Ys, s)))))

# ---------- Β. Παράδειγμα 4.8: ορθογώνιος παλμός ----------
f = 8*(sp.Heaviside(t) - sp.Heaviside(t - sp.pi/2))
sol = sp.dsolve(sp.Eq(y(t).diff(t, 2) + 4*y(t), f), y(t), ics={y(0): 0, y(t).diff(t).subs(t, 0): 0})
print("\nΒ. dsolve:", sp.simplify(sol.rhs))
book = sp.Piecewise((4*sp.sin(t)**2, t < sp.pi/2), (-4*sp.cos(2*t), True))   # η λύση του κειμένου
err = max(abs(float((sol.rhs - book).subs(t, float(tv)))) for tv in np.linspace(0.05, 9, 60))
print("   μέγιστη διαφορά από τη λύση του βιβλίου:", f"{err:.1e}")

# ---------- Γ. Παράδειγμα 4.12: H(s) = 1/(s² + 2s + 5) ----------
sys = signal.lti([1], [1, 2, 5])
tt = np.linspace(0, 8, 2001)
_, ys = signal.step(sys, T=tt)
_, h = signal.impulse(sys, T=tt)
print(f"\nΓ. max βηματικής ≈ {ys.max():.6f},  (1 + e^(-π/2))/5 = {(1 + np.exp(-np.pi/2))/5:.6f}")
print(f"   χρόνος κορυφής ≈ {tt[ys.argmax()]:.4f},  π/2 = {np.pi/2:.4f}")
plt.figure(figsize=(7, 3.6))
plt.plot(tt, ys, label="βηματική απόκριση"); plt.plot(tt, h, label="κρουστική απόκριση")
plt.axhline(0.2, ls="--", c="gray", lw=1); plt.xlabel("t"); plt.grid(alpha=.3); plt.legend()
plt.show()

# ---------- Δ. Παράδειγμα 4.4 αριθμητικά: y'' + 2y' + 5y = 20 cos t ----------
_, ynum, _ = signal.lsim(sys, U=20*np.cos(tt), T=tt)
yex = np.exp(-tt)*(-3*np.sin(2*tt) - 4*np.cos(2*tt)) + 2*np.sin(tt) + 4*np.cos(tt)
print(f"\nΔ. lsim: μέγιστο σφάλμα ως προς την ακριβή λύση = {np.abs(ynum - yex).max():.1e}")

# ---------- Ε. Σχήμα 4.4 και υπερύψωση (Πρόταση 4.4) ----------
plt.figure(figsize=(7, 3.6))
print("\nΕ.   ζ    Mp αριθμητικά   e^(-ζπ/√(1-ζ²))")
for z in (0.2, 0.5, 1.0, 2.0):
    tz, yz = signal.step(signal.lti([1], [1, 2*z, 1]), T=np.linspace(0, 14, 3001))
    plt.plot(tz, yz, label=f"ζ = {z}")
    if z < 1:
        print(f"    {z:.1f}   {yz.max() - 1:.6f}        {np.exp(-z*np.pi/np.sqrt(1 - z*z)):.6f}")
plt.axhline(1, ls="--", c="gray", lw=1); plt.xlabel("ω_n t"); plt.grid(alpha=.3); plt.legend()
plt.show()
