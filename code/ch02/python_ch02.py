# =====================================================================
#  Κεφάλαιο 2 — Βηματικές, δέλτα και αριθμητικός έλεγχος
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
H, D = sp.Heaviside, sp.DiracDelta

def L(f):
    """Μετασχηματισμός Laplace (χωρίς τις συνθήκες σύγκλισης)."""
    return sp.laplace_transform(f, t, s, noconds=True)

# ---------- Α. Παράδειγμα 2.7: μετασχηματισμοί χωρίς ολοκληρώματα ----------
print("Α. Παράδειγμα 2.7")
signals = [("(α) κλιμακωτή τάση     ", 2 + 3*H(t-1) - 2*H(t-3) - 3*H(t-4)),
           ("(β) ράμπα με κορεσμό   ", t - (t-2)*H(t-2)),
           ("(γ) σήμα του ελεγκτή   ", 2*t - 2*(t-1)*H(t-1) - 2*H(t-3))]
for name, f in signals:
    print(" ", name, "F(s) =", sp.expand(L(f)))

# ---------- Β. Παράδειγμα 2.11: ιδιότητα επιλογής ----------
print("\nΒ. Παράδειγμα 2.11")
print("  (α) ∫(t³+1)δ(t-2)dt        =", sp.integrate((t**3 + 1)*D(t - 2), (t, 0, sp.oo)))
print("  (β) ∫e^(-3t) sin t δ(t-π/2)dt =", sp.integrate(sp.exp(-3*t)*sp.sin(t)*D(t - sp.pi/2), (t, 0, sp.oo)))
print("  (γ) L{e^(-t) δ(t-1)}       =", L(sp.exp(-t)*D(t - 1)))

# ---------- Γ. Αντιστροφή με καθυστέρηση: Λυμένη Άσκηση 2.3 ----------
F = 4*(sp.exp(-s) - sp.exp(-3*s))/(s*(s + 2))
f = sp.inverse_laplace_transform(F, s, t)
print("\nΓ. Λυμένη Άσκηση 2.3: f(t) =", f)
for tv in (0.5, 2, 4):
    print(f"   f({tv}) = {float(f.subs(t, tv)):.6f}")

# ---------- Δ. Η «παγίδα» της δέλτα στο t = 0 ----------
print("\nΔ. L{δ(t)} με t θετικό :", L(D(t)))          # 0: το t = 0 θεωρείται εκτός πεδίου
tr = sp.symbols('t')                                  # χωρίς υπόθεση για το πρόσημο
print("   L{δ(t)} με t πραγματικό:", sp.laplace_transform(D(tr), tr, s, noconds=True))
print("   Παράδειγμα 2.11(δ):", sp.laplace_transform(3*D(tr) - 2*D(tr - 4) + sp.cos(sp.pi*tr)*D(tr - 1), tr, s, noconds=True))

# ---------- Ε. Περιοδικά σήματα του Σχήματος 2.5 με NumPy ----------
x = np.linspace(0, 8, 4001)
square = np.where(np.mod(x, 2) < 1, 1.0, -1.0)            # τετραγωνικό ±1, a = 1
pwm = 1.0*(np.mod(x, 2) < 0.5)                            # PWM: V = 1, T = 2, D = 1/4
saw = np.mod(x, 2)                                        # πριονωτό, T = 2
half = np.maximum(np.sin(np.pi*x/2), 0)                   # ημιανορθωμένο ημίτονο
pulse = np.heaviside(x - 1, 1) - np.heaviside(x - 3, 1)   # ένας παλμός με βηματικές
fig, ax = plt.subplots(2, 2, figsize=(8, 5))
for a_, y_, ttl in [(ax[0, 0], square, "τετραγωνικό ±1"), (ax[0, 1], pwm, "παλμοσειρά PWM"),
                    (ax[1, 0], saw, "πριονωτό"), (ax[1, 1], half, "ημιανορθωμένο ημίτονο")]:
    a_.plot(x, y_); a_.set_title(ttl); a_.grid(alpha=.3)
ax[0, 1].plot(x, pulse, "--", label="u(t-1) - u(t-3)"); ax[0, 1].legend(loc="upper right")
plt.tight_layout()
plt.show()

# ---------- ΣΤ. Αριθμητικός έλεγχος του πριονωτού κύματος (Παράδειγμα 2.16) ----------
T, sv = 2.0, 1.3
val, err = quad(lambda u: np.exp(-sv*u)*np.mod(u, T), 0, 40, points=np.arange(2, 40, 2), limit=200)
exact = 1/sv**2 - T*np.exp(-T*sv)/(sv*(1 - np.exp(-T*sv)))
print(f"\nΣΤ. ∫_0^40 e^(-st) f(t) dt ≈ {val:.10f}")
print(f"    τύπος του βιβλίου     = {exact:.10f}   (διαφορά {abs(val - exact):.1e})")
