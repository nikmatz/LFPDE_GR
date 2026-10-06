# =====================================================================
#  Κεφάλαιο 3 — Απλά κλάσματα, αντιστροφή και συνέλιξη με SymPy
#  Συνοδευτικός κώδικας του βιβλίου «Μετασχηματισμοί Laplace και Fourier,
#  Μερικές Διαφορικές Εξισώσεις», Ν. Ματζάκος.
#  Τρέχει όπως είναι σε Python 3 (sympy, numpy, matplotlib)
#  ή απευθείας στη σελίδα του βιβλίου (Pyodide).
# =====================================================================
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

t, s, tau = sp.symbols('t s tau', positive=True)

def iL(F):
    """Αντίστροφος μετασχηματισμός Laplace (για t > 0), όρο προς όρο μετά την ανάλυση
    σε απλά κλάσματα. Προσοχή: η SymPy 1.12, που τρέχει στον browser (Pyodide), δίνει
    λάθος αποτέλεσμα για ορισμένες ρητές συναρτήσεις με πολλαπλούς πόλους, αν δεν
    προηγηθεί η apart (π.χ. για την F του Παραδείγματος 3.8). Έτσι είμαστε ασφαλείς."""
    terms = sp.Add.make_args(sp.apart(sp.together(F), s))
    return sp.expand(sp.simplify(sum(sp.inverse_laplace_transform(q, s, t) for q in terms)))

# ---------- Α. Απλά κλάσματα: Παραδείγματα 3.7–3.9 ----------
print("Α. Απλά κλάσματα")
examples = {"3.7": (4*s**2 + 11*s + 9)/(s**3 + 4*s**2 + s - 6),
            "3.8": (6*s**2 + 7*s - 11)/((s - 2)*(s + 1)**3),
            "3.9": (7*s + 29)/((s + 2)*(s**2 + 6*s + 13))}
for k, F in examples.items():
    print(f"  Παράδειγμα {k}:  F(s) =", sp.apart(F, s))
    print(f"                 f(t) =", iL(F))

# Μιγαδικά απλά κλάσματα για το Παράδειγμα 3.9
F = examples["3.9"]
print("\n  Μιγαδική ανάλυση (full=True):")
for term in sp.Add.make_args(sp.apart(F, s, full=True).doit()):
    n, d = term.as_numer_denom()
    z = sp.symbols('z')                                # (το s έχει δηλωθεί θετικό)
    p = sp.solve(d.subs(s, z), z)[0]                   # ο πόλος του όρου
    print(f"     ({sp.expand(n/sp.Poly(d, s).LC())}) / (s - ({p}))")
s0 = -3 + 2*sp.I
print("  Συντελεστής K του πόλου -3+2i (κάλυψη):", sp.simplify(sp.limit((s - s0)*F, s, s0)))

# ---------- Β. Λυμένες Ασκήσεις 3.1 και 3.2 ----------
print("\nΒ. Λυμένες Ασκήσεις")
for k, G in {"3.1": (3*s**2 - 3*s + 3)/(s**3 - 3*s**2 + 4),
             "3.2": (25 - 3*s**2)/(s**2*(s**2 + 2*s + 5))}.items():
    print(f"  Λ.Α. {k}: παρονομαστής {sp.factor(sp.denom(G))},  f(t) = {iL(G)}")
tr = sp.symbols('t')                                   # χωρίς υπόθεση προσήμου:
print("  Με t πραγματικό, L⁻¹{1/(s+1)} =", sp.inverse_laplace_transform(1/(s + 1), s, tr))

# Η ln((s+3)/(s+1)) με την Πρόταση 3.1: f(t) = -(1/t) L⁻¹{F'(s)}
F3 = sp.log((s + 3)/(s + 1))
print("  L⁻¹{ln((s+3)/(s+1))} =", sp.simplify(-sp.inverse_laplace_transform(sp.diff(F3, s), s, t)/t))

# ---------- Γ. Συνέλιξη (Παραδείγματα 3.13 και 3.16) ----------
def conv(f, g):
    """(f*g)(t) = ∫_0^t f(τ) g(t-τ) dτ"""
    return sp.simplify(sp.integrate(f.subs(t, tau)*g.subs(t, t - tau), (tau, 0, t)))

a, b = sp.symbols('a b', positive=True)
print("\nΓ. Συνελίξεις")
print("  1*1 =", conv(sp.Integer(1), sp.Integer(1)), ",   1*t =", conv(sp.Integer(1), t))
print("  e^(at)*e^(bt) =", conv(sp.exp(a*t), sp.exp(b*t)))
print("  e^(at)*e^(at) =", conv(sp.exp(a*t), sp.exp(a*t)))
print("  sin t * sin t =", conv(sp.sin(t), sp.sin(t)), "  και L⁻¹{1/(s²+1)²} =", iL(1/(s**2 + 1)**2))

# Παράδειγμα 3.16: ολοκληρωτικές εξισώσεις Volterra
Y = sp.symbols('Y')
for k, eq in {"(α) y = 1 + ∫y      ": sp.Eq(Y, 1/s + Y/s),
              "(β) y = t + sin * y ": sp.Eq(Y, 1/s**2 + Y/(s**2 + 1))}.items():
    Ys = sp.solve(eq, Y)[0]
    print("  ", k, " Y(s) =", sp.factor(Ys), ",  y(t) =", iL(Ys))

# ---------- Δ. Οι πόλοι «διαβάζονται» στο γράφημα ----------
x = np.linspace(0, 6, 600)
f37 = sp.lambdify(t, iL(examples["3.7"]), "numpy")
f39 = sp.lambdify(t, iL(examples["3.9"]), "numpy")
plt.figure(figsize=(7, 3.6))
plt.plot(x, f37(x), label="Παρ. 3.7: πόλοι 1, -2, -3 (κυριαρχεί ο e^t)")
plt.plot(x, f39(x), label="Παρ. 3.9: πόλοι -2, -3±2i (φθίνει)")
plt.ylim(-1, 8); plt.xlabel("t"); plt.grid(alpha=.3); plt.legend()
plt.show()
