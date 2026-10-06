# =====================================================================
#  Κεφάλαιο 9 — Η κυματική εξίσωση: d'Alembert, σειρές Fourier και
#  πεπερασμένες διαφορές
#  Συνοδευτικός κώδικας του βιβλίου «Μετασχηματισμοί Laplace και Fourier,
#  Μερικές Διαφορικές Εξισώσεις», Ν. Ματζάκος.
#  Τρέχει όπως είναι σε Python 3 (sympy, numpy, matplotlib)
#  ή απευθείας στη σελίδα του βιβλίου (Pyodide).
# =====================================================================
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# ---------- Α. Ο τύπος του d'Alembert με τη SymPy (Παράδειγμα 9.6) ----------
x, t, s = sp.symbols('x t s', real=True)
c = 3
f = lambda z: z**2
g = lambda z: sp.cos(z)
u = (f(x - c*t) + f(x + c*t))/2 + sp.integrate(g(s), (s, x - c*t, x + c*t))/(2*c)
print("Α. u(x,t) =", sp.simplify(sp.expand_trig(u)))
print("   u_tt - 9 u_xx =", sp.simplify(u.diff(t, 2) - c**2*u.diff(x, 2)),
      "  u(x,0) =", u.subs(t, 0), "  u_t(x,0) =", sp.simplify(u.diff(t).subs(t, 0)))

# ---------- Β. Η τραβηγμένη χορδή (Παράδειγμα 9.13), L = h = c = 1 ----------
L = h = cc = 1.0
def f0(xx):                       # τραβηγμένη στο x = L/3 σε ύψος h
    return np.where(xx < L/3, 3*h*xx/L, 3*h*(L - xx)/(2*L))
def series(xx, tt, N=200):
    n = np.arange(1, N + 1)[:, None]
    A = 9*h/(np.pi**2*n**2)*np.sin(np.pi*n/3)
    return np.sum(A*np.sin(n*np.pi*xx/L)*np.cos(n*np.pi*cc*tt/L), axis=0)
print("\nΒ. A_n για n = 1..6:", np.round([9/(np.pi**2*k**2)*np.sin(np.pi*k/3) for k in range(1, 7)], 4),
      " (η 3η και η 6η λείπουν)")

# ---------- Γ. Πεπερασμένες διαφορές και συνθήκη CFL ----------
def fd(r, J=90, T_end=1.0):
    """Σχήμα u_j^{m+1} = 2u_j^m - u_j^{m-1} + r²(u_{j+1}^m - 2u_j^m + u_{j-1}^m)."""
    xs = np.linspace(0, L, J + 1); dx = L/J; dt = r*dx/cc
    M = int(round(T_end/dt))                  # u1 = u^1, οπότε χρειάζονται M - 1 βήματα
    u0 = f0(xs); u1 = u0.copy()
    u1[1:-1] = u0[1:-1] + r**2/2*(u0[2:] - 2*u0[1:-1] + u0[:-2])       # g = 0
    E = []
    for m in range(M - 1):
        u2 = np.zeros_like(u1)
        u2[1:-1] = 2*u1[1:-1] - u0[1:-1] + r**2*(u1[2:] - 2*u1[1:-1] + u1[:-2])
        # διακριτή ενέργεια με ρ = T = 1 (Ορισμός 9.4)
        E.append(dx/2*(np.sum(((u2 - u1)/dt)**2) + np.sum((np.diff(u1)/dx)**2)))
        u0, u1 = u1, u2
        if np.abs(u1).max() > 1e6: break
    return xs, u1, (m + 2)*dt, np.array(E)

print("\nΓ. Τη στιγμή t = L/c η ακριβής λύση είναι -f(L-x) (Παράδειγμα 9.13)· ακριβής ενέργεια E = 2,25")
for r in (0.9, 1.0, 1.05):
    xs, uh, tt, E = fd(r)
    if abs(tt - 1) < 1e-9:
        print(f"   r = {r:4}:  σφάλμα από την ακριβή {np.abs(uh + f0(L - xs)).max():.1e},"
              f"  από τη σειρά (2000 όροι) {np.abs(uh - series(xs, tt, 2000)).max():.1e},"
              f"  E μεταξύ {E.min():.3f} και {E.max():.3f}")
    else:
        print(f"   r = {r:4}:  «έκρηξη» τη στιγμή t = {tt:.2f}: max|u| = {np.abs(uh).max():.1e}, E = {E[-1]:.1e}")

# ---------- Δ. Στιγμιότυπα (Σχήμα 9.8) ----------
xx = np.linspace(0, L, 601)
plt.figure(figsize=(7, 3.4))
for tt, sty in zip((0, 1/3, 2/3, 1), ("-", "--", "-.", ":")):
    plt.plot(xx, series(xx, tt), sty, label=f"t = {tt:.2f}")
plt.axhline(0, c="k", lw=.5); plt.xlabel("x"); plt.legend(fontsize=8); plt.grid(alpha=.3)
plt.title("Τραβηγμένη χορδή: άθροισμα 200 όρων"); plt.show()

# ---------- Ε. Ενέργεια ανά κανονικό τρόπο (Πρόταση 9.6, Λυμένη Άσκηση 9.5) ----------
n = np.arange(1, 4001)
En = (np.sin(np.pi*n/3)/n)**2                 # ∝ n² A_n² για την τραβηγμένη χορδή
print("\nΕ. ποσοστό ενέργειας στις 3 πρώτες αρμονικές:", np.round(100*En[:3]/En.sum(), 1), "%")

# ---------- ΣΤ. Θεώρημα 9.3: η μη ομογενής κυματική εξίσωση (Παράδειγμα 9.10) ----------
x_, t_, s_, tau_ = sp.symbols('x t s tau', real=True)
def duhamel(F, c=1):
    """(1/2c) ∫_0^t ∫_{x-c(t-τ)}^{x+c(t-τ)} F(s,τ) ds dτ, με F γραμμένη στις μεταβλητές s, tau."""
    inner = sp.integrate(F, (s_, x_ - c*(t_ - tau_), x_ + c*(t_ - tau_)))
    return sp.simplify(sp.integrate(inner, (tau_, 0, t_))/(2*c))

for F in (s_, sp.sin(s_)*sp.sin(tau_)):
    u = duhamel(F)
    print("F =", F, " ->  u =", u, "  έλεγχος:", sp.simplify(u.diff(t_, 2) - u.diff(x_, 2) - F.subs({s_: x_, tau_: t_})))
