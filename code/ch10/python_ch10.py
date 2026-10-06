# =====================================================================
#  Κεφάλαιο 10 — Η εξίσωση θερμότητας: FFT, συνάρτηση σφάλματος, σειρές
#  Συνοδευτικός κώδικας του βιβλίου «Μετασχηματισμοί Laplace και Fourier,
#  Μερικές Διαφορικές Εξισώσεις», Ν. Ματζάκος.
#  Τρέχει όπως είναι σε Python 3 (sympy, numpy, scipy, matplotlib)
#  ή απευθείας στη σελίδα του βιβλίου (Pyodide).
# =====================================================================
import numpy as np
import sympy as sp
from scipy.special import erf, erfc
import matplotlib.pyplot as plt

# ---------- Α. Επαλήθευση λύσεων με τη SymPy ----------
x, t, k, T0, a = sp.symbols('x t k T_0 a', positive=True)
for name, u in [("θερμικός πυρήνας", sp.exp(-x**2/(4*k*t))/sp.sqrt(4*sp.pi*k*t)),
                ("Παράδειγμα 10.10", T0*sp.erfc(x/(2*sp.sqrt(k*t)))),
                ("Παράδειγμα 10.4", (sp.erf((x + a)/(2*sp.sqrt(k*t))) - sp.erf((x - a)/(2*sp.sqrt(k*t))))/2)]:
    print(f"Α. {name:18s}: u_t - k u_xx =", sp.simplify(u.diff(t) - k*u.diff(x, 2)))

# ---------- Β. Η μέθοδος του κεφαλαίου με FFT (Παράδειγμα 10.4, a = k = 1) ----------
# μετασχηματίζουμε, πολλαπλασιάζουμε με e^{-kω²t}, αντιστρέφουμε
N, Lbox, kk = 1024, 40.0, 1.0
xs = -Lbox/2 + Lbox*np.arange(N)/N               # πλέγμα στο [-20, 20)
dx = xs[1] - xs[0]
# ο παλμός ως μέση τιμή σε κάθε κελί: τα άκρα ±1 δεν είναι κόμβοι του πλέγματος
f = np.clip((np.minimum(xs + dx/2, 1) - np.maximum(xs - dx/2, -1))/dx, 0, 1)
w = 2*np.pi*np.fft.fftfreq(N, d=dx)
exact = lambda X, T: 0.5*(erf((X + 1)/(2*np.sqrt(kk*T))) - erf((X - 1)/(2*np.sqrt(kk*T))))
heat = lambda T: np.fft.ifft(np.fft.fft(f)*np.exp(-kk*w**2*T)).real
print("\nΒ. FFT έναντι του τύπου της erf")
plt.figure(figsize=(7, 3.4))
plt.plot(xs, f, "k--", lw=1, label="t = 0")
for T in (0.05, 0.25, 1.0, 100.0):
    u = heat(T)
    print(f"   t = {T:6.2f}:  u(0,t) = {u[N//2]:.5f}  (τύπος {exact(0.0, T):.5f}),"
          f"  μέγιστο σφάλμα {np.abs(u - exact(xs, T)).max():.1e}")
    if T <= 1: plt.plot(xs, u, label=f"t = {T}")
plt.xlim(-5, 5); plt.xlabel("x"); plt.legend(fontsize=8); plt.grid(alpha=.3)
plt.title("Διάχυση του ορθογώνιου παλμού με FFT"); plt.show()
print("   Για t = 100 η FFT «βλέπει» περιοδική ράβδο μήκους 40: τα είδωλα του παλμού")
print("   ανεβάζουν τη θερμοκρασία προς τη μέση τιμή 2/40 = 0,05.")

# ---------- Γ. Ημιάπειρη ράβδος (Παράδειγμα 10.10): η κλίμακα √(kt) ----------
print("\nΓ. Βάθος όπου u = T0/2, δηλαδή erfc(η) = 1/2 με η ≈ 0,4769:")
for T in (1, 4, 16):
    print(f"   t = {T:2}: x = 2·0,4769·√(kt) = {2*0.4769*np.sqrt(kk*T):.3f}")
print("   τετραπλάσιος χρόνος ⇒ διπλάσιο βάθος")

# ---------- Δ. Ράβδος πεπερασμένου μήκους (Παράδειγμα 10.13): πόσοι όροι; ----------
def mid_temp(T, nmax):
    n = np.arange(1, nmax + 1, 2, dtype=float)  # μόνο περιττά n (float: στο Pyodide οι ακέραιοι είναι 32 bit)
    return 400/np.pi*np.sum(np.exp(-2*n**2*T)/n*np.sin(n*np.pi/2))
print("\nΔ. Θερμοκρασία στο μέσο της ράβδου")
for T in (0.001, 0.01, 0.1, 0.467):
    ref = mid_temp(T, 200001)
    nmax = next(m for m in range(1, 200001, 2) if abs(mid_temp(T, m) - ref) < 1)
    terms = "αρκεί ο όρος n = 1" if nmax == 1 else f"αρκούν οι όροι n = 1, 3, …, {nmax} ({(nmax + 1)//2} όροι)"
    print(f"   t = {T:5}: u(π/2,t) = {ref:7.3f};  για σφάλμα < 1 βαθμό {terms}")
print("   Οι παράγοντες e^{-2n²t} σβήνουν γρήγορα τους υψηλούς όρους μόνο όταν το t δεν είναι πολύ μικρό.")

# ---------- Ε. Ο πυρήνας του Duhamel: πότε φτάνει ο θερμικός παλμός (Παράδειγμα 10.12) ----------
kc, xd = 5e-7, 0.1                              # δομικό υλικό, βάθος 10 cm
Ts = np.linspace(1, 20000, 200001)
K = xd/(2*np.sqrt(np.pi*kc)*Ts**1.5)*np.exp(-xd**2/(4*kc*Ts))
print(f"\nΕ. μέγιστο του K στο t = {Ts[np.argmax(K)]:.0f} s  (τύπος x²/(6k) = {xd**2/(6*kc):.0f} s)")

# ---------- ΣΤ. Πρόταση 10.4: πηγή θερμότητας (Παράδειγμα 10.5) ----------
x_, t_, s_ = sp.symbols('x t s', real=True)
ua = sp.integrate(sp.exp(-(t_ - s_))*sp.cos(x_), (s_, 0, t_))              # Q = cos x, k = 1
print("Q = cos x:  u =", sp.simplify(ua), "  έλεγχος:", sp.simplify(ua.diff(t_) - ua.diff(x_, 2) - sp.cos(x_)))
tp = sp.symbols('t', positive=True)
u0 = sp.integrate(1/sp.sqrt(4*sp.pi*(tp - s_)), (s_, 0, tp))               # Q = δ(x): θερμοκρασία στην πηγή
print("Q = δ(x):  u(0,t) =", sp.simplify(u0))

# ---------- Ζ. Τετραγωνική πλάκα (Παράδειγμα 10.16): ισοθερμικές καμπύλες ----------
X, Y = np.meshgrid(np.linspace(0, 1, 201), np.linspace(0, 1, 201))   # a = 1, T0 = 100
U = np.zeros_like(X)
for m in np.arange(1, 400, 2, dtype=float):                         # περιττά n
    ratio = np.exp(m*np.pi*(Y - 1))*(1 - np.exp(-2*m*np.pi*Y))/(1 - np.exp(-2*m*np.pi))
    U += 400/np.pi*ratio*np.sin(m*np.pi*X)/m
print(f"\nΖ. Θερμοκρασία στο κέντρο: {U[100, 100]:.4f}  (ακριβής τιμή T0/4 = 25)")
plt.figure(); cs = plt.contour(X, Y, U, levels=[5, 10, 25, 50, 75]); plt.clabel(cs, fmt='%d')
plt.gca().set_aspect('equal'); plt.title("Ισοθερμικές καμπύλες της πλάκας"); plt.show()
with np.errstate(over='ignore', invalid='ignore'):
    naive = np.sinh(301*np.pi*1.0)/np.sinh(301*np.pi)                # στο y = 1 θα έπρεπε να είναι 1
print("   απλός τύπος sinh(nπy)/sinh(nπ) για n = 301, y = 1:", naive, "(υπερχείλιση: inf/inf)")
