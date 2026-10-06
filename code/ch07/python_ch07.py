# =====================================================================
#  Κεφάλαιο 7 — Ο μετασχηματισμός Fourier με SymPy, NumPy και SciPy
#  Συνοδευτικός κώδικας του βιβλίου «Μετασχηματισμοί Laplace και Fourier,
#  Μερικές Διαφορικές Εξισώσεις», Ν. Ματζάκος.
#  Τρέχει όπως είναι σε Python 3 (sympy, numpy, scipy, matplotlib)
#  ή απευθείας στη σελίδα του βιβλίου (Pyodide).
# =====================================================================
import sympy as sp
import numpy as np
from scipy.integrate import quad
from scipy.special import sici
import matplotlib.pyplot as plt

t, w = sp.symbols('t omega', real=True)
k = sp.symbols('k', real=True)
a = sp.symbols('a', positive=True)

# Η SymPy υπολογίζει ∫ f(t) e^{-2πikt} dt (σύμβαση Hz)· για τη σύμβαση του βιβλίου k = ω/2π.
FT = lambda f: sp.simplify(sp.fourier_transform(f, t, w/(2*sp.pi)))

# ---------- Α. Πίνακας 7.1 ----------
print("Α. u(t)e^{-at}  ->", FT(sp.Heaviside(t)*sp.exp(-a*t)))
print("   e^{-a|t|}    ->", FT(sp.exp(-a*sp.Abs(t))))
print("   e^{-at²}     ->", FT(sp.exp(-a*t**2)))

# ---------- Β. Μια παγίδα: απάντηση που ισχύει μόνο για ω > 0 ----------
print("\nΒ. F{1/(1+t²)}    ->", FT(1/(1 + t**2)), "   (σωστό: π e^{-|ω|})")
print("   F^{-1}{2/(1+ω²)} ->", sp.simplify(sp.inverse_fourier_transform(2/(1 + 4*sp.pi**2*k**2), k, t)),
      "   (σωστό: e^{-|t|})")
# η Πρόταση 7.4 το προδίδει: πραγματική και άρτια συνάρτηση έχει πραγματικό και ΑΡΤΙΟ μετασχηματισμό
print("   αριθμητικά F(-1) =", round(2*quad(lambda s: 1/(1 + s*s), 0, np.inf, weight='cos', wvar=1.0)[0], 6),
      " = π e^{-1} =", round(np.pi*np.exp(-1), 6))

# ---------- Γ. Φάσμα με FFT (Ενότητα 6.6) ----------
N, dt = 4096, 0.01
tt = (np.arange(N) - N//2)*dt
wf = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(N, d=dt))
F1 = dt*np.fft.fftshift(np.fft.fft(np.fft.ifftshift(np.exp(-np.abs(tt)))))
p1 = np.where(np.abs(tt) < 1, 1.0, 0.0); p1[np.abs(np.abs(tt) - 1) < 1e-9] = 0.5
F2 = dt*np.fft.fftshift(np.fft.fft(np.fft.ifftshift(p1)))
band = np.abs(wf) < 30
print("\nΓ. FFT e^{-|t|}: μέγιστο σφάλμα", f"{np.max(np.abs(F1 - 2/(1 + wf**2))):.1e}")
print("   FFT p_1:      μέγιστο σφάλμα στο |ω|<30", f"{np.max(np.abs(F2 - 2*np.sinc(wf/np.pi))[band]):.1e}")

# ---------- Δ. Σχήμα 7.2: μερικά ολοκληρώματα Fourier του παλμού ----------
xs = np.linspace(-2.5, 2.5, 1001)
plt.figure(figsize=(7, 3.4))
plt.plot(xs, np.where(np.abs(xs) < 1, 1.0, 0.0), "k--", lw=1, label="f(x)")
for Om in (4, 16, 64):
    fO = (sici(Om*(1 + xs))[0] + sici(Om*(1 - xs))[0])/np.pi
    plt.plot(xs, fO, label=f"Ω = {Om}")
plt.axhline(0.5 + sici(np.pi)[0]/np.pi, ls=":", c="r", lw=.8)
plt.xlabel("x"); plt.grid(alpha=.3); plt.legend(fontsize=8); plt.show()
print("\nΔ. όριο της υπερύψωσης 1/2 + Si(π)/π =", round(0.5 + sici(np.pi)[0]/np.pi, 4))

# ---------- Ε. Παράδειγμα 7.3: διπολικός παλμός ----------
bip = lambda s: 1.0 if 0 < s < 1 else (-1.0 if -1 < s < 0 else 0.0)
for wv in (0.5, 2.0, 2*np.pi):
    re = quad(lambda s: bip(s)*np.cos(wv*s), -1, 1, points=[0])[0]
    im = -quad(lambda s: bip(s)*np.sin(wv*s), -1, 1, points=[0])[0]
    print(f"Ε. ω = {wv:.3f}: F = {re:+.6f}{im:+.6f}i,   -4i sin²(ω/2)/ω = {-4*np.sin(wv/2)**2/wv:+.6f}i")
x = sp.symbols('x', positive=True)
print("   ∫_0^∞ sin³x/x dx =", sp.integrate(sp.sin(x)**3/x, (x, 0, sp.oo)))

# ---------- ΣΤ. Παράδειγμα 7.14: e^{-|t|} * e^{-|t|} = (1+|t|) e^{-|t|} ----------
for tv in (0.0, 0.8, -2.0):
    h = quad(lambda s: np.exp(-abs(s))*np.exp(-abs(tv - s)), -40, 40, points=[0, tv], limit=200)[0]
    print(f"ΣΤ. t = {tv:+.1f}:  (f*f)(t) = {h:.6f},  (1+|t|)e^(-|t|) = {(1 + abs(tv))*np.exp(-abs(tv)):.6f}")
print("    ∫ dω/(1+ω²)² =", sp.integrate(1/(1 + w**2)**2, (w, -sp.oo, sp.oo)))

# ---------- Ζ. Θεώρημα 7.9 και Εφαρμογή 7.4: δειγματοληψία και ανακατασκευή ----------
# Το μήνυμα m(t) = sin²t/(πt²) έχει φάσμα q_2(ω), άρα W = 2 και ρυθμός Nyquist ω_s = 4.
# Η sinc του βιβλίου, sin x / x, γράφεται με την κανονικοποιημένη numpy.sinc: sinc(ω_s(t-kT_s)/2) = np.sinc((t-k T_s)/T_s).
def m(t):
    t = np.asarray(t, dtype=float)
    out = np.full(t.shape, 1/np.pi)
    nz = np.abs(t) > 1e-12
    out[nz] = np.sin(t[nz])**2/(np.pi*t[nz]**2)
    return out

tt = np.linspace(-6, 6, 241)
k = np.arange(-4000, 4001, dtype=float)
for Ts in (1.0, 2*np.pi/3):                  # ω_s = 2π > 4 (χωρίς αναδίπλωση) και ω_s = 3 < 4 (αναδίπλωση)
    rec = np.array([np.sum(m(k*Ts)*np.sinc((t0 - k*Ts)/Ts)) for t0 in tt])
    print("T_s = %.4f: μέγιστο σφάλμα ανακατασκευής = %.2e" % (Ts, np.max(np.abs(rec - m(tt)))))
