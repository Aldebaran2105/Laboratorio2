import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# ==== Datos experimentales (sin do = 45 cm) ====
di = np.array([30.4, 29, 28, 27, 26.2, 25.4, 25], dtype=float)   # cm
hi = np.array([1.1, 1.1, 1.0, 0.9, 0.8, 0.7, 0.5], dtype=float)  # cm
ho = 2.4  # cm (altura del objeto)

# ==== Magnificación CON SIGNO (imagen real invertida) ====
M = -hi / ho  # signo negativo para reflejar inversión

# ==== AJUSTE LINEAL M = a * di + b ====
a, b = np.polyfit(di, M, 1)
M_fit = a * di + b
R2 = r2_score(M, M_fit)

# ==== DISTANCIA FOCAL Y RADIO DE CURVATURA ====
f_exp = -1 / a
R_exp = 2 * f_exp

# ==== RESULTADOS NUMÉRICOS ====
print("=== AJUSTE LINEAL: M = a * di + b ===")
print(f"a (pendiente) = {a:.6f}")
print(f"b (intercepto) = {b:.6f}")
print(f"R² = {R2:.4f}")
print(f"Distancia focal experimental f = {f_exp:.3f} cm")
print(f"Radio de curvatura experimental R = {R_exp:.3f} cm")

# ==== GRÁFICO ====
plt.figure(figsize=(10,6))
plt.scatter(di, M, color='royalblue', s=100, edgecolor='black', label='Datos experimentales')
plt.plot(di, M_fit, color='green', linewidth=2, label=f"Ajuste lineal: M = {a:.4f}·di + {b:.4f}")

# Línea teórica (modelo con f=20 cm)
di_range = np.linspace(min(di)-1, max(di)+1, 100)
M_teo = -di_range / 20 + 1
plt.plot(di_range, M_teo, 'orange', linestyle='--', linewidth=2, label='Modelo teórico (f=20 cm)')

# Etiquetas y estilo
plt.xlabel('Distancia imagen $d_i$ (cm)', fontsize=12)
plt.ylabel('Magnificación $M = -h_i/h_0$', fontsize=12)
plt.title('Ajuste lineal de $M$ vs $d_i$ (sin do = 45 cm, con signo negativo)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()

# Mostrar resultados dentro del gráfico
texto = (f"a = {a:.4f}\n"
         f"b = {b:.4f}\n"
         f"R² = {R2:.4f}\n"
         f"f_exp = {f_exp:.2f} cm\n"
         f"R_exp = {R_exp:.2f} cm")
plt.text(0.02, 0.02, texto, transform=plt.gca().transAxes, fontsize=10,
         bbox=dict(facecolor='white', alpha=0.8))

plt.show()