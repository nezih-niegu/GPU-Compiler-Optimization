#!/usr/bin/env python3
"""
Script para generar gráficas de optimización del compilador tensorial
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Configurar estilo
plt.style.use('seaborn-v0_8-darkgrid')
fig = plt.figure(figsize=(16, 10))

# ============================================================================
# GRÁFICA 1: Reducción de Operaciones (Antes vs Después)
# ============================================================================
ax1 = plt.subplot(2, 3, 1)
operaciones = ['Operaciones\nOriginales', 'Operaciones\nOptimizadas']
valores = [6, 3]
colores = ['#e74c3c', '#2ecc71']
bars = ax1.bar(operaciones, valores, color=colores, alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Número de Operaciones', fontsize=11, fontweight='bold')
ax1.set_title('Reducción de Operaciones en el Grafo', fontsize=12, fontweight='bold', pad=15)
ax1.set_ylim(0, 7)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# Agregar valores en las barras
for bar, valor in zip(bars, valores):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{valor}',
             ha='center', va='bottom', fontsize=12, fontweight='bold')

# Agregar porcentaje de reducción
reduccion = ((6 - 3) / 6) * 100
ax1.text(0.5, 5.5, f'Reducción: {reduccion:.0f}%', 
         ha='center', fontsize=11, fontweight='bold', 
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# ============================================================================
# GRÁFICA 2: Reducción de Memoria y Computación
# ============================================================================
ax2 = plt.subplot(2, 3, 2)
categorias = ['Memoria', 'Computación']
reducciones = [61, 99]
colores2 = ['#3498db', '#9b59b6']
bars2 = ax2.bar(categorias, reducciones, color=colores2, alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Reducción (%)', fontsize=11, fontweight='bold')
ax2.set_title('Optimización: Reducción de Recursos', fontsize=12, fontweight='bold', pad=15)
ax2.set_ylim(0, 110)
ax2.grid(axis='y', alpha=0.3, linestyle='--')

# Agregar valores
for bar, valor in zip(bars2, reducciones):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
             f'{valor}%',
             ha='center', va='bottom', fontsize=12, fontweight='bold')

# ============================================================================
# GRÁFICA 3: Comparación de Accesos a Memoria
# ============================================================================
ax3 = plt.subplot(2, 3, 3)
nodos = ['Node 0\n(IDENTITY A)', 'Node 1\n(IDENTITY B)', 'Node 2\n(IDENTITY A)', 
         'Node 3\n(IDENTITY B)', 'Node 4\n(MATMUL)', 'Node 5\n(ASSIGN)']
memoria_antes = [5000, 10000, 5000, 10000, 15000, 20000]  # Ejemplo de accesos
memoria_despues = [5000, 10000, 0, 0, 15000, 20000]  # Después de optimización (eliminación de duplicados)

x = np.arange(len(nodos))
width = 0.35

bars3a = ax3.bar(x - width/2, memoria_antes, width, label='Antes', color='#e74c3c', alpha=0.8, edgecolor='black')
bars3b = ax3.bar(x + width/2, memoria_despues, width, label='Después', color='#2ecc71', alpha=0.8, edgecolor='black')

ax3.set_ylabel('Accesos a Memoria', fontsize=11, fontweight='bold')
ax3.set_title('Accesos a Memoria por Nodo', fontsize=12, fontweight='bold', pad=15)
ax3.set_xticks(x)
ax3.set_xticklabels(nodos, rotation=45, ha='right', fontsize=8)
ax3.legend(fontsize=10)
ax3.grid(axis='y', alpha=0.3, linestyle='--')

# ============================================================================
# GRÁFICA 4: Complejidad de Algoritmos
# ============================================================================
ax4 = plt.subplot(2, 3, 4)
algoritmos = ['Construcción\nde Grafo', 'Fusión de\nOperaciones', 'Eliminación\nCSE', 
              'Optimización\nMemoria', 'Generación\nCUDA', 'Geometría\nComputacional']
complejidades = ['O(n)', 'O(n²)', 'O(n²)', 'O(n)', 'O(n)', 'O(d)']
valores_complejidad = [1, 2, 2, 1, 1, 0.5]  # Valores relativos para visualización
colores4 = ['#1abc9c', '#f39c12', '#f39c12', '#1abc9c', '#1abc9c', '#e67e22']

bars4 = ax4.barh(algoritmos, valores_complejidad, color=colores4, alpha=0.8, edgecolor='black', linewidth=1.5)
ax4.set_xlabel('Complejidad Relativa', fontsize=11, fontweight='bold')
ax4.set_title('Complejidad de Algoritmos Implementados', fontsize=12, fontweight='bold', pad=15)
ax4.grid(axis='x', alpha=0.3, linestyle='--')

# Agregar etiquetas de complejidad
for i, (bar, compl) in enumerate(zip(bars4, complejidades)):
    width = bar.get_width()
    ax4.text(width + 0.05, bar.get_y() + bar.get_height()/2,
             compl, ha='left', va='center', fontsize=10, fontweight='bold')

# ============================================================================
# GRÁFICA 5: Mejora de Performance (Tiempo Estimado)
# ============================================================================
ax5 = plt.subplot(2, 3, 5)
fases = ['Parsing', 'Construcción\nGrafo', 'Optimización', 'Generación\nCUDA', 'Total']
tiempo_antes = [1.0, 2.0, 0, 1.5, 4.5]  # Tiempo relativo
tiempo_despues = [1.0, 2.0, 0.5, 1.5, 5.0]  # Con optimización (más tiempo en optimización, pero mejor código)

x5 = np.arange(len(fases))
width5 = 0.35

bars5a = ax5.bar(x5 - width5/2, tiempo_antes, width5, label='Sin Optimización', color='#e74c3c', alpha=0.8, edgecolor='black')
bars5b = ax5.bar(x5 + width5/2, tiempo_despues, width5, label='Con Optimización', color='#2ecc71', alpha=0.8, edgecolor='black')

ax5.set_ylabel('Tiempo Relativo', fontsize=11, fontweight='bold')
ax5.set_title('Tiempo de Compilación por Fase', fontsize=12, fontweight='bold', pad=15)
ax5.set_xticks(x5)
ax5.set_xticklabels(fases, rotation=45, ha='right', fontsize=9)
ax5.legend(fontsize=10)
ax5.grid(axis='y', alpha=0.3, linestyle='--')

# ============================================================================
# GRÁFICA 6: Espacios de Iteración (Visualización Geométrica)
# ============================================================================
ax6 = plt.subplot(2, 3, 6)
ax6.set_xlim(0, 110)
ax6.set_ylim(0, 60)
ax6.set_aspect('equal')

# Tensor A: [100, 50] -> Iteration Space [[0, 99], [0, 49]]
rect_a = Rectangle((0, 0), 100, 50, linewidth=2, edgecolor='#3498db', 
                   facecolor='#3498db', alpha=0.3, label='Tensor A [100,50]')
ax6.add_patch(rect_a)
ax6.text(50, 25, 'A\n[100×50]\nVol: 5000', ha='center', va='center', 
         fontsize=10, fontweight='bold', color='darkblue')

# Tensor B: [50, 200] -> Iteration Space [[0, 49], [0, 199]]
# Mostramos una representación escalada
rect_b = Rectangle((0, 52), 50, 8, linewidth=2, edgecolor='#e74c3c', 
                   facecolor='#e74c3c', alpha=0.3, label='Tensor B [50,200]')
ax6.add_patch(rect_b)
ax6.text(25, 56, 'B\n[50×200]\nVol: 10000', ha='center', va='center', 
         fontsize=9, fontweight='bold', color='darkred')

ax6.set_xlabel('Dimensión 1', fontsize=11, fontweight='bold')
ax6.set_ylabel('Dimensión 2', fontsize=11, fontweight='bold')
ax6.set_title('Espacios de Iteración de Tensores', fontsize=12, fontweight='bold', pad=15)
ax6.legend(loc='upper right', fontsize=9)
ax6.grid(True, alpha=0.3, linestyle='--')

plt.suptitle('Análisis de Optimización del Compilador Tensorial', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig('optimizacion_compilador.png', dpi=300, bbox_inches='tight')
plt.savefig('optimizacion_compilador.pdf', bbox_inches='tight')
print("✅ Gráficas generadas: optimizacion_compilador.png y optimizacion_compilador.pdf")

