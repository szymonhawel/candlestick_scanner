"""
Generator wykresów skalowania wydajności dla testów PERF-01 do PERF-04.
Uruchomienie:
python -m pytest tests/test_candlestick_scanner.py::TestPerformance -v
python tests/generate_performance_report.py
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from collections import defaultdict

plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'Arial'

# ============================================================================
# WCZYTAJ DANE
# ============================================================================

if os.path.basename(os.getcwd()) == 'tests':
    json_path = 'results/performance_results.json'
    output_dir = 'results/'
else:
    json_path = 'tests/results/performance_results.json'
    output_dir = 'tests/results/'

if not os.path.exists(json_path):
    print(f"❌ BŁĄD: Nie znaleziono pliku {json_path}")
    print("   Uruchom najpierw: pytest tests/test_candlestick_scanner.py::TestPerformance -v")
    sys.exit(1)

with open(json_path, 'r', encoding='utf-8') as f:
    results = json.load(f)

print(f"✓ Wczytano {len(results)} wyników z {json_path}")

# ============================================================================
# PRZYGOTUJ DANE - Grupuj po nazwach testów
# ============================================================================

grouped = defaultdict(lambda: {'x': [], 'y': [], 'extra': []})

for r in results:
    test_name = r.get('test')
    num_candles = r.get('num_candles')
    
    if test_name and num_candles:
        grouped[test_name]['x'].append(num_candles)
        
        if 'czas_ms' in r:
            grouped[test_name]['y'].append(r['czas_ms'])
            grouped[test_name]['extra'].append(r.get('patterns_found', 0))
        elif 'memory_mb' in r:
            grouped[test_name]['y'].append(r['memory_mb'])

# Sortuj dane po liczbie świec
for test_name in grouped:
    if grouped[test_name]['extra']:
        data = sorted(zip(grouped[test_name]['x'], grouped[test_name]['y'], grouped[test_name]['extra']))
        grouped[test_name]['x'] = [d[0] for d in data]
        grouped[test_name]['y'] = [d[1] for d in data]
        grouped[test_name]['extra'] = [d[2] for d in data]
    else:
        data = sorted(zip(grouped[test_name]['x'], grouped[test_name]['y']))
        grouped[test_name]['x'] = [d[0] for d in data]
        grouped[test_name]['y'] = [d[1] for d in data]

print(f"✓ Pogrupowano dane dla {len(grouped)} testów")

# ============================================================================
# WYKRES 1: PERF-01 - Wczytywanie danych
# ============================================================================

test_key = 'test_perf_large_dataset_loading'
if test_key in grouped:
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = grouped[test_key]['x']
    y = grouped[test_key]['y']
    
    ax.plot(x, y, marker='o', markersize=10, linewidth=2.5, color='#3498db',
            label='Czas wczytywania', zorder=3)
    ax.fill_between(x, 0, y, alpha=0.2, color='#3498db')
    
    # Adnotacje
    for xi, yi in zip(x, y):
        ax.annotate(f'{yi:.1f}ms', xy=(xi, yi), xytext=(0, 10),
                   textcoords='offset points', ha='center', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))
    
    ax.set_xlabel('Liczba świec', fontsize=13, fontweight='bold')
    ax.set_ylabel('Czas (ms)', fontsize=13, fontweight='bold')
    ax.set_title('PERF-01: Skalowanie wczytywania dużego zbioru danych CSV', 
                 fontsize=15, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'perf01_large_dataset_loading.png'), dpi=600, bbox_inches='tight')
    print("✓ Zapisano: perf01_large_dataset_loading.png")
    plt.close()

# ============================================================================
# WYKRES 2: PERF-02 - Wykrywanie formacji
# ============================================================================

test_key = 'test_perf_pattern_detection_speed'
if test_key in grouped:
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = grouped[test_key]['x']
    y = grouped[test_key]['y']
    
    # Wykres czasu
    ax.plot(x, y, marker='s', markersize=10, linewidth=2.5, color='#e74c3c',
            label='Czas wykrywania formacji', zorder=3)
    ax.fill_between(x, 0, y, alpha=0.2, color='#e74c3c')
    
    # Adnotacje
    for xi, yi in zip(x, y):
        ax.annotate(f'{yi:.1f}ms', xy=(xi, yi), xytext=(0, 10),
                   textcoords='offset points', ha='center', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))
    
    ax.set_xlabel('Liczba świec', fontsize=13, fontweight='bold')
    ax.set_ylabel('Czas (ms)', fontsize=13, fontweight='bold')
    ax.set_title('PERF-02: Skalowanie wykrywania formacji świecowych (TA-Lib)', 
                 fontsize=15, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'perf02_pattern_detection_speed.png'), dpi=600, bbox_inches='tight')
    print("✓ Zapisano: perf02_pattern_detection_speed.png")
    plt.close()

# ============================================================================
# WYKRES 3: PERF-03 - Generowanie wykresów
# ============================================================================

test_key = 'test_perf_chart_generation_speed'
if test_key in grouped:
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = grouped[test_key]['x']
    y = grouped[test_key]['y']
    
    ax.plot(x, y, marker='^', markersize=10, linewidth=2.5, color='#2ecc71',
            label='Czas generowania wykresu', zorder=3)
    ax.fill_between(x, 0, y, alpha=0.2, color='#2ecc71')
    
    for xi, yi in zip(x, y):
        ax.annotate(f'{yi:.1f}ms', xy=(xi, yi), xytext=(0, 10),
                   textcoords='offset points', ha='center', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))
    
    ax.set_xlabel('Liczba świec', fontsize=13, fontweight='bold')
    ax.set_ylabel('Czas (ms)', fontsize=13, fontweight='bold')
    ax.set_title('PERF-03: Skalowanie generowania wykresów interaktywnych (Plotly)', 
                 fontsize=15, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'perf03_chart_generation_speed.png'), dpi=600, bbox_inches='tight')
    print("✓ Zapisano: perf03_chart_generation_speed.png")
    plt.close()

# ============================================================================
# WYKRES 4: PERF-04 - Zużycie pamięci
# ============================================================================

test_key = 'test_perf_memory_usage'
if test_key in grouped:
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = grouped[test_key]['x']
    y = grouped[test_key]['y']
    
    ax.plot(x, y, marker='D', markersize=10, linewidth=2.5, color='#9b59b6',
            label='Zużycie pamięci', zorder=3)
    ax.fill_between(x, 0, y, alpha=0.2, color='#9b59b6')
    
    for xi, yi in zip(x, y):
        ax.annotate(f'{yi:.2f}MB', xy=(xi, yi), xytext=(0, 10),
                   textcoords='offset points', ha='center', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))
    
    ax.set_xlabel('Liczba świec', fontsize=13, fontweight='bold')
    ax.set_ylabel('Pamięć (MB)', fontsize=13, fontweight='bold')
    ax.set_title('PERF-04: Skalowanie zużycia pamięci RAM', 
                 fontsize=15, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'perf04_memory_usage.png'), dpi=600, bbox_inches='tight')
    print("✓ Zapisano: perf04_memory_usage.png")
    plt.close()

# ============================================================================
# WYKRES 5: Porównanie wszystkich testów (znormalizowane)
# ============================================================================

fig, ax = plt.subplots(figsize=(14, 8))

test_mapping = {
    'test_perf_large_dataset_loading': ('PERF-01: Wczytywanie', '#3498db', 'o'),
    'test_perf_pattern_detection_speed': ('PERF-02: Wykrywanie formacji', '#e74c3c', 's'),
    'test_perf_chart_generation_speed': ('PERF-03: Generowanie wykresu', '#2ecc71', '^'),
    'test_perf_memory_usage': ('PERF-04: Pamięć', '#9b59b6', 'D')
}

for test_name, data in grouped.items():
    if test_name in test_mapping:
        label, color, marker = test_mapping[test_name]
        x = data['x']
        y = data['y']
        
        # Normalizacja do 0-100
        y_norm = [(val / max(y)) * 100 for val in y]
        
        ax.plot(x, y_norm, marker=marker, markersize=8, linewidth=2,
                label=label, color=color, zorder=3)

ax.set_xlabel('Liczba świec', fontsize=13, fontweight='bold')
ax.set_ylabel('Względna wydajność (% od maksimum)', fontsize=13, fontweight='bold')
ax.set_title('Porównanie skalowania wszystkich testów wydajnościowych (znormalizowane)', 
             fontsize=15, fontweight='bold', pad=15)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=11, loc='best')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'perf_all_comparison.png'), dpi=600, bbox_inches='tight')
print("✓ Zapisano: perf_all_comparison.png")
plt.close()

# ============================================================================
# TABELA LaTeX
# ============================================================================

print("\n" + "="*80)
print("TABELA LaTeX - Wyniki testów wydajnościowych:")
print("="*80 + "\n")

latex = r"""\begin{table}[h!]
\centering
\caption{Wyniki testów skalowania wydajności aplikacji Candlestick Scanner}
\label{tab:performance_scalability}
\small
\begin{tabular}{|l|r|r|r|r|r|}
\hline
\textbf{Test} & \textbf{100 świec} & \textbf{500 świec} & \textbf{1000 świec} & \textbf{5000 świec} & \textbf{10000 świec} \\
\hline
"""

for test_name in ['test_perf_large_dataset_loading', 'test_perf_pattern_detection_speed',
                   'test_perf_chart_generation_speed', 'test_perf_memory_usage']:
    if test_name in grouped:
        label = test_mapping[test_name][0]
        row = label.replace('PERF-0', 'Test ').replace(':', ' -')
        
        for val in grouped[test_name]['y']:
            if test_name == 'test_perf_memory_usage':
                row += f" & {val:.2f} MB"
            else:
                row += f" & {val:.2f} ms"
        
        latex += row + " \\\\\n\\hline\n"

latex += r"""\end{tabular}
\end{table}"""

print(latex)

# ============================================================================
# STATYSTYKI
# ============================================================================

print("\n" + "="*80)
print("STATYSTYKI:")
print("="*80)

for test_name, (label, _, _) in test_mapping.items():
    if test_name in grouped:
        x = grouped[test_name]['x']
        y = grouped[test_name]['y']
        
        print(f"\n{label}:")
        print(f"  Najmniejszy zbiór ({x[0]} świec): {y[0]:.2f} {'MB' if 'memory' in test_name else 'ms'}")
        print(f"  Największy zbiór ({x[-1]} świec): {y[-1]:.2f} {'MB' if 'memory' in test_name else 'ms'}")
        print(f"  Wzrost: {(y[-1]/y[0]):.2f}x")

print("\n" + "="*80)
print("✅ Wygenerowano wszystkie wykresy!")
print("="*80)
