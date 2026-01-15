"""
Konfiguracja pytest i hooki dla testów wydajnościowych.
Ten plik jest automatycznie wykrywany przez pytest.
"""

import pytest
import json
import os

# Globalna lista wyników wydajnościowych
_performance_results = []


def add_performance_result(result_dict):
    """
    Funkcja pomocnicza do dodawania wyników wydajnościowych.
    Używana w testach przez fixture perf_tracker.
    """
    _performance_results.append(result_dict)
    print(f"[TRACKER] Dodano wynik: {result_dict['test']}")


@pytest.fixture
def perf_tracker():
    """
    Fixture dostarczająca funkcję do zapisywania wyników wydajności.
    Użycie w teście: perf_tracker({'test': 'nazwa', 'czas_ms': 123, ...})
    """
    return add_performance_result


def pytest_sessionfinish(session, exitstatus):
    """
    Hook pytest wywoływany automatycznie po zakończeniu wszystkich testów.
    Zapisuje zebrane wyniki wydajnościowe do pliku JSON.
    """
    print(f"\n{'='*70}")
    print(f"[HOOK conftest.py] pytest_sessionfinish wywołany!")
    print(f"[HOOK] Exitstatus: {exitstatus}")
    print(f"[HOOK] Liczba wyników wydajnościowych: {len(_performance_results)}")
    print(f"{'='*70}")
    
    if _performance_results:
        # Ścieżka do zapisu wyników
        output_file = 'tests/results/performance_results.json'
        
        # Utwórz katalog jeśli nie istnieje
        os.makedirs('tests/results', exist_ok=True)
        
        # Zapisz do JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(_performance_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ SUKCES! Zapisano {len(_performance_results)} wyników wydajnościowych do:")
        print(f"   📁 {os.path.abspath(output_file)}\n")
        
        # Wyświetl podsumowanie
        print("="*70)
        print("PODSUMOWANIE WYNIKÓW:")
        print("="*70)
        for result in _performance_results:
            status_icon = "✅" if result['status'] == 'PASSED' else "❌"
            test_name = result['test']
            num_candles = result.get('num_candles', 'N/A')
            
            # Sprawdź czy to test czasu czy pamięci
            if 'czas_ms' in result:
                print(f"{status_icon} {test_name} [{num_candles} świec]: {result['czas_ms']}ms / {result['limit_ms']}ms")
            elif 'memory_mb' in result:
                print(f"{status_icon} {test_name} [{num_candles} świec]: {result['memory_mb']}MB / {result['limit_mb']}MB")
        print("="*70 + "\n")
    else:
        print("\n⚠️ UWAGA: Brak wyników wydajnościowych do zapisania!\n")

