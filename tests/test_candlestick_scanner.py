"""
Test Suite dla aplikacji Candlestick Scanner
============================================

Struktura testów według modelu V:
1. Testy jednostkowe (Unit Tests) - testowanie pojedynczych metod
2. Testy integracyjne (Integration Tests) - testowanie współpracy komponentów
3. Testy systemowe (System Tests) - testowanie całego systemu
4. Testy wydajności (Performance Tests) - sprawdzanie efektywności
5. Testy zgodności (Compatibility Tests) - różne formaty danych

Uruchomienie:
    pytest test_candlestick_scanner.py -v --tb=short
    pytest test_candlestick_scanner.py -v --cov=models --cov=controllers
    pytest test_candlestick_scanner.py -v -k "test_unit"
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from io import BytesIO
from unittest.mock import Mock, patch, MagicMock
import json

# Import testowanych komponentów
import sys
sys.path.insert(0, os.path.abspath('.'))

from models.candlestick_model import CandlestickModel
from controllers.scanner_controller import ScannerController
from app import create_app
from config import Config


# ============================================================================
# FIXTURES - Przygotowanie środowiska testowego
# ============================================================================

@pytest.fixture
def sample_ohlc_data():
    """Fixture: Przykładowe dane OHLC do testów"""
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    data = pd.DataFrame({
        'open': 100 + np.random.randn(100).cumsum(),
        'high': 102 + np.random.randn(100).cumsum(),
        'low': 98 + np.random.randn(100).cumsum(),
        'close': 100 + np.random.randn(100).cumsum(),
        'volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)
    
    # Zapewnij spójność OHLC (high >= max(open, close), low <= min(open, close))
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)
    
    return data


@pytest.fixture
def sample_csv_file(tmp_path, sample_ohlc_data):
    """Fixture: Tymczasowy plik CSV z danymi OHLC"""
    csv_file = tmp_path / "test_data.csv"
    sample_ohlc_data.to_csv(csv_file)
    return str(csv_file)


@pytest.fixture
def sample_csv_with_different_format(tmp_path):
    """Fixture: CSV z różnymi formatami kolumn (test adaptera)"""
    csv_file = tmp_path / "test_data_different.csv"
    
    # Dane z różnymi nazwami kolumn i formatowaniem
    data = """Date;Opening;Highest;Lowest;Closing;Vol
2024-01-01;$100.50;$102.30;$99.80;$101.20;1,234,567
2024-01-02;$101.20;$103.50;$100.90;$102.80;1,456,789
2024-01-03;$102.80;$104.20;$101.50;$103.90;1,567,890
"""
    csv_file.write_text(data)
    return str(csv_file)



@pytest.fixture
def candlestick_model():
    """Fixture: Instancja modelu"""
    return CandlestickModel()


@pytest.fixture
def candlestick_model_with_data(candlestick_model, sample_ohlc_data):
    """Fixture: Model z załadowanymi danymi"""
    candlestick_model.data = sample_ohlc_data
    return candlestick_model


@pytest.fixture
def flask_app():
    """Fixture: Aplikacja Flask w trybie testowym"""
    app = create_app(Config)
    app.config['TESTING'] = True
    app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
    return app


@pytest.fixture
def client(flask_app):
    """Fixture: Klient testowy Flask"""
    return flask_app.test_client()


@pytest.fixture
def scanner_controller(flask_app):
    """Fixture: Kontroler z mockiem aplikacji"""
    return ScannerController(flask_app)


# ============================================================================
# 1. TESTY JEDNOSTKOWE (UNIT TESTS)
# ============================================================================

class TestUnitModel:
    """Testy jednostkowe dla modelu CandlestickModel"""
    
    def test_unit_model_initialization(self, candlestick_model):
        """Test UT-01: Inicjalizacja modelu"""
        assert candlestick_model.data is None
        assert candlestick_model.patterns == {}
        assert candlestick_model.support_resistance_levels == []
        assert candlestick_model.source_filename is None
    
    def test_unit_load_data_from_csv(self, candlestick_model, sample_csv_file):
        """Test UT-02: Wczytywanie danych z pliku CSV"""
        result = candlestick_model.load_data_from_file(sample_csv_file)
        
        assert result is True
        assert candlestick_model.data is not None
        assert len(candlestick_model.data) == 100
        assert all(col in candlestick_model.data.columns for col in ['open', 'high', 'low', 'close'])
    
    def test_unit_load_data_adapter_different_columns(self, candlestick_model, sample_csv_with_different_format):
        """Test UT-03: Adapter - różne formaty kolumn"""
        result = candlestick_model.load_data_from_file(sample_csv_with_different_format)
        
        assert result is True
        assert 'open' in candlestick_model.data.columns
        assert 'close' in candlestick_model.data.columns
        # Sprawdź czy wartości zostały oczyszczone z $, przecinków
        assert candlestick_model.data['open'].dtype in [np.float64, float]
    
    def test_unit_load_data_invalid_file(self, candlestick_model):
        """Test UT-04: Obsługa nieprawidłowego pliku"""
        result = candlestick_model.load_data_from_file("nonexistent_file.csv")
        assert result is False
    
    def test_unit_detect_patterns_without_data(self, candlestick_model):
        """Test UT-05: Wykrywanie formacji bez danych (Null Object Pattern)"""
        patterns = candlestick_model.detect_patterns()
        
        assert isinstance(patterns, dict)
        assert len(patterns) == 0  # Pusty słownik zamiast None
    
    def test_unit_detect_patterns_with_data(self, candlestick_model_with_data):
        """Test UT-06: Wykrywanie formacji z danymi"""
        patterns = candlestick_model_with_data.detect_patterns()
        
        assert isinstance(patterns, dict)
        # Powinny być wykryte jakieś formacje (lub pusty dict jeśli nie wykryto)
        assert patterns is not None
    
    def test_unit_interpret_patterns(self, candlestick_model_with_data):
        """Test UT-07: Interpretacja formacji"""
        candlestick_model_with_data.detect_patterns()
        interpretations = candlestick_model_with_data.interpret_patterns()
        
        assert isinstance(interpretations, list)
        # Jeśli są interpretacje, sprawdź strukturę
        if len(interpretations) > 0:
            assert 'pattern' in interpretations[0]
            assert 'trend' in interpretations[0]
            assert 'date' in interpretations[0]
    
    def test_unit_calculate_support_resistance(self, candlestick_model_with_data):
        """Test UT-08: Obliczanie poziomów wsparcia i oporu"""
        levels = candlestick_model_with_data.calculate_support_resistance(window=10)
        
        assert isinstance(levels, list)
        # Sprawdź strukturę jeśli są poziomy
        if len(levels) > 0:
            assert len(levels[0]) == 2  # (level, type)
            assert levels[0][1] in ['Wsparcie', 'Opór']
    
    def test_unit_verify_pattern_effectiveness(self, candlestick_model_with_data):
        """Test UT-09: Weryfikacja skuteczności formacji"""
        candlestick_model_with_data.detect_patterns()
        candlestick_model_with_data.calculate_support_resistance()
        
        effectiveness = candlestick_model_with_data.verify_pattern_effectiveness(lookback_max=5)
        
        assert isinstance(effectiveness, list)
        # Jeśli są wyniki, sprawdź strukturę
        if len(effectiveness) > 0:
            assert 'pattern' in effectiveness[0]
            assert 'effective' in effectiveness[0]
            assert 'reliability' in effectiveness[0]
    
    def test_unit_null_object_pattern_empty_patterns(self, candlestick_model):
        """Test UT-10: Null Object Pattern - puste kolekcje"""
        # Test że metody zwracają puste kolekcje zamiast None
        patterns = candlestick_model.detect_patterns()
        interpretations = candlestick_model.interpret_patterns()
        levels = candlestick_model.calculate_support_resistance()
        
        assert patterns == {}
        assert interpretations == []
        assert levels == []
        # Wszystkie są iterowalne i nie rzucają błędów
        for _ in patterns.items():
            pass
        for _ in interpretations:
            pass
        for _ in levels:
            pass


class TestUnitController:
    """Testy jednostkowe dla kontrolera ScannerController"""
    
    def test_unit_controller_initialization(self, scanner_controller):
        """Test UT-11: Inicjalizacja kontrolera"""
        assert scanner_controller.app is not None
        assert scanner_controller.model is not None
        assert isinstance(scanner_controller.model, CandlestickModel)
    
    def test_unit_allowed_file_valid(self, scanner_controller):
        """Test UT-12: Walidacja rozszerzenia pliku - poprawne"""
        assert scanner_controller._allowed_file('data.csv') is True
    
    def test_unit_allowed_file_invalid(self, scanner_controller):
        """Test UT-13: Walidacja rozszerzenia pliku - niepoprawne"""
        assert scanner_controller._allowed_file('data.txt') is False
        assert scanner_controller._allowed_file('data.exe') is False
        assert scanner_controller._allowed_file('data') is False


# ============================================================================
# 2. TESTY INTEGRACYJNE (INTEGRATION TESTS)
# ============================================================================

class TestIntegrationControllerModel:
    """Testy integracji kontrolera z modelem"""
    
    def test_integration_upload_and_scan(self, client, sample_csv_file):
        """Test IT-01: Integracja upload → model → scan"""
        # 1. Upload pliku
        with open(sample_csv_file, 'rb') as f:
            response = client.post('/upload', data={
                'file': (f, 'test_data.csv')
            }, content_type='multipart/form-data')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        
        # 2. Skanowanie formacji
        response = client.post('/api/scan')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'patterns_count' in data
        assert data['success'] is True
    
    def test_integration_load_ticker_and_scan(self, client):
        """Test IT-02: Integracja ticker API → model → scan"""
        # 1. Załaduj dane z Yahoo Finance
        with patch('yfinance.download') as mock_download:
            # Mock danych z yfinance
            dates = pd.date_range('2024-01-01', periods=50, freq='D')
            mock_data = pd.DataFrame({
                'Open': np.random.randn(50).cumsum() + 100,
                'High': np.random.randn(50).cumsum() + 102,
                'Low': np.random.randn(50).cumsum() + 98,
                'Close': np.random.randn(50).cumsum() + 100,
                'Volume': np.random.randint(1000000, 5000000, 50)
            }, index=dates)
            mock_download.return_value = mock_data
            
            response = client.post('/api/load-ticker', 
                json={'ticker': 'AAPL', 'period': '1mo'})
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['success'] is True
        
        # 2. Skanowanie
        response = client.post('/api/scan')
        assert response.status_code == 200
    
    def test_integration_full_workflow(self, scanner_controller, sample_csv_file):
        """Test IT-03: Pełny workflow - load → detect → interpret → verify"""
        # 1. Załaduj dane
        result = scanner_controller.model.load_data_from_file(sample_csv_file)
        assert result is True
        
        # 2. Wykryj formacje
        patterns = scanner_controller.model.detect_patterns()
        assert isinstance(patterns, dict)
        
        # 3. Oblicz wsparcie/opór
        levels = scanner_controller.model.calculate_support_resistance()
        assert isinstance(levels, list)
        
        # 4. Interpretuj
        interpretations = scanner_controller.model.interpret_patterns()
        assert isinstance(interpretations, list)
        
        # 5. Weryfikuj skuteczność
        effectiveness = scanner_controller.model.verify_pattern_effectiveness()
        assert isinstance(effectiveness, list)


# ============================================================================
# 3. TESTY SYSTEMOWE (SYSTEM TESTS)
# ============================================================================

class TestSystemEndToEnd:
    """Testy systemowe - końcowy workflow użytkownika"""
    
    def test_system_complete_csv_analysis(self, client, sample_csv_file):
        """Test ST-01: Kompletna analiza z pliku CSV"""
        # Scenariusz: Użytkownik wgrywa CSV i dostaje wyniki
        
        # Krok 1: Upload pliku
        with open(sample_csv_file, 'rb') as f:
            response = client.post('/upload', data={
                'file': (f, 'test.csv')
            }, content_type='multipart/form-data')
        assert response.status_code == 200
        
        # Krok 2: Skanowanie
        response = client.post('/api/scan')
        assert response.status_code == 200
        result = json.loads(response.data)
        assert result['success'] is True
        
        # Krok 3: Wyświetlenie wyników (GET /results)
        response = client.get('/results')
        assert response.status_code == 200
    
    def test_system_error_handling_no_data(self, client):
        """Test ST-02: Obsługa błędu - skanowanie bez danych"""
        response = client.post('/api/scan')
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_system_error_handling_invalid_file(self, client, tmp_path):
        """Test ST-03: Obsługa błędu - nieprawidłowy plik"""
        invalid_file = tmp_path / "invalid.txt"
        invalid_file.write_text("not a csv")
        
        with open(invalid_file, 'rb') as f:
            response = client.post('/upload', data={
                'file': (f, 'invalid.txt')
            }, content_type='multipart/form-data')
        
        assert response.status_code == 400
    
    def test_system_mvc_separation(self, scanner_controller, sample_csv_file):
        """Test ST-04: Separacja MVC - model nie zna kontrolera"""
        # Model powinien działać niezależnie od kontrolera
        model = CandlestickModel()
        
        # Model powinien działać bez Flask
        result = model.load_data_from_file(sample_csv_file)
        assert result is True
        
        patterns = model.detect_patterns()
        assert isinstance(patterns, dict)
        
        # Model nie ma żadnych referencji do Flask/HTTP
        assert not hasattr(model, 'app')
        assert not hasattr(model, 'request')


# ============================================================================
# 4. TESTY WYDAJNOŚCIOWE (PERFORMANCE TESTS)
# ============================================================================

class TestPerformance:
    """Testy wydajności i efektywności"""
    
    def test_perf_large_dataset_loading(self, candlestick_model, tmp_path):
        """Test PERF-01: Wczytywanie dużego zbioru danych (10000 świec)"""
        import time
        
        # Przygotuj duży CSV
        dates = pd.date_range('2000-01-01', periods=10000, freq='D')
        large_data = pd.DataFrame({
            'open': np.random.randn(10000).cumsum() + 100,
            'high': np.random.randn(10000).cumsum() + 102,
            'low': np.random.randn(10000).cumsum() + 98,
            'close': np.random.randn(10000).cumsum() + 100,
            'volume': np.random.randint(1000000, 5000000, 10000)
        }, index=dates)
        
        csv_file = tmp_path / "large_data.csv"
        large_data.to_csv(csv_file)
        
        # Zmierz czas wczytywania
        start = time.time()
        result = candlestick_model.load_data_from_file(str(csv_file))
        elapsed = time.time() - start
        
        assert result is True
        assert len(candlestick_model.data) == 10000
        assert elapsed < 5.0  # Maksymalnie 5 sekund
        print(f"\n✓ Wczytano 10000 świec w {elapsed:.2f}s")
    
    def test_perf_pattern_detection_speed(self, candlestick_model_with_data):
        """Test PERF-02: Szybkość wykrywania formacji"""
        import time
        
        start = time.time()
        patterns = candlestick_model_with_data.detect_patterns()
        elapsed = time.time() - start
        
        assert elapsed < 2.0  # Maksymalnie 2 sekundy dla 100 świec
        print(f"\n✓ Wykryto formacje w {elapsed:.2f}s")
    
    def test_perf_chart_generation_speed(self, candlestick_model_with_data):
        """Test PERF-03: Szybkość generowania wykresów"""
        import time
        
        candlestick_model_with_data.detect_patterns()
        candlestick_model_with_data.calculate_support_resistance()
        
        start = time.time()
        chart = candlestick_model_with_data.generate_chart()
        elapsed = time.time() - start
        
        assert chart is not None
        assert elapsed < 3.0  # Maksymalnie 3 sekundy
        print(f"\n✓ Wygenerowano wykres w {elapsed:.2f}s")
    
    def test_perf_memory_usage(self, candlestick_model, tmp_path):
        """Test PERF-04: Zużycie pamięci dla dużego zbioru"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Załaduj duży zbiór
        dates = pd.date_range('2000-01-01', periods=10000, freq='D')
        large_data = pd.DataFrame({
            'open': np.random.randn(10000).cumsum() + 100,
            'high': np.random.randn(10000).cumsum() + 102,
            'low': np.random.randn(10000).cumsum() + 98,
            'close': np.random.randn(10000).cumsum() + 100,
            'volume': np.random.randint(1000000, 5000000, 10000)
        }, index=dates)
        
        csv_file = tmp_path / "large_data.csv"
        large_data.to_csv(csv_file)
        
        candlestick_model.load_data_from_file(str(csv_file))
        candlestick_model.detect_patterns()
        
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_increase = mem_after - mem_before
        
        # Zużycie nie powinno przekroczyć 100 MB dla 10k świec
        assert mem_increase < 100
        print(f"\n✓ Wzrost pamięci: {mem_increase:.2f} MB")


# ============================================================================
# 5. TESTY ZGODNOŚCI (COMPATIBILITY TESTS)
# ============================================================================

class TestCompatibility:
    """Testy zgodności z różnymi formatami danych"""
    
    def test_compat_csv_comma_separator(self, candlestick_model, tmp_path):
        """Test COMP-01: CSV z przecinkami"""
        csv_file = tmp_path / "comma.csv"
        csv_file.write_text("""date,open,high,low,close,volume
2024-01-01,100,102,99,101,1000000
2024-01-02,101,103,100,102,1100000
""", encoding='utf-8')
        
        result = candlestick_model.load_data_from_file(str(csv_file))
        assert result is True
        assert len(candlestick_model.data) == 2
    
    def test_compat_csv_semicolon_separator(self, candlestick_model, tmp_path):
        """Test COMP-02: CSV ze średnikami (polski Excel)"""
        csv_file = tmp_path / "semicolon.csv"
        csv_file.write_text("""date;open;high;low;close;volume
2024-01-01;100;102;99;101;1000000
2024-01-02;101;103;100;102;1100000
""", encoding='utf-8')
        
        result = candlestick_model.load_data_from_file(str(csv_file))
        assert result is True
        assert len(candlestick_model.data) == 2
    
    def test_compat_csv_with_currency_symbols(self, candlestick_model, tmp_path):
        """Test COMP-03: CSV z symbolami walut ($, €, £)"""
        csv_file = tmp_path / "currency.csv"
        csv_file.write_text("""date,open,high,low,close,volume
2024-01-01,$100.50,$102.30,$99.80,$101.20,"1,234,567"
2024-01-02,€101.20,€103.50,€100.90,€102.80,"1,456,789"
2024-01-03,£102.80,£104.20,£101.50,£103.90,"1,567,890"
""", encoding='utf-8')
        
        result = candlestick_model.load_data_from_file(str(csv_file))
        assert result is True
        assert candlestick_model.data['open'].dtype in [np.float64, float]
        assert candlestick_model.data['open'].iloc[0] > 90

    
    def test_compat_various_column_names(self, candlestick_model, tmp_path):
        """Test COMP-04: Różne nazwy kolumn"""
        test_cases = [
            ("Open,High,Low,Close,Volume", "CamelCase"),
            ("OPEN,HIGH,LOW,CLOSE,VOLUME", "UPPERCASE"),
            ("o,h,l,c,v", "single_letter"),
            ("Opening,Highest,Lowest,Closing,Vol", "full_words"),
        ]
        
        for columns, test_name in test_cases:
            csv_file = tmp_path / f"columns_{test_name}.csv"
            csv_file.write_text(f"""{columns}
    100,102,99,101,1000000
    101,103,100,102,1100000
    """, encoding='utf-8')
            
            result = candlestick_model.load_data_from_file(str(csv_file))
            assert result is True, f"Failed for {test_name}"
            assert 'open' in candlestick_model.data.columns
            assert 'close' in candlestick_model.data.columns

    
    def test_compat_date_formats(self, candlestick_model, tmp_path):
        """Test COMP-05: Różne formaty dat"""
        date_formats = [
            ("2024-01-01", "ISO format"),
            ("01/01/2024", "US format"),
            ("01-01-2024", "dash format"),
        ]
        
        for date_str, format_name in date_formats:
            csv_file = tmp_path / f"date_{format_name.replace(' ', '_')}.csv"
            csv_file.write_text(f"""date,open,high,low,close,volume
{date_str},100,102,99,101,1000000
""")
            
            result = candlestick_model.load_data_from_file(str(csv_file))
            assert result is True, f"Failed for {format_name}"


# ============================================================================
# 6. TESTY WZORCÓW PROJEKTOWYCH (DESIGN PATTERN TESTS)
# ============================================================================

# ============================================================================
# 6. TESTY WZORCÓW PROJEKTOWYCH (DESIGN PATTERN TESTS)
# ============================================================================

class TestDesignPatterns:
    """Weryfikacja poprawności implementacji wzorców projektowych"""
    
    def test_pattern_strategy_data_loading(self, candlestick_model, sample_csv_file):
        """Test DP-01: Wzorzec Strategy - różne strategie ładowania"""
        # Strategia 1: CSV
        result1 = candlestick_model.load_data_from_file(sample_csv_file)
        assert result1 is True
        
        # Strategia 2: Yahoo Finance (z mockiem)
        with patch('yfinance.download') as mock:
            mock_data = pd.DataFrame({
                'Open': [100], 'High': [102], 'Low': [98], 
                'Close': [101], 'Volume': [1000000]
            }, index=pd.date_range('2024-01-01', periods=1))
            mock.return_value = mock_data
            
            result2 = candlestick_model.load_data_from_ticker('AAPL', '1d')
            assert result2 is True
    
    def test_pattern_facade_detect_patterns(self, candlestick_model_with_data):
        """Test DP-02: Wzorzec Facade - ukrycie złożoności TA-Lib"""
        # Jedna metoda ukrywa 61 funkcji TA-Lib
        patterns = candlestick_model_with_data.detect_patterns()
        
        assert isinstance(patterns, dict)
        # Użytkownik nie musi wiedzieć o CDLHAMMER, CDLDOJI, etc.
        # Wszystko jest ukryte za prostym interfejsem
    
    def test_pattern_factory_create_app(self):
        """Test DP-03: Wzorzec Factory - tworzenie aplikacji"""
        # Factory method tworzy aplikację z konfiguracją
        app1 = create_app(Config)
        assert app1 is not None
        
        # Można utworzyć różne instancje z różnymi konfiguracjami
        class TestConfig(Config):
            TESTING = True
        
        app2 = create_app(TestConfig)
        assert app2 is not None
        assert app2.config['TESTING'] is True
    
    def test_pattern_command_encapsulation(self, scanner_controller, sample_csv_file):
        """Test DP-04: Wzorzec Command - enkapsulacja żądań"""
        # Każda metoda kontrolera to Command enkapsulujące żądanie
        scanner_controller.model.load_data_from_file(sample_csv_file)
        
        # Command: scan_patterns() - enkapsuluje całą logikę skanowania
        # Wywołanie wymaga kontekstu aplikacji Flask (bo używa jsonify)
        with scanner_controller.app.app_context():
            result = scanner_controller.scan_patterns()
        
        assert result is not None
        assert isinstance(result, tuple)  # (response, status_code)
        
        # Sprawdź strukturę odpowiedzi
        response, status_code = result
        assert status_code == 200


# ============================================================================
# URUCHOMIENIE TESTÓW
# ============================================================================

if __name__ == '__main__':
    """
    Uruchomienie testów z terminala:
    
    # Wszystkie testy z szczegółami
    pytest test_candlestick_scanner.py -v
    
    # Tylko testy jednostkowe
    pytest test_candlestick_scanner.py -v -k "test_unit"
    
    # Z pokryciem kodu
    pytest test_candlestick_scanner.py -v --cov=models --cov=controllers
    
    # Z raportem HTML
    pytest test_candlestick_scanner.py -v --html=report.html --self-contained-html
    """
    pytest.main([__file__, '-v', '--tb=short'])
