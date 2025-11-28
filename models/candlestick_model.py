import matplotlib
matplotlib.use('Agg')
import talib
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Tuple
import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from io import BytesIO
import base64

class CandlestickModel:
    """Model odpowiedzialny za analizę formacji świecowych"""
    
    def __init__(self):
        self.data = None
        self.patterns = {}
        self.support_resistance_levels = []
        self.source_filename = None

        import matplotlib
        matplotlib.use('Agg')

    @staticmethod
    def _cleanup_matplotlib():
        """Czyści zasoby matplotlib"""
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
            import gc
            gc.collect()
        except:
            pass
    
    def load_data_from_file(self, filepath: str) -> bool:
        """Wczytuje dane z pliku CSV z elastycznym rozpoznawaniem kolumn"""
        try:
            # Spróbuj wykryć separator
            with open(filepath, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
            
            # Wybierz separator
            if ';' in first_line and first_line.count(';') > first_line.count(','):
                separator = ';'
                print("Wykryto separator: średnik (;)")
            else:
                separator = ','
                print("Wykryto separator: przecinek (,)")
            
            # Wczytaj dane
            self.data = pd.read_csv(filepath, sep=separator)
            
            print(f"Oryginalne kolumny: {list(self.data.columns)}")
            
            # Wyczyść nazwy kolumn (usuń białe znaki, znaki specjalne)
            self.data.columns = self.data.columns.str.strip().str.lower()
            
            # Zamień problematyczne znaki w nazwach kolumn
            self.data.columns = self.data.columns.str.replace('/', '_', regex=False)
            self.data.columns = self.data.columns.str.replace(' ', '_', regex=False)
            
            print(f"Kolumny po czyszczeniu: {list(self.data.columns)}")
            
            # Mapowanie możliwych nazw kolumn (różne warianty)
            column_mapping = {
                'open': ['open', 'open_price', 'opening_price', 'o', 'opening', 'open price'],
                'high': ['high', 'high_price', 'highest', 'h', 'hi', 'high price'],
                'low': ['low', 'low_price', 'lowest', 'l', 'lo', 'low price'],
                'close': ['close', 'close_price', 'closing_price', 'c', 'closing', 'last', 
                        'close_last', 'close/last'],
                'volume': ['volume', 'vol', 'v', 'quantity', 'qty', 'shares'],
                'date': ['date', 'datetime', 'time', 'timestamp', 'day']
            }
            
            # Znajdź i zmapuj kolumny
            renamed_columns = {}
            
            for standard_name, possible_names in column_mapping.items():
                for col in self.data.columns:
                    # Sprawdź czy kolumna pasuje do którejś z możliwych nazw
                    if col in possible_names or any(possible in col for possible in possible_names):
                        renamed_columns[col] = standard_name
                        break
            
            # Zmień nazwy kolumn
            if renamed_columns:
                self.data.rename(columns=renamed_columns, inplace=True)
                print(f"Zmapowane kolumny: {renamed_columns}")
            
            print(f"Końcowe kolumny: {list(self.data.columns)}")
            
            # Walidacja wymaganych kolumn
            required_cols = ['open', 'high', 'low', 'close']
            missing_cols = [col for col in required_cols if col not in self.data.columns]
            
            if missing_cols:
                print(f"❌ Brak wymaganych kolumn: {missing_cols}")
                print(f"Dostępne kolumny: {list(self.data.columns)}")
                return False
            
            # NOWE: Czyszczenie wartości przed konwersją
            for col in required_cols:
                if col in self.data.columns:
                    # Usuń znaki dolara, przecinki i inne symbole walut
                    if self.data[col].dtype == 'object':
                        self.data[col] = self.data[col].astype(str).str.replace('$', '', regex=False)
                        self.data[col] = self.data[col].str.replace('€', '', regex=False)
                        self.data[col] = self.data[col].str.replace('£', '', regex=False)
                        self.data[col] = self.data[col].str.replace(',', '', regex=False)
                        self.data[col] = self.data[col].str.strip()
                    
                    # Konwertuj do liczb
                    self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
            
            # Konwersja volume jeśli istnieje
            if 'volume' in self.data.columns:
                if self.data['volume'].dtype == 'object':
                    # Usuń znaki dolara, przecinki
                    self.data['volume'] = self.data['volume'].astype(str).str.replace('$', '', regex=False)
                    self.data['volume'] = self.data['volume'].astype(str).str.replace(',', '', regex=False)
                    self.data['volume'] = self.data['volume'].str.strip()
                
                self.data['volume'] = pd.to_numeric(self.data['volume'], errors='coerce')
            
            # Obsługa kolumny daty
            date_cols = ['date', 'datetime', 'time', 'timestamp']
            date_col_found = None
            
            for date_col in date_cols:
                if date_col in self.data.columns:
                    date_col_found = date_col
                    break
            
            if date_col_found:
                try:
                    # Próbuj różnych formatów daty (automatyczne wykrywanie)
                    self.data[date_col_found] = pd.to_datetime(
                        self.data[date_col_found], 
                        errors='coerce',
                        infer_datetime_format=True
                    )
                    
                    # Sortuj po dacie rosnąco (od najstarszej do najnowszej)
                    self.data.sort_values(by=date_col_found, inplace=True)
                    
                    self.data.set_index(date_col_found, inplace=True)
                    print(f"✓ Ustawiono indeks na kolumnę: {date_col_found}")
                except Exception as e:
                    print(f"Ostrzeżenie: Nie udało się ustawić indeksu daty: {e}")
            else:
                # Jeśli brak kolumny daty, utwórz indeks numeryczny
                print("⚠ Brak kolumny daty - używam indeksu numerycznego")
            
            # Usuń wiersze z NaN w kluczowych kolumnach
            initial_rows = len(self.data)
            self.data.dropna(subset=required_cols, inplace=True)
            dropped_rows = initial_rows - len(self.data)
            
            if dropped_rows > 0:
                print(f"⚠ Usunięto {dropped_rows} wierszy z brakującymi danymi")
            
            if len(self.data) == 0:
                print("❌ Brak poprawnych danych po czyszczeniu")
                return False
            
            # Sprawdź zakres wartości (podstawowa walidacja)
            for col in required_cols:
                min_val = self.data[col].min()
                max_val = self.data[col].max()
                
                if min_val <= 0:
                    print(f"⚠ Ostrzeżenie: Kolumna {col} zawiera wartości <= 0 (min: {min_val})")
                
                print(f"  {col}: min={min_val:.2f}, max={max_val:.2f}")
            
            print(f"✓ Wczytano {len(self.data)} wierszy danych")
            if hasattr(self.data.index, '__getitem__') and len(self.data) > 0:
                print(f"Zakres dat: {self.data.index[0]} do {self.data.index[-1]}")
            
            import os
            self.source_filename = os.path.splitext(os.path.basename(filepath))[0]
            print(f"Nazwa źródła: {self.source_filename}")

            return True
            
        except Exception as e:
            print(f"❌ Błąd wczytywania danych: {e}")
            import traceback
            traceback.print_exc()
            return False

    
    def load_data_from_ticker(self, ticker: str, period: str = '1y') -> bool:
        """Pobiera dane z Yahoo Finance"""
        try:
            # Pobierz dane z auto_adjust=True (domyślne od wersji 0.2.45+)
            self.data = yf.download(
                ticker, 
                period=period, 
                progress=False, 
                auto_adjust=True  # Unikaj ostrzeżenia FutureWarning
            )
            
            # Sprawdź czy dane zostały pobrane
            if self.data is None or len(self.data) == 0:
                print(f"Brak danych dla symbolu: {ticker}")
                return False
            
            # FIX: Obsługa MultiIndex kolumn (yfinance 0.2.45+)
            if isinstance(self.data.columns, pd.MultiIndex):
                # Dla pojedynczego tickera - weź pierwszy poziom
                self.data.columns = self.data.columns.get_level_values(0)
            
            # Bezpieczna konwersja kolumn na małe litery
            self.data.columns = [
                col.lower() if isinstance(col, str) else str(col).lower() 
                for col in self.data.columns
            ]
            
            # Walidacja wymaganych kolumn
            required_cols = ['open', 'high', 'low', 'close']
            if not all(col in self.data.columns for col in required_cols):
                print(f"Brak wymaganych kolumn. Dostępne: {list(self.data.columns)}")
                return False
            
            print(f"✓ Pobrano {len(self.data)} wierszy danych dla {ticker}")

            self.source_filename = f"{ticker}_{period}"
            
            return True
            
        except Exception as e:
            print(f"Błąd pobierania danych: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def detect_patterns(self) -> Dict[str, pd.Series]:
        """Wykrywa formacje świecowe przy użyciu TA-Lib"""
        if self.data is None or len(self.data) == 0:
            return {}
        
        open_prices = self.data['open'].values
        high_prices = self.data['high'].values
        low_prices = self.data['low'].values
        close_prices = self.data['close'].values
        
        # KOMPLETNA LISTA WSZYSTKICH 61 FORMACJI Z TA-LIB
        pattern_functions = [
            # === Formacje jednoświecowe ===
            ('CDLDOJI', 'Doji'),
            ('CDLDOJISTAR', 'Doji Star'),
            ('CDLDRAGONFLYDOJI', 'Dragonfly Doji'),
            ('CDLGRAVESTONEDOJI', 'Gravestone Doji'),
            ('CDLLONGLEGGEDDOJI', 'Long Legged Doji'),
            ('CDLHAMMER', 'Hammer'),
            ('CDLINVERTEDHAMMER', 'Inverted Hammer'),
            ('CDLHANGINGMAN', 'Hanging Man'),
            ('CDLSHOOTINGSTAR', 'Shooting Star'),
            ('CDLMARUBOZU', 'Marubozu'),
            ('CDLCLOSINGMARUBOZU', 'Closing Marubozu'),
            ('CDLSPINNINGTOP', 'Spinning Top'),
            ('CDLHIGHWAVE', 'High-Wave Candle'),
            ('CDLRICKSHAWMAN', 'Rickshaw Man'),
            ('CDLSHORTLINE', 'Short Line Candle'),
            ('CDLLONGLINE', 'Long Line Candle'),
            ('CDLBELTHOLD', 'Belt-hold'),
            ('CDLTAKURI', 'Takuri'),
            
            # === Formacje dwuświecowe ===
            ('CDLENGULFING', 'Engulfing'),
            ('CDLHARAMI', 'Harami'),
            ('CDLHARAMICROSS', 'Harami Cross'),
            ('CDLPIERCING', 'Piercing'),
            ('CDLDARKCLOUDCOVER', 'Dark Cloud Cover'),
            ('CDLTWEEZERSBOTTOM', 'Tweezers Bottom'),
            ('CDLTWEEZERSTOP', 'Tweezers Top'),
            ('CDLKICKING', 'Kicking'),
            ('CDLKICKINGBYLENGTH', 'Kicking Bull/Bear'),
            ('CDLMATCHINGLOW', 'Matching Low'),
            ('CDLHOMINGPIGEON', 'Homing Pigeon'),
            ('CDLINNECK', 'In-Neck'),
            ('CDLONNECK', 'On-Neck'),
            ('CDLTHRUSTING', 'Thrusting'),
            ('CDLSEPARATINGLINES', 'Separating Lines'),
            ('CDLCOUNTERATTACK', 'Counterattack'),
            ('CDL2CROWS', 'Two Crows'),
            ('CDLUPSIDEGAP2CROWS', 'Upside Gap Two Crows'),
            ('CDLGAPSIDESIDEWHITE', 'Gap Side-by-Side White'),
            
            # === Formacje trzyświecowe ===
            ('CDLMORNINGSTAR', 'Morning Star'),
            ('CDLMORNINGDOJISTAR', 'Morning Doji Star'),
            ('CDLEVENINGSTAR', 'Evening Star'),
            ('CDLEVENINGDOJISTAR', 'Evening Doji Star'),
            ('CDL3WHITESOLDIERS', 'Three White Soldiers'),
            ('CDL3BLACKCROWS', 'Three Black Crows'),
            ('CDL3INSIDE', 'Three Inside Up/Down'),
            ('CDL3OUTSIDE', 'Three Outside Up/Down'),
            ('CDLABANDONEDBABY', 'Abandoned Baby'),
            ('CDLTRISTAR', 'Tristar'),
            ('CDL3LINESTRIKE', 'Three Line Strike'),
            ('CDL3STARSINSOUTH', 'Three Stars In The South'),
            ('CDLIDENTICAL3CROWS', 'Identical Three Crows'),
            ('CDLUNIQUE3RIVER', 'Unique 3 River'),
            ('CDLXSIDEGAP3METHODS', 'Gap Three Methods'),
            
            # === Formacje wieloświecowe (4-5 świec) ===
            ('CDLCONCEALBABYSWALL', 'Concealing Baby Swallow'),
            ('CDLBREAKAWAY', 'Breakaway'),
            ('CDLLADDERBOTTOM', 'Ladder Bottom'),
            ('CDLADVANCEBLOCK', 'Advance Block'),
            ('CDLSTALLEDPATTERN', 'Stalled Pattern'),
            ('CDLMATHOLD', 'Mat Hold'),
            ('CDLRISEFALL3METHODS', 'Rising/Falling Three Methods'),
            ('CDLSTICKSANDWICH', 'Stick Sandwich'),
            ('CDLHIKKAKE', 'Hikkake'),
            ('CDLHIKKAKEMOD', 'Modified Hikkake'),
            ('CDLTASUKIGAP', 'Tasuki Gap'),
        ]
        
        detected_patterns = {}
        
        for func_name, pattern_name in pattern_functions:
            try:
                func = getattr(talib, func_name)
                result = func(open_prices, high_prices, low_prices, close_prices)
                
                # Zapisz tylko jeśli wykryto formację
                if np.any(result != 0):
                    detected_patterns[pattern_name] = result
            except Exception as e:
                print(f"Błąd wykrywania {pattern_name}: {e}")
        
        self.patterns = detected_patterns
        print(f"Wykryto {len(detected_patterns)} różnych typów formacji")
        return detected_patterns
    
    def interpret_patterns(self) -> List[Dict]:
        """Interpretuje wykryte formacje jako wzrostowe/spadkowe"""
        interpretations = []
        
        # Rozszerzony słownik interpretacji formacji
        bullish_patterns = [
            # Jednoświecowe wzrostowe
            'Hammer', 'Inverted Hammer', 'Dragonfly Doji', 'Takuri', 'Belt-hold',
            # Dwuświecowe wzrostowe
            'Piercing', 'Tweezers Bottom', 'Homing Pigeon', 'Matching Low',
            # Trzyświecowe wzrostowe
            'Morning Star', 'Morning Doji Star', 'Three White Soldiers', 
            'Three Inside Up/Down', 'Three Outside Up/Down',
            # Wieloświecowe wzrostowe
            'Abandoned Baby', 'Breakaway', 'Ladder Bottom', 'Concealing Baby Swallow',
            'Three Stars In The South', 'Unique 3 River', 'Mat Hold',
            'Rising/Falling Three Methods', 'Stick Sandwich'
        ]
        
        bearish_patterns = [
            # Jednoświecowe spadkowe
            'Hanging Man', 'Shooting Star', 'Gravestone Doji',
            # Dwuświecowe spadkowe
            'Dark Cloud Cover', 'Tweezers Top', 'Two Crows', 'Upside Gap Two Crows',
            'Counterattack',
            # Trzyświecowe spadkowe
            'Evening Star', 'Evening Doji Star', 'Three Black Crows',
            'Identical Three Crows', 'Tristar',
            # Wieloświecowe spadkowe
            'Advance Block', 'Stalled Pattern'
        ]
        
        for pattern_name, values in self.patterns.items():
            indices = np.where(values != 0)[0]
            
            for idx in indices:
                signal_value = values[idx]
                
                # Określ typ sygnału
                if signal_value > 0:
                    trend = 'Wzrostowa'
                    pattern_type = 'Odwrócenie wzrostowe' if any(p in pattern_name for p in bullish_patterns) else 'Kontynuacja wzrostowa'
                else:
                    trend = 'Spadkowa'
                    pattern_type = 'Odwrócenie spadkowe' if any(p in pattern_name for p in bearish_patterns) else 'Kontynuacja spadkowa'
                
                interpretations.append({
                    'index': int(idx),
                    'date': str(self.data.index[idx]) if hasattr(self.data.index, '__getitem__') else f"Index {idx}",
                    'pattern': pattern_name,
                    'trend': trend,
                    'type': pattern_type,
                    'signal': int(signal_value)
                })
        
        return sorted(interpretations, key=lambda x: x['index'])
    
    def calculate_support_resistance(self, window: int = 20) -> List[Tuple[float, str]]:
        """Oblicza poziomy wsparcia i oporu"""
        if self.data is None:
            return []
        
        levels = []
        highs = self.data['high'].values
        lows = self.data['low'].values
        
        # Znajdź lokalne maksima (opór)
        for i in range(window, len(highs) - window):
            if highs[i] == max(highs[i-window:i+window]):
                levels.append((float(highs[i]), 'Opór'))
        
        # Znajdź lokalne minima (wsparcie)
        for i in range(window, len(lows) - window):
            if lows[i] == min(lows[i-window:i+window]):
                levels.append((float(lows[i]), 'Wsparcie'))
        
        self.support_resistance_levels = levels
        return levels
    
    def verify_pattern_effectiveness(self, lookback: int = 5) -> List[Dict]:
        """Weryfikuje skuteczność formacji"""
        if self.data is None or not self.patterns:
            return []
        
        effectiveness = []
        interpretations = self.interpret_patterns()
        
        for interp in interpretations:
            idx = interp['index']
            
            # Sprawdź czy jest wystarczająco dużo danych do analizy
            if idx + lookback >= len(self.data):
                continue
            
            current_close = self.data['close'].iloc[idx]
            future_close = self.data['close'].iloc[idx + lookback]
            
            price_change = ((future_close - current_close) / current_close) * 100
            
            # Sprawdź czy formacja jest w pobliżu wsparcia/oporu
            near_support_resistance = False
            for level, level_type in self.support_resistance_levels:
                if abs(current_close - level) / current_close < 0.02:  # 2% tolerance
                    near_support_resistance = True
                    break
            
            # Oceń skuteczność
            expected_direction = 'up' if interp['signal'] > 0 else 'down'
            actual_direction = 'up' if price_change > 0 else 'down'
            
            effective = (expected_direction == actual_direction)
            
            effectiveness.append({
                'pattern': interp['pattern'],
                'date': interp['date'],
                'expected': expected_direction,
                'actual_change': round(price_change, 2),
                'effective': effective,
                'near_key_level': near_support_resistance,
                'reliability': 'Wysoka' if (effective and near_support_resistance) else 'Średnia' if effective else 'Niska'
            })
        
        return effectiveness
    
    def generate_chart(self, chart_type: str = 'candlestick') -> str:
        """Generuje wykres świecowy z oznaczonymi formacjami"""
        # Wyczyść poprzednie wykresy
        self._cleanup_matplotlib()

        if self.data is None or len(self.data) == 0:
            print("Brak danych do wygenerowania wykresu")
            return ""
        
        try:
            # Dostosuj liczbę świec do dostępnych danych
            available_candles = len(self.data)
            
            # Dla krótszych okresów pokaż wszystkie dane
            if available_candles <= 100:
                max_candles = available_candles
            else:
                max_candles = min(200, available_candles)
            
            print(f"Generowanie wykresu dla {max_candles} świec (dostępne: {available_candles})")
            
            # Przygotuj dane
            plot_data = self.data.tail(max_candles)[['open', 'high', 'low', 'close']].copy()
            
            # KRYTYCZNE: Upewnij się, że indeks jest DatetimeIndex
            if not isinstance(plot_data.index, pd.DatetimeIndex):
                print("⚠ Indeks nie jest DatetimeIndex - próbuję konwersji...")
                try:
                    plot_data.index = pd.to_datetime(plot_data.index)
                    print("✓ Indeks przekonwertowany na DatetimeIndex")
                except:
                    print("❌ Nie udało się przekonwertować indeksu na DatetimeIndex")
                    # Utwórz sztuczny indeks dat
                    plot_data.index = pd.date_range(
                        start='2024-01-01', 
                        periods=len(plot_data), 
                        freq='D'
                    )
                    print("✓ Utworzono sztuczny DatetimeIndex")
            
            # Weryfikacja typu indeksu
            print(f"Typ indeksu: {type(plot_data.index)}")
            print(f"Zakres dat w wykresie: {plot_data.index[0]} do {plot_data.index[-1]}")
            
            # Sprawdź czy są jakiekolwiek dane
            if plot_data.empty or len(plot_data) < 2:
                print("Za mało danych do wygenerowania wykresu (minimum 2 świece)")
                return ""
            
            # Dodaj volume jeśli dostępne
            show_volume = False
            if 'volume' in self.data.columns:
                volume_data = self.data['volume'].tail(max_candles)
                if not volume_data.isna().all() and (volume_data > 0).any():
                    plot_data['volume'] = volume_data
                    show_volume = True
            
            # Przygotuj listy dla markerów
            apds = []
            
            # Zbierz formacje tylko jeśli istnieją
            if self.patterns:
                bullish_prices = []
                bearish_prices = []
                
                # Iteruj przez wszystkie formacje
                for pattern_name, values in self.patterns.items():
                    # Weź tylko ostatnie wartości odpowiadające zakresowi wykresu
                    pattern_subset = values[-max_candles:] if len(values) > max_candles else values
                    
                    # Dopasuj długość
                    if len(pattern_subset) < len(plot_data):
                        # Dodaj NaN na początku
                        pattern_subset = np.concatenate([
                            np.full(len(plot_data) - len(pattern_subset), 0),
                            pattern_subset
                        ])
                    elif len(pattern_subset) > len(plot_data):
                        pattern_subset = pattern_subset[-len(plot_data):]
                    
                    # Znajdź indeksy formacji
                    indices = np.where(pattern_subset != 0)[0]
                    
                    for idx in indices:
                        if idx < len(plot_data):
                            signal = pattern_subset[idx]
                            try:
                                if signal > 0:  # Wzrostowa
                                    marker_price = float(plot_data['low'].iloc[idx] * 0.995)
                                    bullish_prices.append((idx, marker_price))
                                else:  # Spadkowa
                                    marker_price = float(plot_data['high'].iloc[idx] * 1.005)
                                    bearish_prices.append((idx, marker_price))
                            except Exception as e:
                                print(f"Błąd przy dodawaniu markera: {e}")
                                continue
                
                # Utwórz serie dla markerów
                if bullish_prices:
                    bullish_series = pd.Series(np.nan, index=plot_data.index)
                    for idx, price in bullish_prices:
                        bullish_series.iloc[idx] = price
                    
                    # Dodaj tylko jeśli są niepuste wartości
                    if not bullish_series.isna().all():
                        apds.append(mpf.make_addplot(
                            bullish_series,
                            type='scatter',
                            markersize=80,
                            marker='^',
                            color='green',
                            alpha=0.7,
                            panel=0
                        ))
                
                if bearish_prices:
                    bearish_series = pd.Series(np.nan, index=plot_data.index)
                    for idx, price in bearish_prices:
                        bearish_series.iloc[idx] = price
                    
                    if not bearish_series.isna().all():
                        apds.append(mpf.make_addplot(
                            bearish_series,
                            type='scatter',
                            markersize=80,
                            marker='v',
                            color='red',
                            alpha=0.7,
                            panel=0
                        ))
            
            # Przygotuj linie wsparcia/oporu
            hlines_dict = None
            if self.support_resistance_levels:
                current_price = float(plot_data['close'].iloc[-1])
                price_range = float(plot_data['high'].max() - plot_data['low'].min())
                
                # Filtruj poziomy w zakresie +/- 50% od zakresu cen
                relevant_levels = [
                    (level, typ) for level, typ in self.support_resistance_levels
                    if abs(level - current_price) <= price_range * 0.5
                ]
                
                # Ogranicz do 8 poziomów
                relevant_levels = sorted(
                    relevant_levels,
                    key=lambda x: abs(x[0] - current_price)
                )[:8]
                
                if relevant_levels:
                    support_lines = [level for level, typ in relevant_levels if typ == 'Wsparcie']
                    resistance_lines = [level for level, typ in relevant_levels if typ == 'Opór']
                    
                    all_lines = support_lines + resistance_lines
                    colors = ['green'] * len(support_lines) + ['red'] * len(resistance_lines)
                    
                    if all_lines:
                        hlines_dict = dict(
                            hlines=all_lines,
                            colors=colors,
                            alpha=0.3,
                            linestyle='--',
                            linewidths=1
                        )
            
            # Konfiguracja stylu
            mc = mpf.make_marketcolors(
                up='#26a69a',
                down='#ef5350',
                edge='inherit',
                wick='inherit',
                volume='in',
                alpha=0.9
            )
            
            style = mpf.make_mpf_style(
                base_mpf_style='charles',
                marketcolors=mc,
                gridcolor='#e0e0e0',
                gridstyle=':',
                y_on_right=False,
                rc={
                    'font.size': 10,
                    'axes.labelsize': 11,
                    'axes.titlesize': 13,
                    'xtick.labelsize': 9,
                    'ytick.labelsize': 10,
                    'figure.facecolor': 'white',
                    'axes.facecolor': 'white'
                }
            )
            
            # Dostosuj rozmiar wykresu do ilości danych
            if max_candles <= 50:
                figsize = (12, 7)
            elif max_candles <= 100:
                figsize = (14, 8)
            else:
                figsize = (16, 9)
            
            # Generuj wykres
            fig, axes = mpf.plot(
                plot_data,
                type='candle',
                style=style,
                title=f'Wykres świecowy z wykrytymi formacjami ({max_candles} świec)',
                ylabel='Cena',
                ylabel_lower='Wolumen' if show_volume else '',
                volume=show_volume,
                figsize=figsize,
                datetime_format='%Y-%m-%d',
                xrotation=15,
                tight_layout=True,
                returnfig=True,
                addplot=apds if apds else None,
                hlines=hlines_dict,
                warn_too_much_data=max_candles + 1000
            )
            
            # Formatowanie osi - NIE używaj mdates jeśli mplfinance już ustawił daty
            if axes:
                ax_main = axes[0]
                
                # Sprawdź czy oś X jest poprawnie ustawiona
                xlim = ax_main.get_xlim()
                print(f"Zakres osi X: {xlim}")
                
                # NIE formatuj ponownie - mplfinance już to zrobił
                # Dodaj tylko legendę i grid
                
                # Legenda
                legend_elements = []
                
                if apds:  # Tylko jeśli są markery
                    legend_elements.extend([
                        Patch(facecolor='green', alpha=0.7, label='Formacje wzrostowe'),
                        Patch(facecolor='red', alpha=0.7, label='Formacje spadkowe')
                    ])
                
                if hlines_dict:
                    legend_elements.extend([
                        Patch(facecolor='green', alpha=0.3, label='Wsparcie'),
                        Patch(facecolor='red', alpha=0.3, label='Opór')
                    ])
                
                if legend_elements:
                    ax_main.legend(handles=legend_elements, loc='upper left', fontsize=9)
                
                # Grid
                ax_main.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
            
            # Konwertuj do base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=120, facecolor='white')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode()
            plt.close(fig)

            plt.close('all')
            buffer.close()
            
            print("✓ Wykres wygenerowany pomyślnie")

            
            return f"data:image/png;base64,{image_base64}"
            
        except Exception as e:
            print(f"Błąd generowania wykresu: {e}")
            import traceback
            traceback.print_exc()
            
            # Spróbuj wygenerować prosty wykres bez dodatków
            try:
                print("Próba wygenerowania prostego wykresu bez markerów...")
                simple_data = self.data.tail(min(100, len(self.data)))[['open', 'high', 'low', 'close']].copy()
                
                # Upewnij się, że indeks jest DatetimeIndex
                if not isinstance(simple_data.index, pd.DatetimeIndex):
                    simple_data.index = pd.to_datetime(simple_data.index)
                
                fig, axes = mpf.plot(
                    simple_data,
                    type='candle',
                    style='charles',
                    title='Wykres świecowy (tryb awaryjny)',
                    figsize=(12, 6),
                    returnfig=True
                )
                
                buffer = BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.read()).decode()
                plt.close(fig)

                plt.close('all')
                buffer.close()
                
                print("✓ Wygenerowano prosty wykres")
                return f"data:image/png;base64,{image_base64}"
                
            except Exception as e2:
                print(f"Błąd przy generowaniu prostego wykresu: {e2}")
                return ""

    def generate_interactive_chart(self, max_candles: int = 1000) -> str:
        """Generuje interaktywny wykres Plotly z tooltipami"""
        # Wyczyść poprzednie wykresy
        self._cleanup_matplotlib()

        if self.data is None or len(self.data) == 0:
            print("Brak danych do wygenerowania wykresu")
            return ""
        
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Przygotuj dane
            available_candles = len(self.data)
            if max_candles is None:
                # Automatyczny dobór w zależności od ilości danych
                if available_candles <= 500:
                    # Do 500 świec - pokaż wszystkie
                    max_candles = available_candles
                    print(f"Wykres interaktywny: używam wszystkich {available_candles} świec")
                elif available_candles <= 2000:
                    # 500-2000 świec - pokaż wszystkie (Plotly radzi sobie dobrze)
                    max_candles = available_candles
                    print(f"Wykres interaktywny: używam wszystkich {available_candles} świec")
                else:
                    # Powyżej 2000 - pokaż ostatnie 2000 (dla wydajności)
                    max_candles = 2000
                    print(f"Wykres interaktywny: ograniczam do {max_candles} z {available_candles} świec (ostatnie 2000)")
            else:
                max_candles = min(max_candles, available_candles)
                print(f"Wykres interaktywny: używam {max_candles} świec")
            plot_data = self.data.tail(max_candles).copy()
            
            # Upewnij się, że indeks jest DatetimeIndex
            if not isinstance(plot_data.index, pd.DatetimeIndex):
                try:
                    plot_data.index = pd.to_datetime(plot_data.index)
                except:
                    plot_data.index = pd.date_range(
                        start='2024-01-01', 
                        periods=len(plot_data), 
                        freq='D'
                    )
            
            # Utwórz subplot (z opcjonalnym wolumenem)
            show_volume = 'volume' in plot_data.columns and not plot_data['volume'].isna().all()
            
            if show_volume:
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.03,
                    row_heights=[0.7, 0.3],
                    subplot_titles=('Wykres świecowy', 'Wolumen')
                )
            else:
                fig = go.Figure()
            
            # Dodaj świeczki
            candlestick = go.Candlestick(
                x=plot_data.index,
                open=plot_data['open'],
                high=plot_data['high'],
                low=plot_data['low'],
                close=plot_data['close'],
                name='Cena',
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350',
                hovertext=[
                    f"Data: {date.strftime('%Y-%m-%d')}<br>"
                    f"Open: {o:.2f}<br>"
                    f"High: {h:.2f}<br>"
                    f"Low: {l:.2f}<br>"
                    f"Close: {c:.2f}<br>"
                    f"Zmiana: {((c-o)/o*100):.2f}%"
                    for date, o, h, l, c in zip(
                        plot_data.index,
                        plot_data['open'],
                        plot_data['high'],
                        plot_data['low'],
                        plot_data['close']
                    )
                ],
                hoverinfo='text'
            )
            
            if show_volume:
                fig.add_trace(candlestick, row=1, col=1)
            else:
                fig.add_trace(candlestick)
            
            # Dodaj markery dla formacji
            if self.patterns:
                interpretations = self.interpret_patterns()
                
                # Przygotuj listy dla wzrostowych i spadkowych formacji
                bullish_dates = []
                bullish_prices = []
                bullish_tooltips = []
                
                bearish_dates = []
                bearish_prices = []
                bearish_tooltips = []
                
                # Dopasuj interpretacje do zakresu wykresu
                start_idx = len(self.data) - max_candles
                
                for interp in interpretations:
                    idx = interp['index']
                    
                    # Pomiń formacje sprzed zakresu wykresu
                    if idx < start_idx:
                        continue
                    
                    # Dopasuj indeks do plot_data
                    adjusted_idx = idx - start_idx
                    
                    if adjusted_idx >= len(plot_data):
                        continue
                    
                    date = plot_data.index[adjusted_idx]
                    signal = interp['signal']
                    pattern_name = interp['pattern']
                    trend = interp['trend']
                    pattern_type = interp['type']
                    
                    # Pobierz dane cen
                    open_price = plot_data['open'].iloc[adjusted_idx]
                    high_price = plot_data['high'].iloc[adjusted_idx]
                    low_price = plot_data['low'].iloc[adjusted_idx]
                    close_price = plot_data['close'].iloc[adjusted_idx]
                    
                    # Utwórz tooltip
                    tooltip = (
                        f"<b>{pattern_name}</b><br>"
                        f"Data: {date.strftime('%Y-%m-%d')}<br>"
                        f"Trend: {trend}<br>"
                        f"Typ: {pattern_type}<br>"
                        f"Cena: {close_price:.2f}<br>"
                        f"Range: {low_price:.2f} - {high_price:.2f}"
                    )
                    
                    if signal > 0:  # Wzrostowa
                        bullish_dates.append(date)
                        bullish_prices.append(low_price * 0.995)
                        bullish_tooltips.append(tooltip)
                    else:  # Spadkowa
                        bearish_dates.append(date)
                        bearish_prices.append(high_price * 1.005)
                        bearish_tooltips.append(tooltip)
                
                # Dodaj markery wzrostowe
                if bullish_dates:
                    bullish_marker = go.Scatter(
                        x=bullish_dates,
                        y=bullish_prices,
                        mode='markers',
                        name='Formacje wzrostowe',
                        marker=dict(
                            symbol='triangle-up',
                            size=12,
                            color='green',
                            line=dict(width=1, color='darkgreen')
                        ),
                        text=bullish_tooltips,
                        hoverinfo='text',
                        showlegend=True
                    )
                    
                    if show_volume:
                        fig.add_trace(bullish_marker, row=1, col=1)
                    else:
                        fig.add_trace(bullish_marker)
                
                # Dodaj markery spadkowe
                if bearish_dates:
                    bearish_marker = go.Scatter(
                        x=bearish_dates,
                        y=bearish_prices,
                        mode='markers',
                        name='Formacje spadkowe',
                        marker=dict(
                            symbol='triangle-down',
                            size=12,
                            color='red',
                            line=dict(width=1, color='darkred')
                        ),
                        text=bearish_tooltips,
                        hoverinfo='text',
                        showlegend=True
                    )
                    
                    if show_volume:
                        fig.add_trace(bearish_marker, row=1, col=1)
                    else:
                        fig.add_trace(bearish_marker)
            
            # Dodaj poziomy wsparcia/oporu
            if self.support_resistance_levels:
                current_price = float(plot_data['close'].iloc[-1])
                price_range = float(plot_data['high'].max() - plot_data['low'].min())
                
                # Filtruj poziomy
                relevant_levels = [
                    (level, typ) for level, typ in self.support_resistance_levels
                    if abs(level - current_price) <= price_range * 0.5
                ]
                
                relevant_levels = sorted(
                    relevant_levels,
                    key=lambda x: abs(x[0] - current_price)
                )[:8]
                
                for level, level_type in relevant_levels:
                    color = 'green' if level_type == 'Wsparcie' else 'red'
                    
                    if show_volume:
                        fig.add_hline(
                            y=level,
                            line_dash="dash",
                            line_color=color,
                            opacity=0.3,
                            annotation_text=f"{level_type}: {level:.2f}",
                            annotation_position="right",
                            row=1, col=1
                        )
                    else:
                        fig.add_hline(
                            y=level,
                            line_dash="dash",
                            line_color=color,
                            opacity=0.3,
                            annotation_text=f"{level_type}: {level:.2f}",
                            annotation_position="right"
                        )
            
            # Dodaj wolumen jeśli dostępny
            if show_volume:
                colors = ['green' if close >= open_val else 'red' 
                        for close, open_val in zip(plot_data['close'], plot_data['open'])]
                
                volume_bar = go.Bar(
                    x=plot_data.index,
                    y=plot_data['volume'],
                    name='Wolumen',
                    marker_color=colors,
                    opacity=0.5,
                    hovertemplate='%{x}<br>Wolumen: %{y:,.0f}<extra></extra>'
                )
                
                fig.add_trace(volume_bar, row=2, col=1)
            
            # Konfiguracja layoutu
            fig.update_layout(
                title=f'Interaktywny wykres świecowy - {max_candles} świec ({plot_data.index[0].strftime("%Y-%m-%d")} do {plot_data.index[-1].strftime("%Y-%m-%d")})',
                xaxis_title='Data',
                yaxis_title='Cena',
                template='plotly_white',
                hovermode='x unified',
                height=700 if show_volume else 600,
                xaxis_rangeslider_visible=False,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Włącz zoom i pan
            fig.update_xaxes(
                rangeselector=dict(
                    buttons=list([
                        dict(count=7, label="1T", step="day", stepmode="backward"),
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(step="all", label="Wszystko")
                    ])
                ),
                type="date"
            )
            
            # Zwróć HTML
            return fig.to_html(
                include_plotlyjs='cdn',
                div_id='interactive-chart',
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                    'toImageButtonOptions': {
                        'format': 'png',
                        'filename': 'wykres_formacji',
                        'height': 1000,
                        'width': 1600,
                        'scale': 2
                    }
                }
            )
            
        except Exception as e:
            print(f"Błąd generowania interaktywnego wykresu: {e}")
            import traceback
            traceback.print_exc()
            return ""


    def export_results(self, filepath: str) -> str:
        """Eksportuje wyniki do pliku CSV (najprostsza wersja)"""
        try:
            import os
        
            # Upewnij się, że folder istnieje
            directory = os.path.dirname(filepath)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                print(f"✓ Utworzono folder: {directory}")
            
            # ← DODAJ TEN BLOK - Zmodyfikuj nazwę pliku
            if self.source_filename:
                # Wyodrębnij folder i rozszerzenie
                base_name = os.path.basename(filepath)
                dir_name = os.path.dirname(filepath)
                
                # Usuń "wyniki_analizy" jeśli istnieje
                if base_name.startswith('wyniki_analizy'):
                    # Utwórz nową nazwę z nazwą źródła
                    new_name = f"wyniki_analizy_{self.source_filename}.csv"
                else:
                    # Jeśli inna nazwa, po prostu dodaj źródło
                    name_without_ext = os.path.splitext(base_name)[0]
                    new_name = f"{name_without_ext}_{self.source_filename}.csv"
                
                filepath = os.path.join(dir_name, new_name)
                print(f"✓ Zmieniono nazwę pliku na: {new_name}")
            
            # Przygotuj dane
            interpretations = self.interpret_patterns()
            
            if not interpretations:
                print("⚠ Brak wykrytych formacji do eksportu")
                return None
            
            print(f"Eksportuję {len(interpretations)} formacji...")
            
            # Zawsze zapisuj jako CSV (prostsze i zawsze działa)
            csv_path = filepath.replace('.xlsx', '.csv')
            
            # Utwórz DataFrame
            df = pd.DataFrame(interpretations)
            
            # Zapisz do CSV z odpowiednim kodowaniem
            df.to_csv(csv_path, index=False, encoding='utf-8-sig', sep=';')
            
            # Sprawdź czy plik został utworzony
            if os.path.exists(csv_path):
                file_size = os.path.getsize(csv_path)
                print(f"✓ Plik CSV utworzony: {csv_path}")
                print(f"  Rozmiar: {file_size} bajtów")
                print(f"  Wierszy: {len(df)}")
                return csv_path
            else:
                print(f"❌ Plik nie istnieje: {csv_path}")
                return None
                
        except Exception as e:
            print(f"❌ Błąd eksportu: {e}")
            import traceback
            traceback.print_exc()
            return None


