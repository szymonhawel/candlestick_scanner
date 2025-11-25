# Candlestick model implementation
import talib
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Tuple
import mplfinance as mpf
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import matplotlib.dates as mdates
from matplotlib.patches import Patch

class CandlestickModel:
    """Model odpowiedzialny za analizę formacji świecowych"""
    
    def __init__(self):
        self.data = None
        self.patterns = {}
        self.support_resistance_levels = []
    
    def load_data_from_file(self, filepath: str) -> bool:
        """Wczytuje dane z pliku CSV z elastycznym rozpoznawaniem kolumn"""
        try:
            # Wczytaj dane
            self.data = pd.read_csv(filepath)
            
            print(f"Oryginalne kolumny: {list(self.data.columns)}")
            
            # Konwertuj nazwy kolumn na małe litery dla porównania
            self.data.columns = self.data.columns.str.strip().str.lower()
            
            # Mapowanie możliwych nazw kolumn (różne warianty)
            column_mapping = {
                'open': ['open', 'open price', 'opening price', 'o', 'opening'],
                'high': ['high', 'high price', 'highest', 'h', 'hi'],
                'low': ['low', 'low price', 'lowest', 'l', 'lo'],
                'close': ['close', 'close price', 'closing price', 'c', 'closing', 'last'],
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
            
            # Konwersja do odpowiednich typów numerycznych
            for col in required_cols:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
            
            # Konwersja volume jeśli istnieje
            if 'volume' in self.data.columns:
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
                    # Próbuj różnych formatów daty
                    self.data[date_col_found] = pd.to_datetime(
                        self.data[date_col_found], 
                        errors='coerce',
                        infer_datetime_format=True
                    )
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
            
            print(f"✓ Wczytano {len(self.data)} wierszy danych")
            print(f"Zakres dat: {self.data.index[0]} do {self.data.index[-1]}")
            
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
            return True
            
        except Exception as e:
            print(f"Błąd pobierania danych: {e}")
            import traceback
            traceback.print_exc()  # Pokaż pełny traceback dla debugowania
            return False

    
    def detect_patterns(self) -> Dict[str, pd.Series]:
        """Wykrywa formacje świecowe przy użyciu TA-Lib"""
        if self.data is None or len(self.data) == 0:
            return {}
        
        open_prices = self.data['open'].values
        high_prices = self.data['high'].values
        low_prices = self.data['low'].values
        close_prices = self.data['close'].values
        
        # Lista wszystkich funkcji rozpoznawania formacji świecowych z TA-Lib
        pattern_functions = [
            # Formacje jednoświecowe
            ('CDLDOJI', 'Doji'),
            ('CDLHAMMER', 'Hammer'),
            ('CDLINVERTEDHAMMER', 'Inverted Hammer'),
            ('CDLHANGINGMAN', 'Hanging Man'),
            ('CDLSHOOTINGSTAR', 'Shooting Star'),
            ('CDLDRAGONFLYDOJI', 'Dragonfly Doji'),
            ('CDLGRAVESTONEDOJI', 'Gravestone Doji'),
            ('CDLMARUBOZU', 'Marubozu'),
            ('CDLSPINNINGTOP', 'Spinning Top'),
            
            # Formacje dwuświecowe
            ('CDLENGULFING', 'Engulfing'),
            ('CDLHARAMI', 'Harami'),
            ('CDLHARAMICROSS', 'Harami Cross'),
            ('CDLPIERCING', 'Piercing'),
            ('CDLDARKCLOUDCOVER', 'Dark Cloud Cover'),
            ('CDLTWEEZERSBOTTOM', 'Tweezers Bottom'),
            ('CDLTWEEZERSBOTTOM', 'Tweezers Top'),
            
            # Formacje trzyświecowe
            ('CDLMORNINGSTAR', 'Morning Star'),
            ('CDLEVENINGSTAR', 'Evening Star'),
            ('CDL3WHITESOLDIERS', 'Three White Soldiers'),
            ('CDL3BLACKCROWS', 'Three Black Crows'),
            ('CDL3INSIDE', 'Three Inside Up/Down'),
            ('CDL3OUTSIDE', 'Three Outside Up/Down'),
            ('CDLABANDONEDBABY', 'Abandoned Baby'),
            ('CDLTRISTAR', 'Tristar'),
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
        return detected_patterns
    
    def interpret_patterns(self) -> List[Dict]:
        """Interpretuje wykryte formacje jako wzrostowe/spadkowe"""
        interpretations = []
        
        # Słownik interpretacji formacji
        bullish_patterns = [
            'Hammer', 'Inverted Hammer', 'Dragonfly Doji', 'Morning Star',
            'Three White Soldiers', 'Piercing', 'Bullish Engulfing'
        ]
        
        bearish_patterns = [
            'Hanging Man', 'Shooting Star', 'Gravestone Doji', 'Evening Star',
            'Three Black Crows', 'Dark Cloud Cover', 'Bearish Engulfing'
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
                
                # Filtruj poziomy w zakresie +/- 20% od obecnej ceny
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
                datetime_format='%Y-%m-%d' if max_candles > 60 else '%Y-%m-%d',
                xrotation=15,
                tight_layout=True,
                returnfig=True,
                addplot=apds if apds else None,
                hlines=hlines_dict,
                warn_too_much_data=max_candles + 1000
            )
            
            # Formatowanie osi
            if axes:
                ax_main = axes[0]
                
                # Dostosuj formatowanie dat do ilości danych
                import matplotlib.dates as mdates
                if max_candles <= 30:
                    ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                    ax_main.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, max_candles // 10)))
                elif max_candles <= 90:
                    ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                    ax_main.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
                else:
                    ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                    ax_main.xaxis.set_major_locator(mdates.MonthLocator())
                
                # Legenda
                from matplotlib.patches import Patch
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
            
            print("✓ Wykres wygenerowany pomyślnie")
            return f"data:image/png;base64,{image_base64}"
            
        except Exception as e:
            print(f"Błąd generowania wykresu: {e}")
            import traceback
            traceback.print_exc()
            
            # Spróbuj wygenerować prosty wykres bez dodatków
            try:
                print("Próba wygenerowania prostego wykresu bez markerów...")
                simple_data = self.data.tail(min(100, len(self.data)))[['open', 'high', 'low', 'close']]
                
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
                
                print("✓ Wygenerowano prosty wykres")
                return f"data:image/png;base64,{image_base64}"
                
            except Exception as e2:
                print(f"Błąd przy generowaniu prostego wykresu: {e2}")
                return ""


    def generate_interactive_chart(self) -> str:
        """Generuje interaktywny wykres używając Plotly"""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        if self.data is None:
            return ""
        
        max_candles = min(200, len(self.data))
        plot_data = self.data.tail(max_candles)
        
        # Utwórz wykres świecowy
        fig = go.Figure(data=[go.Candlestick(
            x=plot_data.index,
            open=plot_data['open'],
            high=plot_data['high'],
            low=plot_data['low'],
            close=plot_data['close'],
            name='OHLC'
        )])
        
        # Dodaj markery formacji
        for pattern_name, values in self.patterns.items():
            pattern_subset = values[-max_candles:]
            indices = np.where(pattern_subset != 0)[0]
            
            for idx in indices:
                if idx < len(plot_data):
                    signal = pattern_subset[idx]
                    fig.add_annotation(
                        x=plot_data.index[idx],
                        y=plot_data['high'].iloc[idx] if signal < 0 else plot_data['low'].iloc[idx],
                        text='▼' if signal < 0 else '▲',
                        showarrow=False,
                        font=dict(size=20, color='red' if signal < 0 else 'green')
                    )
        
        fig.update_layout(
            title='Wykres świecowy z formacjami (interaktywny)',
            yaxis_title='Cena',
            xaxis_title='Data',
            height=700,
            template='plotly_white'
        )
        
        return fig.to_html(include_plotlyjs='cdn', div_id='candlestick-chart')


    def export_results(self, filepath: str) -> bool:
        """Eksportuje wyniki do pliku CSV"""
        try:
            interpretations = self.interpret_patterns()
            effectiveness = self.verify_pattern_effectiveness()
            
            # Połącz dane
            df_interp = pd.DataFrame(interpretations)
            df_effect = pd.DataFrame(effectiveness)
            
            # Zapisz do pliku
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                df_interp.to_excel(writer, sheet_name='Formacje', index=False)
                df_effect.to_excel(writer, sheet_name='Skuteczność', index=False)
                
                # Eksportuj poziomy wsparcia/oporu
                df_levels = pd.DataFrame(self.support_resistance_levels, columns=['Poziom', 'Typ'])
                df_levels.to_excel(writer, sheet_name='Wsparcie_Opór', index=False)
            
            return True
        except Exception as e:
            print(f"Błąd eksportu: {e}")
            # Fallback do CSV
            try:
                df_interp = pd.DataFrame(self.interpret_patterns())
                df_interp.to_csv(filepath.replace('.xlsx', '.csv'), index=False)
                return True
            except:
                return False
