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
        """Wczytuje dane z pliku CSV"""
        try:
            self.data = pd.read_csv(filepath)
            self.data.columns = [col.lower() for col in self.data.columns]
            
            # Walidacja kolumn
            required_cols = ['open', 'high', 'low', 'close']
            if not all(col in self.data.columns for col in required_cols):
                return False
            
            # Konwersja do odpowiednich typów
            for col in required_cols:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
            
            if 'date' in self.data.columns:
                self.data['date'] = pd.to_datetime(self.data['date'])
                self.data.set_index('date', inplace=True)
            
            return True
        except Exception as e:
            print(f"Błąd wczytywania danych: {e}")
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
            return ""
        
        try:
            # Przygotuj dane dla mplfinance (ostatnie 100-200 świec dla czytelności)
            max_candles = min(200, len(self.data))
            plot_data = self.data.tail(max_candles)[['open', 'high', 'low', 'close']].copy()
            
            # Dodaj volume jeśli dostępne
            if 'volume' in self.data.columns:
                plot_data['volume'] = self.data['volume'].tail(max_candles)
                show_volume = True
            else:
                show_volume = False
            
            # Przygotuj markery dla formacji świecowych
            apds = []  # Lista dla addplot
            
            # Zbierz wszystkie formacje w jednym przebiegu
            bullish_markers = []
            bearish_markers = []
            
            for pattern_name, values in self.patterns.items():
                pattern_subset = values[-max_candles:] if len(values) > max_candles else values
                indices = np.where(pattern_subset != 0)[0]
                
                for idx in indices:
                    if idx < len(plot_data):
                        signal = pattern_subset[idx]
                        # Umieść marker poniżej/powyżej świecy z marginesem
                        if signal > 0:  # Wzrostowa
                            marker_price = plot_data['low'].iloc[idx] * 0.995
                            bullish_markers.append(marker_price)
                        else:  # Spadkowa
                            marker_price = plot_data['high'].iloc[idx] * 1.005
                            bearish_markers.append(marker_price)
                    else:
                        bullish_markers.append(np.nan)
                        bearish_markers.append(np.nan)
            
            # Wyrównaj długość do plot_data
            while len(bullish_markers) < len(plot_data):
                bullish_markers.insert(0, np.nan)
            while len(bearish_markers) < len(plot_data):
                bearish_markers.insert(0, np.nan)
            
            bullish_markers = bullish_markers[:len(plot_data)]
            bearish_markers = bearish_markers[:len(plot_data)]
            
            # Dodaj markery jako scatter plot
            if any(~np.isnan(bullish_markers)):
                apds.append(mpf.make_addplot(
                    bullish_markers,
                    type='scatter',
                    markersize=80,
                    marker='^',
                    color='green',
                    alpha=0.7,
                    panel=0
                ))
            
            if any(~np.isnan(bearish_markers)):
                apds.append(mpf.make_addplot(
                    bearish_markers,
                    type='scatter',
                    markersize=80,
                    marker='v',
                    color='red',
                    alpha=0.7,
                    panel=0
                ))
            
            # Dodaj linie wsparcia/oporu
            hlines_dict = None
            if self.support_resistance_levels:
                # Ogranicz do 8 najbardziej znaczących poziomów
                levels_sorted = sorted(self.support_resistance_levels, 
                                    key=lambda x: abs(x[0] - plot_data['close'].iloc[-1]))[:8]
                
                support_lines = [level for level, typ in levels_sorted if typ == 'Wsparcie']
                resistance_lines = [level for level, typ in levels_sorted if typ == 'Opór']
                
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
                up='#26a69a',      # Zielony (wzrost)
                down='#ef5350',    # Czerwony (spadek)
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
            
            # Generuj wykres
            fig, axes = mpf.plot(
                plot_data,
                type='candle',
                style=style,
                title='Wykres świecowy z wykrytymi formacjami',
                ylabel='Cena',
                ylabel_lower='Wolumen' if show_volume else '',
                volume=show_volume,
                figsize=(14, 8),
                datetime_format='%Y-%m-%d',
                xrotation=15,
                tight_layout=True,
                returnfig=True,
                addplot=apds if apds else None,
                hlines=hlines_dict,
                warn_too_much_data=len(plot_data) + 100  # Zwiększ limit ostrzeżeń
            )
            
            # Dostosuj formatowanie osi X
            if axes:
                ax_main = axes[0]
                
                # Lepsze formatowanie dat
                import matplotlib.dates as mdates
                ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax_main.xaxis.set_major_locator(mdates.AutoDateLocator())
                
                # Dodaj legendę dla markerów
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='green', alpha=0.7, label='Formacje wzrostowe'),
                    Patch(facecolor='red', alpha=0.7, label='Formacje spadkowe')
                ]
                
                if self.support_resistance_levels:
                    legend_elements.extend([
                        Patch(facecolor='green', alpha=0.3, label='Wsparcie'),
                        Patch(facecolor='red', alpha=0.3, label='Opór')
                    ])
                
                ax_main.legend(handles=legend_elements, loc='upper left', fontsize=9)
                
                # Dodaj grid dla lepszej czytelności
                ax_main.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
            
            # Konwertuj do base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=120, facecolor='white')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode()
            plt.close(fig)
            
            return f"data:image/png;base64,{image_base64}"
            
        except Exception as e:
            print(f"Błąd generowania wykresu: {e}")
            import traceback
            traceback.print_exc()
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
