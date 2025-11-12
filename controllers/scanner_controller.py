# Scanner controller implementation
from flask import render_template, request, jsonify, send_file, redirect, url_for
from werkzeug.utils import secure_filename
import os
from models.candlestick_model import CandlestickModel

class ScannerController:
    """Kontroler obsługujący logikę aplikacji"""
    
    def __init__(self, app):
        self.app = app
        self.model = CandlestickModel()
    
    def index(self):
        """Strona główna"""
        return render_template('index.html')
    
    def upload_page(self):
        """Strona uploadowania danych"""
        return render_template('upload.html')
    
    def upload_file(self):
        """Obsługa uploadu pliku"""
        if 'file' not in request.files:
            return jsonify({'error': 'Brak pliku'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'Nie wybrano pliku'}), 400
        
        if file and self._allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(self.app.config['UPLOAD_FOLDER'], filename)
            
            # Utwórz folder jeśli nie istnieje
            os.makedirs(self.app.config['UPLOAD_FOLDER'], exist_ok=True)
            
            file.save(filepath)
            
            # Wczytaj dane
            if self.model.load_data_from_file(filepath):
                return jsonify({'success': True, 'message': 'Plik wczytany pomyślnie'}), 200
            else:
                return jsonify({'error': 'Błąd wczytywania pliku'}), 400
        
        return jsonify({'error': 'Niedozwolony typ pliku'}), 400
    
    def load_from_ticker(self):
        """Ładowanie danych z Yahoo Finance"""
        data = request.get_json()
        ticker = data.get('ticker', '').upper()
        period = data.get('period', '1y')
        
        if not ticker:
            return jsonify({'error': 'Brak symbolu ticker'}), 400
        
        if self.model.load_data_from_ticker(ticker, period):
            return jsonify({'success': True, 'message': f'Dane dla {ticker} wczytane pomyślnie'}), 200
        else:
            return jsonify({'error': 'Błąd pobierania danych'}), 400
    
    def scan_patterns(self):
        """Skanuje formacje świecowe"""
        if self.model.data is None:
            return jsonify({'error': 'Brak wczytanych danych'}), 400
        
        # Wykryj formacje
        patterns = self.model.detect_patterns()
        
        # Oblicz wsparcie/opór
        self.model.calculate_support_resistance()
        
        # Interpretuj formacje
        interpretations = self.model.interpret_patterns()
        
        # Weryfikuj skuteczność
        effectiveness = self.model.verify_pattern_effectiveness()
        
        # Generuj wykres
        chart_image = self.model.generate_chart()
        
        return jsonify({
            'success': True,
            'patterns_count': len(patterns),
            'interpretations': interpretations,
            'effectiveness': effectiveness,
            'chart': chart_image
        }), 200
    
    def results_page(self):
        """Strona wyników"""
        if self.model.data is None:
            return redirect(url_for('upload'))
        
        # Wykryj formacje jeśli jeszcze nie wykryto
        if not self.model.patterns:
            self.model.detect_patterns()
            self.model.calculate_support_resistance()
        
        interpretations = self.model.interpret_patterns()
        effectiveness = self.model.verify_pattern_effectiveness()
        chart = self.model.generate_chart()
        
        return render_template(
            'results.html',
            interpretations=interpretations,
            effectiveness=effectiveness,
            chart=chart,
            pattern_count=len(self.model.patterns)
        )
    
    def export_results(self):
        """Eksport wyników do pliku"""
        if self.model.data is None:
            return jsonify({'error': 'Brak danych do eksportu'}), 400
        
        export_path = os.path.join(self.app.config['UPLOAD_FOLDER'], 'wyniki_analizy.xlsx')
        
        if self.model.export_results(export_path):
            return send_file(export_path, as_attachment=True, download_name='wyniki_analizy.xlsx')
        else:
            # Fallback do CSV
            export_path = export_path.replace('.xlsx', '.csv')
            return send_file(export_path, as_attachment=True, download_name='wyniki_analizy.csv')
    
    def _allowed_file(self, filename):
        """Sprawdza czy rozszerzenie pliku jest dozwolone"""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in self.app.config['ALLOWED_EXTENSIONS']
