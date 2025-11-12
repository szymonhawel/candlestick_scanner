# Flask application entry point
from flask import Flask
from config import Config
from controllers.scanner_controller import ScannerController

def create_app(config_class=Config):
    """Factory pattern dla aplikacji Flask"""
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Inicjalizacja kontrolera
    controller = ScannerController(app)
    
    # Routing
    @app.route('/')
    def index():
        return controller.index()
    
    @app.route('/upload', methods=['GET'])
    def upload_page():
        return controller.upload_page()
    
    @app.route('/upload', methods=['POST'])
    def upload_file():
        return controller.upload_file()
    
    @app.route('/api/load-ticker', methods=['POST'])
    def load_ticker():
        return controller.load_from_ticker()
    
    @app.route('/api/scan', methods=['POST'])
    def scan():
        return controller.scan_patterns()
    
    @app.route('/results')
    def results():
        return controller.results_page()
    
    @app.route('/export')
    def export():
        return controller.export_results()
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
