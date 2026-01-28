# Skaner formacji świecowych

Aplikacja webowa do analizy formacji świecowych (candlestick patterns) na wykresach giełdowych. System umożliwia wczytywanie danych historycznych, identyfikację wzorców technicznych oraz wizualizację wyników analizy.

## Opis projektu

Candlestick Scanner to narzędzie wspomagające analizy techniczne instrumentów finansowych poprzez automatyczne rozpoznawanie formacji świecowych. Aplikacja bazuje na bibliotece TA-Lib do wykrywania wzorców oraz Flask jako framework webowy.

## Funkcjonalności

- Wczytywanie danych giełdowych z różnych źródeł (upload pliku CSV, ticker)
- Automatyczne rozpoznawanie formacji świecowych
- Wizualizacja wyników analizy za pomocą interaktywnych wykresów
- Eksport wyników do plików
- Responsywny interfejs webowy

## Wymagania systemowe

- Python 3.8 lub nowszy
- System operacyjny: Windows, Linux lub macOS
- Zależności w pliku requirements.txt

## Struktura projektu
```
candlestick_scanner/
├── app.py                    # Punkt wejścia aplikacji
├── config.py                 # Konfiguracja aplikacji
├── requirements.txt          # Zależności Pythona
├── controllers/              # Kontrolery (logika biznesowa)
│   └── scanner_controller.py
├── models/                   # Modele danych i logika rozpoznawania wzorców
│   └── candlestick_model.py
├── templates/                # Szablony HTML (widoki)
├── static/                   # Pliki statyczne (CSS, JS, obrazy)
├── data/                     # Dane wejściowe
└── tests/                    # Testy jednostkowe
```

## Instalacja
1. Sklonuj repozytorium
```
git clone https://github.com/szymonhawel/candlestick_scanner.git
cd candlestick_scanner
```
2. Zainstaluj zależności
```
pip install -r requirements.txt
```
3. Uruchom aplikację
```
python app.py
```
Aplikacja po uruchomieniu jest dostępna pod adresem http://localhost:5000

## Sposób obsługi
1. Otwórz aplikację w przeglądarce.
2. Wybierz metodę wczytania danych: wczytywanie z pliku CSV lub pobranie danych z Yahoo Finance.
3. Kliknij "Skanuj formacje świecowe".
4. Przejrzyj wykryte formacje świecowe wraz z wizualizacją.
5. Opcjonalnie wyeksportuj wyniki.

## Autor
Szymon Hawel

## Licencja
MIT
