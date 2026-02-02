# 🎬 YT Idea Evaluator Pro v4

## Kompletna aplikacja do oceny pomysłów na filmy YouTube

### ✨ 25 Funkcji

#### 🔧 UI/UX (1-13)
| # | Funkcja | Opis |
|---|---------|------|
| 1 | YouTube API Sync | Auto-pobieranie danych z YT API |
| 2 | Pamięć lokalna | Historia ocen zapisywana na dysku |
| 3 | Dark mode fix | Poprawione kolory dla dark mode |
| 4 | Zapamiętaj API key | Klucz zapisywany lokalnie |
| 5 | Tooltips ❓ | Wyjaśnienia przy każdej opcji |
| 6 | Rozbudowana historia | Pełne szczegóły każdej oceny |
| 7 | Kopiuj raport | Export oceny do tekstu |
| 8 | Wyjaśnienie bonusów | Tooltips dla Trend/Competition/DNA |
| 9 | Oceny wariantów | Score przy każdym wariancie tytułu |
| 10 | Wyjaśnienie kary | Szczegóły risk penalty |
| 11 | Wyjaśnienie bonusu | Szczegóły score bonus |
| 12 | LLM tooltip | Co oznacza LLM Score |
| 13 | Data tooltip | Co oznacza Data Score |

#### 📊 Analizy (14-19)
| # | Funkcja | Opis |
|---|---------|------|
| 14 | Porównanie pomysłów | Ranking 3-5 pomysłów |
| 15 | Tracking accuracy | Weryfikacja predykcji po publikacji |
| 16 | Promise Generator | AI generuje propozycje obietnic |
| 17 | Dashboard kanału | Wykresy, statystyki, DNA |
| 18 | Trend alerting | Monitorowanie trendów |
| 19 | Analiza serii | Które serie działają najlepiej |

#### 🚀 Zaawansowane (20-25)
| # | Funkcja | Opis |
|---|---------|------|
| 20 | A/B Title Tester | Porównanie 2 tytułów |
| 21 | Audience Overlap | Analiza konkurencji |
| 22 | Optimal Calendar | Kiedy publikować |
| 23 | "Dlaczego wtopa" | Analiza nieudanych filmów |
| 24 | Content Gap Finder | Tematy których nie robiłeś |
| 25 | Idea Vault | Zapisz pomysły na później |

---

## 🚀 Szybki start

### Checklist (2 min)
1. **Zainstaluj zależności** (`pip install -r requirements.txt`)
2. **Uruchom aplikację** (`streamlit run app.py`)
3. **Dodaj OpenAI API key** w panelu bocznym i kliknij **Zapisz klucz**
4. **Załaduj dane kanału** (CSV) albo użyj **YouTube Sync**

### Windows
```batch
# Kliknij dwukrotnie:
start.bat
```

### Linux/Mac
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Pierwszy raz
1. Uruchom aplikację
2. Wpisz OpenAI API key w panelu bocznym
3. Kliknij "Zapisz klucz"
4. Wgraj dane kanału (CSV) lub użyj YouTube Sync

---


---

## 🔑 Co musisz dodać (raz) żeby wszystko działało

### 1) OpenAI API Key (obowiązkowe dla LLM)
- Wpisujesz w sidebarze i klikasz **Zapisz klucz**
- Aplikacja zapisuje go lokalnie w `app_data/config.json`

### 2) Dane kanału (obowiązkowe)
Masz 2 opcje:
- **CSV**: wrzuć plik z kolumnami `title, views` (minimum)
- **YouTube Sync (właściciel kanału)**: wymaga Google OAuth

### 3) YouTube Sync (opcjonalne, ale polecam)
Żeby pobierać dane prosto z API:
- Utwórz projekt w Google Cloud i włącz **YouTube Data API v3**
- Pobierz OAuth client i zapisz jako `client_secret.json` w katalogu głównym aplikacji
- Alternatywnie: ustaw `youtube_api_key` w `app_data/config.json` (do prostych zapytań, bez metryk właścicielskich)

### 4) Lista konkurencji i słowa niszy (opcjonalne, ale daje duży boost)
- Zakładka **Narzędzia -> Trendy/Konkurencja** pozwala dodać kanały i keywords
- Zapis jest lokalny w `app_data/`


## 📁 Struktura plików

```
yt_evaluator_v3/
├── app.py                    # Główna aplikacja Streamlit
├── yt_idea_evaluator_pro_v2.py  # Core evaluator (ML + LLM)
├── advanced_analytics.py     # Zaawansowane analizy
├── config_manager.py         # Zarządzanie konfiguracją
├── youtube_sync.py           # YouTube API sync
├── requirements.txt          # Zależności Python
├── start.bat                 # Launcher Windows
├── README.md                 # Ten plik
│
├── channel_data/             # Dane kanału (auto-tworzone)
│   ├── merged_channel_data.csv
│   └── synced_channel_data.csv
│
└── app_data/                 # Dane aplikacji (auto-tworzone)
    ├── config.json           # Ustawienia
    ├── evaluation_history.json
    ├── idea_vault.json
    └── trend_alerts.json
```

---

## 📊 Format danych CSV

### Wymagane kolumny:
```
title, views
```

### Zalecane kolumny:
```
title, views, retention, label, published_at
```

### Przykład:
```csv
title,views,retention,label,published_at
"Dlaczego ta katastrofa musiała się wydarzyć?",150000,42.5,PASS,2024-01-15
"Tajemnica która wstrząsnęła Polską",85000,38.2,PASS,2024-02-20
"Co naprawdę się stało?",12000,22.1,FAIL,2024-03-10
```

### Labels:
- `PASS` - hit (np. views > 50k lub retention > 40%)
- `FAIL` - wtopa (np. views < 15k i retention < 25%)
- `BORDER` - średniak

---

## 🔑 YouTube API Setup (opcjonalne)

Jeśli chcesz używać automatycznej synchronizacji:

1. **Google Cloud Console**
   - Utwórz projekt na https://console.cloud.google.com/
   - Włącz YouTube Data API v3
   - Włącz YouTube Analytics API

2. **OAuth Credentials**
   - APIs & Services → Credentials
   - Create Credentials → OAuth client ID
   - Typ: Desktop application
   - Pobierz JSON

3. **W aplikacji**
   - Skopiuj JSON do `app_data/youtube_credentials.json`
   - Kliknij "Zaloguj do YouTube"
   - Zaloguj się przez przeglądarkę

---

## 💡 Tips

### Dla najlepszych wyników:
- **Min. 10 filmów** - więcej = lepsze predykcje
- **Dodaj retention** - znacznie poprawia accuracy
- **Używaj labels** - PASS/FAIL pomagają modelowi
- **Regularny tracking** - po publikacji dodaj rzeczywiste views

### Interpretacja wyników:
- **70+** = 🟢 PASS - publikuj śmiało
- **50-69** = 🟡 BORDER - popraw wg sugestii
- **<50** = 🔴 FAIL - przemyśl ponownie

### Bonusy/Kary:
- **Trend bonus**: +10 do -10 (Google Trends)
- **Competition bonus**: +15 do -15 (nasycenie YT)
- **DNA bonus**: 0 do +20 (dopasowanie do Twoich hitów)

---

## 🐛 Troubleshooting

### Aplikacja się nie uruchamia
1. Sprawdź wersję Pythona: `python --version` (zalecane >= 3.10)
2. Upewnij się, że instalujesz zależności w tym samym środowisku, w którym uruchamiasz `streamlit run app.py`

### Brak danych / puste wykresy
- Sprawdź czy CSV ma **kolumny `title` i `views`**
- Upewnij się, że wartości w `views` są liczbami

### "proxies" error
```bash
pip install httpx==0.24.1
```

### Streamlit nie działa
```bash
python -m streamlit run app.py
```

### Brak modułów
```bash
pip install -r requirements.txt --upgrade
```

### API key nie działa
- Sprawdź czy klucz jest poprawny
- Sprawdź czy masz kredyty na koncie OpenAI
- Jeśli używasz Google AI Studio, upewnij się, że klucz jest zapisany w polu Google API key

---

## 📞 Support

Stworzone dla Dawid 🎬

Made with ❤️ by Claude
