# RAPORT AUDYTOWY: YT Idea Evaluator Pro v4

**Data:** 2026-02-02
**Audytor:** Claude
**Wersja aplikacji:** v4

---

## PODSUMOWANIE WYKONAWCZE

Aplikacja YT Idea Evaluator Pro to solidne narzędzie do oceny pomysłów na filmy YouTube. Po przeprowadzeniu szczegółowego audytu ~10,600 linii kodu Python zidentyfikowałem **23 obszary do usprawnienia**, podzielone na kategorie: wydajność, jakość kodu, UX, architektura i bezpieczeństwo.

### Priorytetyzacja (Quick Wins vs Long-term)

| Priorytet | Ilość | Szacowany wpływ |
|-----------|-------|-----------------|
| KRYTYCZNY | 3 | Stabilność aplikacji |
| WYSOKI | 8 | Znacząca poprawa UX/wydajności |
| ŚREDNI | 7 | Jakość kodu i maintainability |
| NISKI | 5 | Nice-to-have |

---

## 1. PROBLEMY KRYTYCZNE

### 1.1 Brak obsługi błędów API w krytycznych ścieżkach
**Lokalizacja:** `app.py:673-680`, `yt_idea_evaluator_pro_v2.py:673-682`

**Problem:** Wywołania API (OpenAI, Google) mogą failować bez graceful degradation. Użytkownik widzi cryptic error zamiast helpful message.

**Rekomendacja:**
```python
# PRZED (app.py:715-723)
response = client.chat.completions.create(...)
if response.choices and response.choices[0].message.content:
    return True, "Połączono z OpenAI"
return False, "Brak odpowiedzi z API"

# PO
try:
    response = client.chat.completions.create(
        ...,
        timeout=10.0  # Dodaj timeout
    )
    if response.choices and response.choices[0].message.content:
        return True, "Połączono z OpenAI"
    return False, "Brak odpowiedzi z API"
except openai.APIConnectionError:
    return False, "Brak połączenia z API. Sprawdź internet."
except openai.RateLimitError:
    return False, "Przekroczono limit zapytań. Poczekaj chwilę."
except openai.AuthenticationError:
    return False, "Nieprawidłowy klucz API."
except Exception as e:
    return False, f"Nieoczekiwany błąd: {type(e).__name__}"
```

### 1.2 Potencjalna utrata danych przy zapisie JSON
**Lokalizacja:** `config_manager.py:130-137`, `config_manager.py:192-199`

**Problem:** Zapis JSON nie jest atomowy. Crash w trakcie zapisu = uszkodzony plik.

**Rekomendacja:**
```python
# Użyj atomic write pattern
def save(self):
    ensure_config_dir()
    temp_file = CONFIG_FILE.with_suffix('.tmp')
    try:
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        temp_file.replace(CONFIG_FILE)  # Atomowa operacja
    except OSError as e:
        temp_file.unlink(missing_ok=True)
        print(f"Nie udało się zapisać config.json: {e}")
```

### 1.3 Memory leak przy cache embeddingów
**Lokalizacja:** `yt_idea_evaluator_pro_v2.py:453-483`

**Problem:** Cache embeddingów rośnie bez limitu. Przy dużej liczbie ocen może zająć całą pamięć.

**Rekomendacja:** Dodaj LRU cache z limitem lub czyszczenie starych wpisów.

---

## 2. PROBLEMY WYSOKIEGO PRIORYTETU

### 2.1 Duplikacja klasy PromiseGenerator
**Lokalizacje:**
- `advanced_analytics.py` (klasa PromiseGenerator - linia ~1500+)
- `topic_analyzer.py:341-498` (klasa PromiseGenerator)

**Problem:** Dwie różne implementacje tej samej funkcjonalności. Może prowadzić do niespójnych wyników.

**Rekomendacja:** Usuń duplikat z `advanced_analytics.py` i importuj z `topic_analyzer.py`.

### 2.2 Monolityczny app.py (~3939 linii)
**Lokalizacja:** `app.py`

**Problem:** Plik jest za duży, trudny do nawigacji i utrzymania.

**Rekomendacja:** Podziel na moduły:
```
ui/
├── __init__.py
├── sidebar.py        # Sekcja sidebar (API keys, YouTube sync)
├── tab_evaluate.py   # Zakładka "Oceń pomysł"
├── tab_tools.py      # Zakładka "Narzędzia"
├── tab_analytics.py  # Zakładka "Analytics"
├── tab_history.py    # Zakładka "Historia"
├── tab_vault.py      # Zakładka "Idea Vault"
├── tab_data.py       # Zakładka "Dane"
├── tab_diagnostics.py# Zakładka "Diagnostyka"
├── components.py     # Współdzielone komponenty (render_verdict_card, etc.)
└── styles.py         # CSS i stałe stylistyczne
```

### 2.3 Powtarzające się wczytywanie danych CSV
**Lokalizacja:** `app.py:463-472`, wielokrotne wywołania `load_merged_data()`

**Problem:** CSV jest wczytywane wielokrotnie w różnych miejscach.

**Rekomendacja:**
```python
# Użyj Streamlit cache z TTL
@st.cache_data(ttl=300)  # 5 minut
def load_merged_data() -> Optional[pd.DataFrame]:
    synced_file = CHANNEL_DATA_DIR / "synced_channel_data.csv"
    if synced_file.exists():
        return pd.read_csv(synced_file)
    if MERGED_DATA_FILE.exists():
        return pd.read_csv(MERGED_DATA_FILE)
    return None
```

### 2.4 Brak warstwy abstrakcji LLM
**Lokalizacja:** `app.py:543-574`, `app.py:692-700`

**Problem:** Kod obsługi OpenAI i Google AI Studio jest rozproszony. Dodanie nowego providera wymaga zmian w wielu miejscach.

**Rekomendacja:** Stwórz wspólny interfejs:
```python
# llm_provider.py
from abc import ABC, abstractmethod

class LLMProvider(ABC):
    @abstractmethod
    def chat_complete(self, messages: List[Dict], **kwargs) -> str:
        pass

    @abstractmethod
    def test_connection(self) -> Tuple[bool, str]:
        pass

class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model: str):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def chat_complete(self, messages, **kwargs):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        return response.choices[0].message.content

class GoogleAIProvider(LLMProvider):
    # ... analogicznie
```

### 2.5 Hardcoded progi i wagi
**Lokalizacja:** `yt_idea_evaluator_pro_v2.py:46-62`, `topic_analyzer.py:181-260`

**Problem:** Progi (THRESHOLD_PASS=68, THRESHOLD_BORDER=52) i wagi są zahardcodowane. Zmiana wymaga edycji kodu.

**Rekomendacja:** Przenieś do `config.json`:
```json
{
  "scoring": {
    "threshold_pass": 68,
    "threshold_border": 52,
    "weight_data": 0.30,
    "weight_metrics": 0.25,
    "weight_llm": 0.45
  },
  "title_scoring": {
    "ideal_length": 52,
    "max_power_words_bonus": 10,
    "caps_penalty": -6
  }
}
```

### 2.6 Brak progress indicators dla długich operacji
**Lokalizacja:** Różne miejsca w `app.py`

**Problem:** Niektóre operacje (np. bulk analytics, trend discovery) trwają długo bez informacji o postępie.

**Rekomendacja:**
```python
# Użyj st.progress dla operacji z wieloma krokami
progress_bar = st.progress(0)
status_text = st.empty()

for i, item in enumerate(items):
    status_text.text(f"Przetwarzam {i+1}/{len(items)}: {item['title'][:30]}...")
    # ... processing
    progress_bar.progress((i + 1) / len(items))

progress_bar.empty()
status_text.empty()
```

### 2.7 Niespójne nazewnictwo kolumn CSV
**Lokalizacja:** `yt_idea_evaluator_pro_v2.py:349-389`, `youtube_sync.py`

**Problem:** Kod obsługuje wiele wariantów nazw kolumn (views, viewCount, views_0_7d, etc.) ale logika jest rozproszona.

**Rekomendacja:** Stwórz funkcję normalizującą na wejściu:
```python
COLUMN_ALIASES = {
    'views': ['views', 'viewCount', 'views_0_7d', 'Views', 'VIEWS'],
    'retention': ['retention', 'avgViewPercentage', 'avgViewPercentage_0_7d', 'Retention'],
    'title': ['title', 'title_api', 'Title', 'TITLE'],
    'published_at': ['published_at', 'publishedAt', 'date', 'published'],
}

def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizuje nazwy kolumn do standardowych."""
    df = df.copy()
    for standard_name, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in df.columns and standard_name not in df.columns:
                df[standard_name] = df[alias]
                break
    return df
```

### 2.8 Session state może być chaotyczny
**Lokalizacja:** `app.py` - różne miejsca

**Problem:** Session state używa wielu kluczy bez spójnej konwencji. Może prowadzić do konfliktów.

**Rekomendacja:** Użyj struktury zagnieżdżonej:
```python
# Zamiast:
st.session_state["topic_job_main"]
st.session_state["topic_result_main"]
st.session_state["last_wtopa"]

# Użyj:
if "app_state" not in st.session_state:
    st.session_state.app_state = {
        "evaluate": {"job": None, "result": None},
        "tools": {"wtopa": None, "calendar": None},
        "diagnostics": {"logs": [], "llm_stats": {}},
    }
```

---

## 3. PROBLEMY ŚREDNIEGO PRIORYTETU

### 3.1 Nadmiarowe try/except z pass
**Lokalizacje:** Wiele miejsc, np. `topic_analyzer.py:62-67`, `advanced_analytics.py:755-763`

**Problem:** Ciche łykanie błędów utrudnia debugowanie.

**Rekomendacja:** Loguj błędy:
```python
import logging
logger = logging.getLogger(__name__)

try:
    # ...
except Exception as e:
    logger.warning(f"Nie udało się {operation}: {e}")
    # fallback behavior
```

### 3.2 Nieoptymalne wyrażenia regularne
**Lokalizacja:** `advanced_analytics.py:47-123`, `topic_analyzer.py:195-248`

**Problem:** Wiele regex jest kompilowanych przy każdym wywołaniu.

**Rekomendacja:** Prekompiluj na poziomie klasy:
```python
class TitleGenerator:
    # Kompiluj raz przy imporcie modułu
    _CONTEXT_NUM_PATTERN = re.compile(r'\d+[\s\.\-]*(min|h|godz|lat|osób|ofiar|dni|ciała|km|mln|tys)', re.I)
    _DATE_YEAR_PATTERN = re.compile(r'(19|20)\d{2}')

    def _score_title(self, title: str) -> Tuple[int, str]:
        if self._CONTEXT_NUM_PATTERN.search(title) or self._DATE_YEAR_PATTERN.search(title):
            score += 10
```

### 3.3 Brak walidacji inputu użytkownika
**Lokalizacja:** `app.py:1452-1458`, `config_manager.py:378-399`

**Problem:** Input od użytkownika nie jest sanityzowany.

**Rekomendacja:**
```python
def sanitize_topic(topic: str) -> str:
    """Czyści i waliduje temat."""
    topic = topic.strip()
    if len(topic) > 500:
        topic = topic[:500]
    # Usuń potencjalnie problematyczne znaki
    topic = re.sub(r'[<>{}|\\^`]', '', topic)
    return topic
```

### 3.4 Brak type hints w wielu miejscach
**Lokalizacja:** Różne funkcje w `app.py`, `config_manager.py`

**Problem:** Utrudnia zrozumienie kodu i IDE autocomplete.

**Rekomendacja:** Dodaj type hints do publicznych funkcji:
```python
# PRZED
def render_verdict_card(result):

# PO
def render_verdict_card(result: Dict[str, Any]) -> None:
```

### 3.5 Zduplikowane definicje stałych
**Lokalizacja:**
- `TOOLTIPS` w `app.py:242-435`
- Podobne opisy w różnych miejscach

**Problem:** Te same teksty są definiowane w wielu miejscach.

**Rekomendacja:** Wydziel do osobnego pliku `constants.py`:
```python
# constants.py
TOOLTIPS = { ... }
RISK_EXPLANATIONS = { ... }
DIMENSION_NAMES_PL = { ... }
```

### 3.6 Nieoptymalne ładowanie modułów
**Lokalizacja:** `app.py:30-84`

**Problem:** Wszystkie moduły są ładowane na starcie, nawet jeśli nie są używane.

**Rekomendacja:** Lazy import dla opcjonalnych modułów:
```python
def get_topic_analyzer():
    """Lazy import topic analyzera."""
    global _topic_analyzer_module
    if '_topic_analyzer_module' not in globals():
        try:
            from topic_analyzer import TopicEvaluator, TitleGenerator
            _topic_analyzer_module = (TopicEvaluator, TitleGenerator)
        except ImportError:
            _topic_analyzer_module = None
    return _topic_analyzer_module
```

### 3.7 Brak testów jednostkowych
**Lokalizacja:** Cały projekt

**Problem:** Brak testów utrudnia refactoring i może prowadzić do regresji.

**Rekomendacja:** Dodaj podstawowe testy:
```python
# tests/test_title_scoring.py
import pytest
from topic_analyzer import TitleGenerator

def test_title_length_scoring():
    gen = TitleGenerator()

    # Idealny tytuł (52 znaki)
    score, _ = gen._score_title("a" * 52)
    assert score >= 50

    # Za krótki
    score, _ = gen._score_title("abc")
    assert score < 50

def test_power_words_bonus():
    gen = TitleGenerator()
    score_with, _ = gen._score_title("Tragedia: ktoś zginął")
    score_without, _ = gen._score_title("Coś się wydarzyło")
    assert score_with > score_without
```

---

## 4. PROBLEMY NISKIEGO PRIORYTETU

### 4.1 CSS inline w app.py
**Lokalizacja:** `app.py:106-235`

**Problem:** Duży blok CSS w kodzie Python.

**Rekomendacja:** Przenieś do osobnego pliku `styles.css` i ładuj:
```python
def load_css():
    css_path = Path(__file__).parent / "styles.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)
```

### 4.2 Nieużywane importy
**Lokalizacja:** Różne pliki

**Problem:** Niektóre importy nie są używane.

**Rekomendacja:** Uruchom `ruff check --select=F401` lub `pylint` i usuń nieużywane importy.

### 4.3 Magiczne liczby w kodzie
**Lokalizacja:** Różne miejsca, np. `topic_analyzer.py:184` (`52` jako idealna długość)

**Problem:** Magiczne liczby bez wyjaśnienia.

**Rekomendacja:** Użyj nazwanych stałych:
```python
IDEAL_TITLE_LENGTH = 52  # Optymalna długość tytułu dla YouTube
MIN_TITLE_LENGTH = 30
MAX_TITLE_LENGTH = 75
```

### 4.4 Brak dokumentacji funkcji
**Lokalizacja:** Wiele funkcji w `app.py`

**Problem:** Brak docstringów w wielu funkcjach.

**Rekomendacja:** Dodaj docstringi do publicznych funkcji.

### 4.5 Nieoptymalne formatowanie stringów
**Lokalizacja:** Różne miejsca

**Problem:** Mieszanka f-strings, .format() i %.

**Rekomendacja:** Ujednolic do f-strings.

---

## 5. REKOMENDACJE ARCHITEKTONICZNE (DŁUGOTERMINOWE)

### 5.1 Rozważ migrację persystencji do SQLite
**Aktualnie:** JSON files
**Problem:** Przy dużej historii ocen (>1000 wpisów) JSON staje się wolny.

**Rekomendacja:**
```python
import sqlite3
from contextlib import contextmanager

class Database:
    def __init__(self, path="app_data/app.db"):
        self.path = path
        self._init_db()

    @contextmanager
    def connection(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def add_evaluation(self, data: dict):
        with self.connection() as conn:
            conn.execute("""
                INSERT INTO evaluations (id, timestamp, title, promise, score, verdict, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (data['id'], data['timestamp'], data['title'],
                  data['promise'], data['score'], data['verdict'],
                  json.dumps(data)))
            conn.commit()
```

### 5.2 Rozważ async dla wywołań API
**Problem:** Wywołania API blokują UI.

**Rekomendacja:** Użyj `asyncio` z `httpx` dla równoległych zapytań:
```python
import asyncio
import httpx

async def check_trends_async(topics: List[str]) -> List[Dict]:
    async with httpx.AsyncClient() as client:
        tasks = [check_single_trend(client, topic) for topic in topics]
        return await asyncio.gather(*tasks)
```

### 5.3 Dodaj system pluginów dla źródeł zewnętrznych
**Problem:** Dodanie nowego źródła (np. Reddit, Twitter) wymaga modyfikacji kodu.

**Rekomendacja:** Zdefiniuj interfejs pluginu:
```python
class ExternalSourcePlugin(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def check_topic(self, topic: str) -> Dict:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass
```

---

## 6. PLAN WDROŻENIA

### Faza 1: Quick Wins (1-2 dni)
1. [ ] Napraw obsługę błędów API (1.1)
2. [ ] Dodaj atomic write dla JSON (1.2)
3. [ ] Dodaj progress indicators (2.6)
4. [ ] Usuń duplikat PromiseGenerator (2.1)

### Faza 2: Stabilizacja (3-5 dni)
1. [ ] Wydziel moduły UI z app.py (2.2)
2. [ ] Stwórz warstwę abstrakcji LLM (2.4)
3. [ ] Przenieś progi do config.json (2.5)
4. [ ] Ujednolic nazewnictwo kolumn CSV (2.7)

### Faza 3: Jakość kodu (1 tydzień)
1. [ ] Dodaj type hints do kluczowych funkcji
2. [ ] Dodaj podstawowe testy jednostkowe (3.7)
3. [ ] Prekompiluj wyrażenia regularne (3.2)
4. [ ] Dodaj logging zamiast print (3.1)

### Faza 4: Długoterminowe (opcjonalne)
1. [ ] Migracja do SQLite
2. [ ] Async dla API calls
3. [ ] System pluginów

---

## METRYKI SUKCESU

Po wdrożeniu zmian z Fazy 1-2 oczekuję:
- **Stabilność:** 0 crashy związanych z API/JSON
- **UX:** Użytkownik zawsze wie co się dzieje (progress bars)
- **Maintainability:** Pliki <500 linii, łatwiejszy onboarding

---

## ZAŁĄCZNIKI

### A. Pełna lista plików z liniami kodu

| Plik | Linie | Priorytet refactoringu |
|------|-------|------------------------|
| app.py | 3939 | WYSOKI |
| advanced_analytics.py | 2279 | ŚREDNI |
| yt_idea_evaluator_pro_v2.py | 1155 | NISKI |
| topic_analyzer.py | 973 | ŚREDNI |
| youtube_sync.py | 852 | NISKI |
| config_manager.py | 835 | NISKI |
| external_sources.py | 476 | NISKI |
| competitor_tracker.py | 143 | NISKI |

### B. Komendy do uruchomienia audytu kodu

```bash
# Sprawdź nieużywane importy
ruff check --select=F401 .

# Sprawdź type hints
mypy . --ignore-missing-imports

# Sprawdź złożoność cyklomatyczną
radon cc . -a -s

# Znajdź duplikaty kodu
pylint --disable=all --enable=duplicate-code .
```

---

*Raport wygenerowany automatycznie. Rekomendacje oparte na analizie statycznej kodu.*
