# Instrukcja podziału app.py na mniejsze moduły

**Autor:** Claude (audyt)
**Data:** 2026-02-02

---

## Podsumowanie

Plik `app.py` ma obecnie ~4000 linii kodu, co czyni go trudnym do utrzymania i nawigacji. Ten dokument opisuje krok po kroku jak podzielić go na logiczne moduły.

## Proponowana struktura

```
apka/
├── app.py                    # Główny entry point (~200 linii)
├── ui/                       # Moduły UI
│   ├── __init__.py
│   ├── styles.py            # CSS i stałe stylistyczne
│   ├── tooltips.py          # Definicje tooltipów (TOOLTIPS dict)
│   ├── sidebar.py           # Sekcja sidebar
│   ├── components.py        # Współdzielone komponenty UI
│   ├── tab_evaluate.py      # Zakładka "Oceń pomysł"
│   ├── tab_tools.py         # Zakładka "Narzędzia"
│   ├── tab_analytics.py     # Zakładka "Analytics"
│   ├── tab_history.py       # Zakładka "Historia"
│   ├── tab_vault.py         # Zakładka "Idea Vault"
│   ├── tab_data.py          # Zakładka "Dane"
│   └── tab_diagnostics.py   # Zakładka "Diagnostyka"
├── config_manager.py        # Bez zmian
├── yt_idea_evaluator_pro_v2.py  # Bez zmian
├── advanced_analytics.py    # Bez zmian
├── topic_analyzer.py        # Bez zmian
├── youtube_sync.py          # Bez zmian
├── external_sources.py      # Bez zmian
├── llm_provider.py          # Nowy (stworzony w tym audycie)
└── tests/                   # Testy
```

---

## Krok 1: Stwórz folder `ui/`

```bash
mkdir -p ui
touch ui/__init__.py
```

---

## Krok 2: Wydziel `ui/styles.py`

Przenieś cały blok CSS (linie ~150-290 w app.py):

```python
# ui/styles.py
"""
Style CSS dla aplikacji (dark mode friendly)
"""

import streamlit as st

CSS_STYLES = """
<style>
/* Dark mode friendly colors */
.stAlert > div {
    color: inherit !important;
}

/* Layout helpers */
.section-card {
    background: #161616;
    border: 1px solid #333;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
}

/* ... reszta CSS ... */
</style>
"""

def inject_styles():
    """Wstrzykuje style CSS do aplikacji"""
    st.markdown(CSS_STYLES, unsafe_allow_html=True)
```

---

## Krok 3: Wydziel `ui/tooltips.py`

Przenieś słownik TOOLTIPS (linie ~290-480):

```python
# ui/tooltips.py
"""
Definicje tooltipów dla UI
"""

TOOLTIPS = {
    "curiosity_gap": "Jak silna jest luka informacyjna...",
    "specificity": "Czy tytuł/obietnica są konkretne...",
    # ... wszystkie tooltips
}

def get_tooltip(key: str) -> str:
    """Zwraca tooltip dla klucza lub pusty string"""
    return TOOLTIPS.get(key, "")
```

---

## Krok 4: Wydziel `ui/components.py`

Przenieś funkcje renderujące komponenty:

```python
# ui/components.py
"""
Współdzielone komponenty UI
"""

import streamlit as st
from typing import Dict, List, Any, Optional


def render_verdict_card(result: Dict) -> None:
    """Renderuje kartę werdyktu (PASS/BORDER/FAIL)"""
    verdict = result.get("verdict", "UNKNOWN")
    score = result.get("packaging_score", 0)

    if verdict == "PASS":
        color = "#28a745"
        icon = "✅"
    elif verdict == "BORDER":
        color = "#ffc107"
        icon = "⚠️"
    else:
        color = "#dc3545"
        icon = "❌"

    st.markdown(f"""
    <div style="background: {color}20; border: 2px solid {color};
                border-radius: 12px; padding: 20px; text-align: center;">
        <h2 style="color: {color}; margin: 0;">{icon} {verdict}</h2>
        <h3 style="margin: 10px 0 0 0;">Score: {score}/100</h3>
    </div>
    """, unsafe_allow_html=True)


def render_dimensions(dimensions: Dict[str, int], tooltips: Dict[str, str]) -> None:
    """Renderuje wymiary oceny jako progress bars"""
    for key, value in dimensions.items():
        label = key.replace("_", " ").title()
        tooltip = tooltips.get(key, "")
        st.progress(value / 100, text=f"{label}: {value}/100")
        if tooltip:
            st.caption(tooltip)


def render_risk_flags(flags: List[Dict]) -> None:
    """Renderuje flagi ryzyka"""
    if not flags:
        return

    st.markdown("### ⚠️ Flagi ryzyka")
    for flag in flags:
        severity = flag.get("severity", "medium")
        color = {"high": "#dc3545", "medium": "#ffc107", "low": "#17a2b8"}.get(severity, "#6c757d")
        st.markdown(f"""
        <div style="border-left: 4px solid {color}; padding: 8px 12px; margin: 8px 0;">
            <strong>{flag.get('flag', '')}</strong><br>
            <small>{flag.get('explanation', '')}</small>
        </div>
        """, unsafe_allow_html=True)


def render_title_variants(variants: List[Dict]) -> None:
    """Renderuje warianty tytułów z ocenami"""
    if not variants:
        return

    st.markdown("### 📝 Warianty tytułów")
    for i, v in enumerate(variants, 1):
        score = v.get("score", 0)
        title = v.get("title", "")
        color = "#28a745" if score >= 70 else "#ffc107" if score >= 50 else "#dc3545"

        st.markdown(f"""
        <div style="display: flex; align-items: center; padding: 8px;
                    border-bottom: 1px solid #333;">
            <span style="background: {color}; color: white; padding: 2px 8px;
                         border-radius: 4px; margin-right: 12px;">{score}</span>
            <span>{title}</span>
        </div>
        """, unsafe_allow_html=True)
```

---

## Krok 5: Wydziel `ui/sidebar.py`

Przenieś logikę sidebar (linie ~1200-1450):

```python
# ui/sidebar.py
"""
Sidebar aplikacji - API keys, YouTube sync, statystyki
"""

import streamlit as st
from typing import Tuple
from config_manager import AppConfig


def render_sidebar(config: AppConfig) -> Tuple[str, str, str]:
    """
    Renderuje sidebar i zwraca (provider, api_key, model).
    """
    with st.sidebar:
        st.header("🔑 Konfiguracja API")

        # === LLM Provider Selection ===
        llm_provider = st.radio(
            "Provider LLM",
            options=["openai", "google"],
            index=0 if config.get("llm_provider") == "openai" else 1,
            horizontal=True
        )

        # === API Key Input ===
        if llm_provider == "openai":
            api_key = st.text_input(
                "OpenAI API Key",
                value=config.get_api_key(),
                type="password"
            )
            model = st.selectbox(
                "Model",
                options=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
                index=0
            )
        else:
            api_key = st.text_input(
                "Google AI API Key",
                value=config.get_google_api_key(),
                type="password"
            )
            model = st.selectbox(
                "Model",
                options=["gemini-1.5-pro", "gemini-1.5-flash"],
                index=0
            )

        # === Test Connection Button ===
        if st.button("🔗 Testuj połączenie"):
            # ... logika testowania
            pass

        st.divider()

        # === YouTube Sync ===
        st.subheader("📺 YouTube Sync")
        # ... logika YouTube sync

        st.divider()

        # === Statistics ===
        render_sidebar_stats()

    return llm_provider, api_key, model


def render_sidebar_stats():
    """Renderuje statystyki w sidebarze"""
    stats = st.session_state.get("llm_stats", {})
    st.metric("Wywołania LLM", stats.get("calls", 0))
    st.metric("Cache hits", stats.get("cached_hits", 0))
```

---

## Krok 6: Wydziel zakładki do osobnych plików

Dla każdej zakładki stwórz osobny plik:

### `ui/tab_evaluate.py` (główna zakładka oceny)

```python
# ui/tab_evaluate.py
"""
Zakładka: Oceń pomysł
"""

import streamlit as st
from typing import Dict, Optional
import pandas as pd

from ui.components import render_verdict_card, render_dimensions, render_title_variants


def render_evaluate_tab(
    merged_df: Optional[pd.DataFrame],
    evaluator,
    llm_provider: str,
    api_key: str,
    model: str
) -> None:
    """Renderuje zakładkę oceny pomysłu"""

    st.header("🎯 Oceń pomysł na film")

    # Input section
    topic = st.text_input(
        "Temat filmu",
        placeholder="np. Katastrofa lotnicza TWA 800"
    )

    col1, col2 = st.columns(2)
    with col1:
        n_titles = st.slider("Liczba tytułów", 1, 12, 6)
    with col2:
        n_promises = st.slider("Liczba obietnic", 1, 12, 6)

    # Options
    with st.expander("⚙️ Opcje zaawansowane"):
        check_competition = st.checkbox("Sprawdź konkurencję", value=True)
        check_trends = st.checkbox("Sprawdź trendy", value=True)
        check_external = st.checkbox("Źródła zewnętrzne", value=False)

    # Evaluate button
    if st.button("🚀 Oceń", type="primary", use_container_width=True):
        if not topic:
            st.warning("Wpisz temat")
            return

        if not api_key:
            st.error("Brak klucza API")
            return

        with st.spinner("Analizuję..."):
            result = _run_evaluation(
                topic, n_titles, n_promises,
                check_competition, check_trends, check_external,
                evaluator, llm_provider, api_key, model, merged_df
            )

        if result:
            _display_results(result)


def _run_evaluation(...) -> Optional[Dict]:
    """Uruchamia ocenę tematu"""
    # ... logika oceny
    pass


def _display_results(result: Dict) -> None:
    """Wyświetla wyniki oceny"""
    render_verdict_card(result)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Wymiary oceny")
        render_dimensions(result.get("dimensions", {}), {})

    with col2:
        st.subheader("📝 Tytuły")
        render_title_variants(result.get("title_variants", []))
```

### Podobnie dla pozostałych zakładek:

- `ui/tab_tools.py` - Narzędzia (Wtopa Analyzer, Content Gap, Kalendarz, etc.)
- `ui/tab_analytics.py` - Dashboard i wykresy
- `ui/tab_history.py` - Historia ocen
- `ui/tab_vault.py` - Idea Vault
- `ui/tab_data.py` - Zarządzanie danymi
- `ui/tab_diagnostics.py` - Diagnostyka i cache

---

## Krok 7: Uprość główny `app.py`

Po wydzieleniu modułów, `app.py` powinno wyglądać tak:

```python
# app.py
"""
YT Idea Evaluator Pro v4
========================
Główny entry point aplikacji.
"""

import streamlit as st

# === Config ===
st.set_page_config(
    page_title="YT Idea Evaluator Pro v4",
    page_icon="🎬",
    layout="wide"
)

# === Imports ===
from ui.styles import inject_styles
from ui.sidebar import render_sidebar
from ui.tab_evaluate import render_evaluate_tab
from ui.tab_tools import render_tools_tab
from ui.tab_analytics import render_analytics_tab
from ui.tab_history import render_history_tab
from ui.tab_vault import render_vault_tab
from ui.tab_data import render_data_tab
from ui.tab_diagnostics import render_diagnostics_tab

from config_manager import AppConfig, EvaluationHistory, IdeaVault
from yt_idea_evaluator_pro_v2 import YTIdeaEvaluatorV2


def main():
    """Główna funkcja aplikacji"""

    # === Inject styles ===
    inject_styles()

    # === Initialize ===
    config = AppConfig()
    history = EvaluationHistory()
    vault = IdeaVault()

    # === Sidebar ===
    llm_provider, api_key, model = render_sidebar(config)

    # === Load data ===
    merged_df = load_merged_data()

    # === Initialize evaluator ===
    evaluator = get_evaluator(api_key, merged_df)

    # === Main tabs ===
    tabs = st.tabs([
        "🎯 Oceń pomysł",
        "🛠️ Narzędzia",
        "📊 Analytics",
        "📜 Historia",
        "💡 Idea Vault",
        "📁 Dane",
        "🧪 Diagnostyka"
    ])

    with tabs[0]:
        render_evaluate_tab(merged_df, evaluator, llm_provider, api_key, model)

    with tabs[1]:
        render_tools_tab(merged_df, llm_provider, api_key, model)

    with tabs[2]:
        render_analytics_tab(merged_df, history)

    with tabs[3]:
        render_history_tab(history)

    with tabs[4]:
        render_vault_tab(vault)

    with tabs[5]:
        render_data_tab(config)

    with tabs[6]:
        render_diagnostics_tab()


if __name__ == "__main__":
    main()
```

---

## Kolejność wykonania

1. **Stwórz folder `ui/` i `__init__.py`**
2. **Wydziel `styles.py`** - łatwe, bez zależności
3. **Wydziel `tooltips.py`** - łatwe, bez zależności
4. **Wydziel `components.py`** - średnie, wymaga testowania
5. **Wydziel `sidebar.py`** - średnie, ma zależności od config
6. **Wydziel zakładki jedna po drugiej** - zaczynając od najprostszej (np. `tab_diagnostics.py`)
7. **Zaktualizuj główny `app.py`** - na końcu

## Wskazówki

### Import circular dependencies

Jeśli natrafisz na circular imports:

```python
# Zamiast:
from ui.components import render_verdict_card  # na początku pliku

# Użyj lazy import:
def some_function():
    from ui.components import render_verdict_card
    render_verdict_card(...)
```

### Session state

Wszystkie klucze session state są już zdefiniowane w `init_session_state()`. Używaj ich konsekwentnie:

```python
# Dobrze
st.session_state["topic_result_main"]

# Źle - może nie istnieć
st.session_state.topic_result_main
```

### Testowanie po każdym kroku

Po wydzieleniu każdego modułu:

1. Uruchom aplikację: `streamlit run app.py`
2. Przetestuj wszystkie zakładki
3. Sprawdź czy nie ma błędów w konsoli

---

## Szacowany czas

| Krok | Szacowany czas |
|------|----------------|
| Stwórz strukturę folderów | 5 min |
| Wydziel styles.py | 15 min |
| Wydziel tooltips.py | 10 min |
| Wydziel components.py | 30 min |
| Wydziel sidebar.py | 45 min |
| Wydziel tab_evaluate.py | 1-2h |
| Wydziel tab_tools.py | 1-2h |
| Wydziel pozostałe zakładki | 2-3h |
| Testowanie i debugowanie | 1-2h |
| **RAZEM** | **6-10h** |

---

## Po zakończeniu refaktoryzacji

Po pomyślnym podziale:

1. Uruchom wszystkie testy: `pytest tests/ -v`
2. Sprawdź czy aplikacja działa poprawnie
3. Usuń stary backup `app.py.bak` (jeśli tworzyłeś)
4. Zaktualizuj dokumentację

---

*Ten dokument jest częścią audytu przeprowadzonego 2026-02-02.*
