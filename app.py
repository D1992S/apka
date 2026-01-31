"""
YT IDEA EVALUATOR PRO v4
========================
Rozszerzona aplikacja do oceny pomysłów na filmy YouTube.

NOWOŚĆ w v4:
- Oceń TEMAT (nie tylko tytuł) - AI generuje tytuły z ocenami
- Generowanie obietnic z ocenami
- Zewnętrzne źródła: Wikipedia, News, Sezonowość
- Analiza konkurencji YouTube
- Viral Score Prediction

+ wszystkie 25 funkcji z v3
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import plotly.graph_objects as go
import plotly.express as px

# Import modułów v3
from yt_idea_evaluator_pro_v2 import YTIdeaEvaluatorV2, Config, format_result
from config_manager import (
    AppConfig, EvaluationHistory, IdeaVault, TrendAlerts,
    get_config, get_history, get_vault, get_alerts, generate_report,
    get_series_manager, get_competitor_manager
)
from youtube_sync import YouTubeSync, get_youtube_sync, GOOGLE_API_AVAILABLE

# Import NOWYCH modułów v4
try:
    from topic_analyzer import (
        TopicEvaluator, TitleGenerator, PromiseGenerator as TopicPromiseGenerator,
        CompetitorAnalyzer, ViralScorePredictor, SimilarVideosFinder,
        get_topic_evaluator
    )
    from external_sources import (
        get_wiki_api, get_news_checker, get_seasonality, get_trend_discovery
    )
    TOPIC_ANALYZER_AVAILABLE = True
except ImportError as e:
    TOPIC_ANALYZER_AVAILABLE = False
    print(f"Topic analyzer not available: {e}")



# Opcjonalnie: Competitor Tracker
try:
    from competitor_tracker import get_competitor_tracker
    COMPETITOR_TRACKER_AVAILABLE = True
except Exception:
    COMPETITOR_TRACKER_AVAILABLE = False

try:
    from advanced_analytics import (
        AdvancedAnalytics, HookAnalyzer, TrendsAnalyzer, 
        CompetitionScanner, PackagingDNA, TimingPredictor,
        PromiseGenerator, ABTitleTester, ContentGapFinder,
        WtopaAnalyzer, SeriesAnalyzer
    )
    ADVANCED_AVAILABLE = True
except ImportError as e:
    ADVANCED_AVAILABLE = False
    print(f"Advanced analytics not available: {e}")

# =============================================================================
# KONFIGURACJA STRONY
# =============================================================================

st.set_page_config(
    page_title="YT Idea Evaluator Pro v4",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# SESSION STATE - zachowuje dane między przeładowaniami
# =============================================================================

if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_title" not in st.session_state:
    st.session_state.last_title = ""
if "last_promise" not in st.session_state:
    st.session_state.last_promise = ""
if "comparison_result" not in st.session_state:
    st.session_state.comparison_result = None
if "ab_result" not in st.session_state:
    st.session_state.ab_result = None

# =============================================================================
# STYLE CSS (DARK MODE FRIENDLY)
# =============================================================================

st.markdown("""
<style>
/* Dark mode friendly colors */
.stAlert > div {
    color: inherit !important;
}

/* Cards */
.metric-card {
    background: var(--background-secondary, #1e1e1e);
    border: 1px solid var(--border-color, #333);
    border-radius: 10px;
    padding: 1rem;
    margin: 0.5rem 0;
}

/* Verdict colors */
.verdict-pass {
    background: linear-gradient(135deg, #1a472a 0%, #2d5a3d 100%);
    color: #90EE90;
}
.verdict-border {
    background: linear-gradient(135deg, #4a4000 0%, #5a5010 100%);
    color: #FFD700;
}
.verdict-fail {
    background: linear-gradient(135deg, #4a1a1a 0%, #5a2a2a 100%);
    color: #FF6B6B;
}

/* Tooltips */
.tooltip-icon {
    cursor: help;
    opacity: 0.7;
    margin-left: 5px;
}
.tooltip-icon:hover {
    opacity: 1;
}

/* Progress bars */
.dim-bar {
    height: 8px;
    border-radius: 4px;
    background: #333;
    margin: 5px 0;
}
.dim-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.3s;
}

/* Copy button */
.copy-btn {
    background: #4a4a4a;
    border: none;
    padding: 5px 15px;
    border-radius: 5px;
    cursor: pointer;
}
.copy-btn:hover {
    background: #5a5a5a;
}

/* Info boxes */
.info-box {
    background: rgba(100, 149, 237, 0.1);
    border-left: 4px solid #6495ED;
    padding: 1rem;
    border-radius: 0 8px 8px 0;
    margin: 1rem 0;
}

/* Warning boxes */
.warning-box {
    background: rgba(255, 193, 7, 0.1);
    border-left: 4px solid #FFC107;
    padding: 1rem;
    border-radius: 0 8px 8px 0;
    margin: 1rem 0;
}

/* Success boxes */
.success-box {
    background: rgba(40, 167, 69, 0.1);
    border-left: 4px solid #28a745;
    padding: 1rem;
    border-radius: 0 8px 8px 0;
    margin: 1rem 0;
}

/* Ranking */
.rank-1 { color: #FFD700; font-weight: bold; }
.rank-2 { color: #C0C0C0; }
.rank-3 { color: #CD7F32; }


/* Score badge with tooltip */
.score-badge{
    display:inline-block;
    min-width:38px;
    text-align:center;
    padding:2px 8px;
    margin-right:8px;
    border-radius:999px;
    border:1px solid var(--border-color, #333);
    background: rgba(255,255,255,0.06);
    font-weight:700;
    font-size:0.9rem;
    cursor: help;
}

</style>
""", unsafe_allow_html=True)

# =============================================================================
# TOOLTIPS - WYJAŚNIENIA
# =============================================================================

TOOLTIPS = {
    "judges": """
**Liczba sędziów LLM**

Ile razy model GPT oceni Twój pomysł. Więcej = dokładniejsza ocena, ale wolniejsza i droższa.

- **1 sędzia**: Szybko, tanie, ale może być niestabilne
- **2 sędziów**: Dobry balans (zalecane)
- **3 sędziów**: Najdokładniejsze, ale 3x dłużej
""",
    
    "topn": """
**Podobne przykłady**

Ile Twoich filmów model weźmie pod uwagę jako kontekst.

- **3-5**: Szybkie, ogólne porównanie
- **5-7**: Dobry balans (zalecane)
- **8-10**: Głębsza analiza, wolniejsze
""",
    
    "optimize": """
**Optymalizuj warianty**

Gdy włączone, model wygeneruje warianty tytułu i oceni każdy z nich osobno, szukając najlepszego.

⚠️ Wydłuża czas oceny 2-3x
""",
    
    "data_score": """
**Data Score (ML)**

Ocena z modelu Machine Learning trenowanego na TWOICH danych.

Model nauczył się wzorców z Twoich hitów vs wtop i przewiduje czy nowy pomysł pasuje do wzorca sukcesu.

- Używa embeddingów OpenAI
- Trenowany na Ridge Regression i LogisticRegression
- Im więcej danych, tym dokładniejszy
""",
    
    "llm_score": """
**LLM Score**

Ocena od GPT-4o który analizuje:
- Curiosity gap (czy buduje ciekawość)
- Specyficzność (czy jest konkretny)
- Dark niche fit (czy pasuje do niszy)
- Hook potential (potencjał na mocny hook)
- Shareability (czy ludzie będą udostępniać)
- Title craft (jakość tytułu)
""",
    
    "risk_penalty": """
**Kara za ryzyko**

Punkty odjęte za wykryte ryzyka:
- CLICKBAIT_BACKFIRE: Tytuł obiecuje za dużo
- OVERSATURATED: Temat przesycony
- TOO_NICHE: Za wąski temat
- WEAK_HOOK: Słaby potencjał na hook
- LOW_SHAREABILITY: Niska viralowość
- TITLE_TOO_LONG/SHORT: Problem z długością
- NO_CLEAR_PROMISE: Brak obietnicy
- CONTROVERSIAL: Ryzykowny temat
""",
    
    "trend_bonus": """
**Bonus/Kara za Trend**

Sprawdza Google Trends:
- 🔥 +10: Temat HOT, trending up
- ➡️ +5: Evergreen, stabilny
- 📉 -5: Trend spadkowy
- 💀 -10: Temat martwy
""",
    
    "competition_bonus": """
**Bonus/Kara za Konkurencję**

Skanuje YouTube:
- 🟢 +15: Blue ocean, brak konkurencji
- 🟢 +10: Niska konkurencja
- 🟡 0: Umiarkowana
- 🟠 -5: Wysoka konkurencja
- 🔴 -15: Temat przesycony
""",
    
    "dna_bonus": """
**Bonus za DNA Match**

Sprawdza czy tytuł pasuje do wzorców Twoich hitów:
- Optymalna długość
- Trigger words z Twoich hitów
- Struktury które działają
- Max +20 punktów
""",
}

def show_tooltip(key: str):
    """Wyświetla tooltip jako help w Streamlit"""
    return TOOLTIPS.get(key, "")

# =============================================================================
# INICJALIZACJA
# =============================================================================

# Załaduj konfigurację
config = get_config()
history = get_history()
vault = get_vault()
alerts = get_alerts()

# Paths
CHANNEL_DATA_DIR = Path("./channel_data")
MERGED_DATA_FILE = CHANNEL_DATA_DIR / "merged_channel_data.csv"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_merged_data() -> Optional[pd.DataFrame]:
    """Ładuje połączone dane kanału"""
    # Najpierw sprawdź synced data
    synced_file = CHANNEL_DATA_DIR / "synced_channel_data.csv"
    if synced_file.exists():
        return pd.read_csv(synced_file)
    
    if MERGED_DATA_FILE.exists():
        return pd.read_csv(MERGED_DATA_FILE)
    return None

@st.cache_resource
def get_evaluator(api_key: str, data_path: str) -> YTIdeaEvaluatorV2:
    """Cache'owany evaluator"""
    evaluator = YTIdeaEvaluatorV2()
    evaluator.initialize(api_key)
    evaluator.load_data(data_path)
    evaluator.build_embeddings()
    evaluator.train_models()
    return evaluator

@st.cache_resource
def get_advanced_analytics(data_path: str, _api_key: str = None):
    """Cache'owane advanced analytics"""
    if not ADVANCED_AVAILABLE:
        return None
    
    from openai import OpenAI
    client = OpenAI(api_key=_api_key) if _api_key else None
    
    analytics = AdvancedAnalytics(openai_client=client)
    df = pd.read_csv(data_path)
    analytics.load_data(df)
    return analytics

def copy_to_clipboard_button(text: str, button_text: str = "📋 Kopiuj"):
    """Przycisk do kopiowania do schowka"""
    # Streamlit nie ma natywnego copy to clipboard, używamy JS
    st.code(text, language=None)
    st.caption("↑ Zaznacz i skopiuj (Ctrl+C)")

# =============================================================================
# RENDER FUNCTIONS
# =============================================================================

def render_verdict_card(result: Dict):
    """Renderuje główną kartę z werdyktem"""
    score = result.get("final_score_with_bonus", result.get("final_score", 0))
    verdict = result.get("final_verdict", "BORDER")
    advanced_bonus = result.get("advanced_bonus", 0)
    
    verdict_class = f"verdict-{verdict.lower()}"
    emoji = {"PASS": "🟢", "BORDER": "🟡", "FAIL": "🔴"}.get(verdict, "🟡")
    
    data_score = result.get('data_score', 0)
    llm_score = result.get('llm_score', 0)
    risk_penalty = result.get('risk_penalty', 0)
    
    st.markdown(f"""
    <div class="{verdict_class}" style="padding: 2rem; border-radius: 12px; text-align: center; margin: 1rem 0;">
        <div style="font-size: 3rem; font-weight: bold;">
            {emoji} {verdict}
        </div>
        <div style="font-size: 2.5rem; margin-top: 0.5rem;">
            {score:.0f}/100
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Breakdown
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Data (ML)", 
            f"{data_score:.0f}",
            help=TOOLTIPS["data_score"]
        )
    with col2:
        st.metric(
            "LLM", 
            f"{llm_score:.0f}",
            help=TOOLTIPS["llm_score"]
        )
    with col3:
        st.metric(
            "Kara", 
            f"-{risk_penalty:.0f}",
            help=TOOLTIPS["risk_penalty"]
        )
    with col4:
        st.metric(
            "Bonus", 
            f"+{advanced_bonus:.0f}" if advanced_bonus >= 0 else f"{advanced_bonus:.0f}",
            help="Suma bonusów z trendów, konkurencji i DNA match"
        )

def render_diagnosis(result: Dict):
    """Renderuje diagnozę i ryzyka"""
    why = result.get('why', 'Brak diagnozy')
    
    st.markdown(f"""
    <div class="info-box">
        <strong>💬 DIAGNOZA:</strong><br>
        {why}
    </div>
    """, unsafe_allow_html=True)
    
    # Risk flags z wyjaśnieniami
    risk_flags = result.get('risk_flags', [])
    if risk_flags:
        st.markdown("**🚩 Wykryte ryzyka:**")
        
        risk_explanations = {
            "CLICKBAIT_BACKFIRE": "Tytuł może być postrzegany jako clickbait",
            "OVERSATURATED": "Temat jest przesycony na YouTube",
            "TOO_NICHE": "Temat może być za wąski",
            "WEAK_HOOK": "Trudno będzie zrobić mocny hook",
            "LOW_SHAREABILITY": "Mała szansa na udostępnienia",
            "TITLE_TOO_LONG": "Tytuł za długi (może być ucięty)",
            "TITLE_TOO_SHORT": "Tytuł za krótki (brak kontekstu)",
            "NO_CLEAR_PROMISE": "Brak jasnej obietnicy dla widza",
            "CONTROVERSIAL": "Temat kontrowersyjny - ryzyko",
        }
        
        for flag in risk_flags:
            explanation = risk_explanations.get(flag, "")
            st.markdown(f"""
            <span style="background: rgba(255,107,107,0.2); padding: 4px 12px; border-radius: 15px; margin: 2px; display: inline-block;">
                ⚠️ {flag}
                {f'<span style="opacity:0.7; font-size:0.9em;"> - {explanation}</span>' if explanation else ''}
            </span>
            """, unsafe_allow_html=True)

def render_dimensions(result: Dict):
    """Renderuje wymiary oceny jako progress bary"""
    dims = result.get("dimensions", {})
    if not dims:
        return
    
    dim_names_pl = {
        "curiosity_gap": "Curiosity Gap",
        "specificity": "Specyficzność",
        "dark_niche_fit": "Dark Niche Fit",
        "hook_potential": "Hook Potential",
        "shareability": "Shareability",
        "title_craft": "Title Craft",
    }
    
    for dim, value in dims.items():
        name = dim_names_pl.get(dim, dim)
        color = "#4CAF50" if value >= 70 else "#FFC107" if value >= 50 else "#f44336"
        
        st.markdown(f"""
        <div style="margin: 8px 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                <span>{name}</span>
                <span style="font-weight: bold;">{value}/100</span>
            </div>
            <div class="dim-bar">
                <div class="dim-bar-fill" style="width: {value}%; background: {color};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_improvements(result: Dict):
    """Renderuje listę ulepszeń"""
    improvements = result.get("improvements", [])
    if not improvements:
        return
    
    st.subheader("✨ Co poprawić")
    
    for i, imp in enumerate(improvements[:5], 1):
        priority = "🔴" if i == 1 else "🟡" if i == 2 else "⚪"
        st.markdown(f"{priority} **{i}.** {imp}")

def render_variants_with_scores(result: Dict, evaluator=None, analytics=None):
    """Renderuje warianty tytułów z ocenami"""
    title_variants = result.get("title_variants", [])
    promise_variants = result.get("promise_variants", [])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📝 Warianty tytułu:**")
        for i, var in enumerate(title_variants[:6], 1):
            # Quick score jeśli mamy analytics
            score = ""
            if analytics and hasattr(analytics, 'ab_tester') and analytics.ab_tester:
                try:
                    s = analytics.ab_tester._calculate_ctr_score(var)
                    score = f" `{s:.0f}`"
                except:
                    pass
            st.markdown(f"{i}. {var}{score}")
    
    with col2:
        st.markdown("**💬 Warianty obietnicy:**")
        for i, var in enumerate(promise_variants[:6], 1):
            st.markdown(f"{i}. {var}")

def render_advanced_insights(result: Dict):
    """Renderuje insights z zaawansowanych analiz"""
    insights = result.get("advanced_insights", {})
    if not insights:
        return
    
    st.subheader("🧬 Zaawansowane analizy")
    
    cols = st.columns(3)
    
    # Trend
    with cols[0]:
        if "trends" in insights:
            trends = insights["trends"]
            overall = trends.get("overall", {})
            bonus = overall.get("trend_bonus", 0)
            msg = overall.get("message", "")[:50]
            
            color = "#2d5a3d" if bonus > 0 else "#5a2a2a" if bonus < 0 else "#3a3a3a"
            
            st.markdown(f"""
            <div style="background: {color}; padding: 1rem; border-radius: 8px; text-align: center;">
                <div style="font-size: 0.9rem; opacity: 0.8;">📈 Trend</div>
                <div style="font-size: 1.8rem; font-weight: bold;">{bonus:+d}</div>
                <div style="font-size: 0.8rem; opacity: 0.7;">{msg}</div>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("ℹ️ Co to znaczy?"):
                st.markdown(TOOLTIPS["trend_bonus"])
    
    # Competition
    with cols[1]:
        if "competition" in insights:
            comp = insights["competition"].get("analysis", {})
            bonus = comp.get("score_bonus", 0)
            saturation = comp.get("saturation", "UNKNOWN")
            emoji = comp.get("emoji", "❓")
            
            color = "#2d5a3d" if bonus > 0 else "#5a2a2a" if bonus < 0 else "#3a3a3a"
            
            st.markdown(f"""
            <div style="background: {color}; padding: 1rem; border-radius: 8px; text-align: center;">
                <div style="font-size: 0.9rem; opacity: 0.8;">🔍 Konkurencja</div>
                <div style="font-size: 1.8rem; font-weight: bold;">{emoji} {bonus:+d}</div>
                <div style="font-size: 0.8rem; opacity: 0.7;">{saturation}</div>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("ℹ️ Co to znaczy?"):
                st.markdown(TOOLTIPS["competition_bonus"])
    
    # DNA
    with cols[2]:
        if "dna_match" in insights:
            dna = insights["dna_match"]
            bonus = dna.get("bonus", 0)
            matches = dna.get("matches", [])
            
            color = "#2d5a3d" if bonus > 0 else "#3a3a3a"
            
            st.markdown(f"""
            <div style="background: {color}; padding: 1rem; border-radius: 8px; text-align: center;">
                <div style="font-size: 0.9rem; opacity: 0.8;">🧬 DNA Match</div>
                <div style="font-size: 1.8rem; font-weight: bold;">+{bonus}</div>
                <div style="font-size: 0.8rem; opacity: 0.7;">{len(matches)} dopasowań</div>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("ℹ️ Co to znaczy?"):
                st.markdown(TOOLTIPS["dna_bonus"])
                if matches:
                    for m in matches:
                        st.markdown(f"- {m}")

def render_copy_report(result: Dict):
    """Renderuje przycisk do kopiowania raportu"""
    report = generate_report(result)
    
    with st.expander("📋 Kopiuj pełny raport"):
        st.text_area("Raport do skopiowania:", report, height=400)
        st.caption("Zaznacz wszystko (Ctrl+A) i skopiuj (Ctrl+C)")

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.title("🎬 YT Evaluator Pro")
    st.caption("v3.0 - Kompletna edycja")
    
    st.divider()
    
    # === API KEY ===
    st.subheader("🔑 OpenAI API")
    
    saved_key = config.get_api_key()
    api_key = st.text_input(
        "API Key",
        value=saved_key,
        type="password",
        help="Twój klucz OpenAI API"
    )
    
    if api_key != saved_key:
        if st.button("💾 Zapisz klucz"):
            config.set_api_key(api_key)
            st.success("✅ Zapisano!")
            st.rerun()
    
    if api_key:
        st.success("✅ API Key ustawiony")
    else:
        st.warning("⚠️ Brak API Key")
    
    st.divider()
    
    # === YOUTUBE SYNC ===
    st.subheader("📺 YouTube Sync")
    
    yt_sync = get_youtube_sync()
    last_sync = yt_sync.get_last_sync_time()
    
    if last_sync:
        st.caption(f"Ostatnia sync: {last_sync}")
    
    if not GOOGLE_API_AVAILABLE:
        st.warning("⚠️ Zainstaluj: `pip install google-api-python-client google-auth-oauthlib`")
    elif not yt_sync.has_credentials():
        with st.expander("📖 Jak skonfigurować?"):
            st.markdown(yt_sync.setup_instructions())
    else:
        if st.button("🔄 Synchronizuj dane", use_container_width=True):
            with st.spinner("Loguję do YouTube..."):
                success, msg = yt_sync.authenticate()
                if success:
                    st.success(msg)
                    with st.spinner("Pobieram dane..."):
                        df, sync_msg = yt_sync.sync_all(include_analytics=True)
                        st.success(sync_msg)
                        st.rerun()
                else:
                    st.error(msg)
    
    st.divider()
    
    # === DANE KANAŁU ===
    st.subheader("📊 Dane kanału")
    
    merged_df = load_merged_data()
    if merged_df is not None:
        st.success(f"✅ {len(merged_df)} filmów")
        
        cols = []
        if "views" in merged_df.columns:
            cols.append(f"views: ✅")
        if "retention" in merged_df.columns:
            cols.append(f"retention: ✅")
        if "label" in merged_df.columns:
            cols.append(f"labels: ✅")
        
        st.caption(" | ".join(cols) if cols else "Podstawowe dane")
    else:
        st.warning("⚠️ Brak danych")
        st.caption("Użyj YouTube Sync lub wgraj CSV")
    
    st.divider()
    
    # === STATYSTYKI ===
    st.subheader("📈 Statystyki")
    
    total_evals = len(history.get_all())
    vault_ideas = len(vault.get_all(status="waiting"))
    tracked = len([e for e in history.get_all() if e.get("published")])
    
    st.metric("Ocen w historii", total_evals)
    st.metric("Pomysłów w Vault", vault_ideas)
    st.metric("Tracked filmów", tracked)

# =============================================================================
# MAIN TABS
# =============================================================================

tab_evaluate, tab_compare, tab_tools, tab_analytics, tab_history, tab_vault, tab_data = st.tabs([
    "🎯 Oceń pomysł",
    "⚖️ Porównaj",
    "🛠️ Narzędzia",
    "📊 Analytics",
    "📜 Historia",
    "💡 Idea Vault",
    "📁 Dane"
])

# =============================================================================
# NOWY TAB: OCEŃ TEMAT (v4 feature)
# =============================================================================
# Ten tab jest dostępny w zakładce Narzędzia jako "🎯 Oceń TEMAT"

# =============================================================================
# TAB: OCEŃ POMYSŁ
# =============================================================================

with tab_evaluate:
    st.header("🎯 Oceń pomysł na film")

    # =========================================================================
    # NOWY TRYB: OCENA TEMATU (bez wpisywania tytułu)
    # =========================================================================
    st.subheader("⚡ Ocena tematu (bez wymyślania tytułu)")
    st.caption("Wpisz sam temat, a aplikacja zaproponuje najlepsze tytuły i obietnice oraz policzy wynik.")
    
    # Session state for step-by-step evaluation
    if "topic_job_main" not in st.session_state:
        st.session_state.topic_job_main = None
    if "topic_result_main" not in st.session_state:
        st.session_state.topic_result_main = None
    
    col_t1, col_t2, col_t3 = st.columns([3, 1, 1])
    with col_t1:
        topic_input_main = st.text_input(
            "🧠 Temat / hasło",
            placeholder="np. Operacja Northwoods",
            key="main_topic_input"
        )
    with col_t2:
        n_titles_main = st.slider("Ile tytułów", 3, 12, 6, key="main_n_titles")
    with col_t3:
        n_promises_main = st.slider("Ile obietnic", 3, 12, 6, key="main_n_promises")
    
    with st.expander("⚙️ Co ma brać pod uwagę", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            inc_competition = st.checkbox("Konkurencja (YouTube)", value=True, key="main_inc_comp")
            inc_similar = st.checkbox("Podobne hity (kanał)", value=True, key="main_inc_sim")
        with c2:
            inc_trends = st.checkbox("Google Trends", value=True, key="main_inc_trends")
            inc_external = st.checkbox("Źródła zewnętrzne (Wiki/News)", value=True, key="main_inc_ext")
        with c3:
            inc_viral = st.checkbox("Viral Score", value=True, key="main_inc_viral")
            inc_timeline = st.checkbox("Performance Timeline", value=True, key="main_inc_timeline")
    
    def _score_badge(score: int, tooltip: str) -> str:
        # HTML tooltip via title=
        safe_tip = (tooltip or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
        return f"<span class='score-badge' title=\"{safe_tip}\">{score}</span>"
    
    def _estimate_timeline(final_score: int, similar_hits: list) -> dict:
        """
        Heurystyka timeline: jeśli mamy podobne hity, bierzemy medianę ich views jako proxy.
        Jeśli nie, mapujemy score na percentyl w rozkładzie views kanału.
        """
        try:
            views_candidates = [h.get("views", 0) for h in (similar_hits or []) if isinstance(h.get("views", 0), (int, float))]
            views_candidates = [v for v in views_candidates if v and v > 0]
            if views_candidates:
                base_30 = int(sorted(views_candidates)[len(views_candidates)//2])
            else:
                base_30 = None
        except Exception:
            base_30 = None
    
        if base_30 is None and merged_df is not None and "views" in merged_df.columns:
            try:
                views = merged_df["views"].fillna(0).astype(float)
                views = views[views > 0].sort_values()
                if len(views) > 10:
                    p = min(0.95, max(0.05, final_score / 100.0))
                    idx = int(p * (len(views) - 1))
                    base_30 = int(views.iloc[idx])
            except Exception:
                base_30 = None
    
        if base_30 is None:
            base_30 = int(10000 * (0.5 + final_score / 100.0))  # bardzo zgrubny fallback
    
        # Prosty model krzywej
        day1 = int(base_30 * 0.18)
        day7 = int(base_30 * 0.58)
        day30 = int(base_30)
        return {"day_1": day1, "day_7": day7, "day_30": day30, "basis": "similar_hits" if similar_hits else "channel_distribution"}
    
    def _topic_stage_run(stage: int, job: dict) -> dict:
        """Uruchamia jeden etap oceny tematu, zapisuje wynik w job['result']."""
        topic = job.get("topic", "")
        if not topic:
            return job
    
        # Initialize OpenAI client and evaluator only when needed
        api_key_local = job.get("api_key", "") or ""
        client = None
        if api_key_local:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=api_key_local)
            except Exception:
                client = None
    
        if not TOPIC_ANALYZER_AVAILABLE:
            job["error"] = "Topic analyzer niedostępny. Sprawdź pliki i requirements."
            return job
    
        evaluator = get_topic_evaluator(client, merged_df)
        res = job.get("result", {"topic": topic, "timestamp": datetime.now().isoformat()})
    
        # Stage 0: titles + promises
        if stage == 0:
            res["titles"] = evaluator.title_generator.generate(topic, n=job.get("n_titles", 6))
            if res.get("titles"):
                res["selected_title"] = res["titles"][0]
                best_title = res["selected_title"]["title"]
                res["promises"] = evaluator.promise_generator.generate(best_title, topic, n=job.get("n_promises", 6))
            job["result"] = res
            job["stage_done"] = 0
            return job
    
        # Stage 1: competition
        if stage == 1:
            if job.get("inc_competition", True):
                res["competition"] = evaluator.competitor_analyzer.analyze(topic)
            job["result"] = res
            job["stage_done"] = 1
            return job
    
        # Stage 2: trends + external
        if stage == 2:
            if job.get("inc_trends", True) and ADVANCED_AVAILABLE:
                try:
                    ta = TrendsAnalyzer()
                    trend = ta.check_trend([topic])
                except Exception as e:
                    trend = {"status": "ERROR", "message": str(e)}
                res["trends"] = trend
            if job.get("inc_external", True):
                try:
                    wiki_api = get_wiki_api()
                    news = get_news_checker()
                    season = get_seasonality()
                    discovery = get_trend_discovery()
                    res["external_data"] = {
                        "wikipedia": wiki_api.search(topic, limit=3),
                        "news": news.check_recent_news(topic, days=30),
                        "seasonality": season.analyze_seasonality(topic),
                        "trend_discovery": discovery.discover_related_topics(topic, limit=8)
                    }
                except Exception as e:
                    res["external_data"] = {"error": str(e)}
            job["result"] = res
            job["stage_done"] = 2
            return job
    
        # Stage 3: similar hits
        if stage == 3:
            if job.get("inc_similar", True) and evaluator.similar_finder:
                best_title = res.get("selected_title", {}).get("title") or topic
                res["similar_hits"] = evaluator.similar_finder.find(topic, best_title)
            job["result"] = res
            job["stage_done"] = 3
            return job
    
        # Stage 4: viral + final score + timeline
        if stage == 4:
            best_title = res.get("selected_title", {}).get("title") or topic
            if job.get("inc_viral", True):
                res["viral_score"] = evaluator.viral_predictor.predict(best_title, topic, res.get("competition", {}))
            # base overall score (TopicEvaluator formula)
            title_score = int(res.get("selected_title", {}).get("score", 50))
            competition_score = int(res.get("competition", {}).get("opportunity_score", 50))
            viral_score = int(res.get("viral_score", {}).get("viral_score", 50))
            base_overall = int(title_score * 0.35 + competition_score * 0.30 + viral_score * 0.35)
    
            # trend bonus from TrendsAnalyzer overall.score
            trend_bonus = 0
            if res.get("trends", {}).get("overall", {}).get("score") is not None:
                try:
                    overall_trend_score = int(res["trends"]["overall"]["score"])
                    trend_bonus = max(-10, min(10, int((overall_trend_score - 50) / 5)))
                except Exception:
                    trend_bonus = 0
    
            # similar bonus from similar hits
            similar_bonus = 0
            try:
                hits = res.get("similar_hits") or []
                if hits:
                    best = sorted(hits, key=lambda x: (x.get("views", 0) or 0), reverse=True)[0]
                    if best.get("label") == "PASS" and (best.get("views", 0) or 0) >= 100000:
                        similar_bonus = 6
                    elif best.get("label") == "PASS":
                        similar_bonus = 3
                    elif best.get("label") == "FAIL":
                        similar_bonus = -3
            except Exception:
                similar_bonus = 0
    
            final_score = int(max(0, min(100, base_overall + trend_bonus + similar_bonus)))
            res["overall_score_base"] = base_overall
            res["trend_bonus"] = trend_bonus
            res["similar_bonus"] = similar_bonus
            res["overall_score"] = final_score
            res["recommendation"] = evaluator._generate_recommendation(res)
    
            if job.get("inc_timeline", True):
                res["performance_timeline"] = _estimate_timeline(final_score, res.get("similar_hits", []))
    
            job["result"] = res
            job["stage_done"] = 4
            job["done"] = True
            return job
    
        return job
    
    # Action buttons
    b1, b2, b3, b4 = st.columns([1, 1, 1, 1])
    with b1:
        start_step = st.button("🧩 Start krokowo", use_container_width=True, key="main_topic_step_start")
    with b2:
        full_run = st.button("🚀 Pełna ocena", use_container_width=True, key="main_topic_full")
    with b3:
        cont_step = st.button("➡️ Kontynuuj", use_container_width=True, key="main_topic_step_continue")
    with b4:
        stop_and_save = st.button("⏹️ Zatrzymaj i zapisz", use_container_width=True, key="main_topic_step_stop")
    
    # Initialize job
    if start_step and topic_input_main:
        st.session_state.topic_job_main = {
            "topic": topic_input_main.strip(),
            "n_titles": n_titles_main,
            "n_promises": n_promises_main,
            "api_key": api_key,
            "stage": 0,
            "inc_competition": inc_competition,
            "inc_similar": inc_similar,
            "inc_trends": inc_trends,
            "inc_external": inc_external,
            "inc_viral": inc_viral,
            "inc_timeline": inc_timeline,
            "result": {"topic": topic_input_main.strip(), "timestamp": datetime.now().isoformat()}
        }
        st.session_state.topic_job_main = _topic_stage_run(0, st.session_state.topic_job_main)
        st.session_state.topic_result_main = st.session_state.topic_job_main.get("result")
    
    # Full run
    if full_run and topic_input_main:
        job = {
            "topic": topic_input_main.strip(),
            "n_titles": n_titles_main,
            "n_promises": n_promises_main,
            "api_key": api_key,
            "inc_competition": inc_competition,
            "inc_similar": inc_similar,
            "inc_trends": inc_trends,
            "inc_external": inc_external,
            "inc_viral": inc_viral,
            "inc_timeline": inc_timeline,
            "result": {"topic": topic_input_main.strip(), "timestamp": datetime.now().isoformat()}
        }
        with st.spinner("Oceniam temat (pełna analiza)..."):
            for stg in [0, 1, 2, 3, 4]:
                job = _topic_stage_run(stg, job)
        st.session_state.topic_job_main = job
        st.session_state.topic_result_main = job.get("result")
    
    # Continue step-by-step
    if cont_step and st.session_state.topic_job_main:
        job = st.session_state.topic_job_main
        stage = int(job.get("stage_done", 0)) + 1
        stage = min(stage, 4)
        with st.spinner(f"Etap {stage+1}/5..."):
            job = _topic_stage_run(stage, job)
            job["stage"] = stage
        st.session_state.topic_job_main = job
        st.session_state.topic_result_main = job.get("result")
    
    # Stop and save partial
    if stop_and_save and st.session_state.topic_result_main:
        # Save to Idea Vault and History with full payload
        res = st.session_state.topic_result_main
        best_title = (res.get("selected_title") or {}).get("title") or f"Temat: {res.get('topic','')}"
        best_promise = (res.get("promises") or [{}])[0].get("promise", "")
        score = int(res.get("overall_score", res.get("overall_score_base", 0) or 0))
    
        vault.add(
            title=best_title,
            promise=best_promise,
            score=score,
            reason="Ocena tematu (zatrzymana) - zapis pełnego wyniku",
            tags=["topic_mode"],
            topic=res.get("topic",""),
            payload=res
        )
    
        # Save to history
        history.add({
            "title": best_title,
            "promise": best_promise,
            "final_score": score,
            "final_score_with_bonus": score,
            "final_verdict": "PASS" if score >= 75 else "BORDER" if score >= 60 else "FAIL",
            "data_score": 0,
            "llm_score": 0,
            "risk_penalty": 0,
            "advanced_bonus": 0,
            "advanced_insights": {},
            "topic_mode": True,
            "payload": res
        })
    
        st.success("✅ Zapisano (Vault + Historia).")
        st.session_state.topic_job_main = None
    
    # Display topic result
    if st.session_state.topic_result_main:
        res = st.session_state.topic_result_main
    
        st.divider()
        st.subheader("Wynik oceny tematu")
    
        score_val = int(res.get("overall_score", res.get("overall_score_base", 0) or 0))
        st.metric("Ocena", f"{score_val}/100")
        if res.get("recommendation"):
            st.info(res.get("recommendation"))
    
        # Timeline
        if res.get("performance_timeline"):
            tl = res["performance_timeline"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Predykcja views 1 dzień", f"{tl.get('day_1',0):,}")
            c2.metric("Predykcja views 7 dni", f"{tl.get('day_7',0):,}")
            c3.metric("Predykcja views 30 dni", f"{tl.get('day_30',0):,}")
    
        # Titles with tooltips
        st.markdown("### Proponowane tytuły")
        if res.get("titles"):
            titles = res["titles"]
            # selection
            title_opts = [t.get("title","") for t in titles]
            selected_title_str = st.selectbox("Wybierz tytuł do dalszej oceny", title_opts, index=0, key="main_selected_title")
            selected_obj = next((t for t in titles if t.get("title")==selected_title_str), titles[0])
            res["selected_title"] = selected_obj
    
            for t in titles:
                badge = _score_badge(int(t.get("score",0)), t.get("reason",""))
                st.markdown(f"{badge} <b>{t.get('title','')}</b>", unsafe_allow_html=True)
                if t.get("tags"):
                    st.caption("Tagi: " + ", ".join(t.get("tags", [])))
        else:
            st.warning("Brak wygenerowanych tytułów.")
    
        # Promises
        st.markdown("### Proponowane obietnice (hook/promise)")
        if res.get("promises"):
            promises = res["promises"]
            promise_opts = [p.get("promise","") for p in promises]
            chosen_promise = st.selectbox("Wybierz obietnicę", promise_opts, index=0, key="main_selected_promise")
            for p in promises:
                badge = _score_badge(int(p.get("score",0)), p.get("reason",""))
                st.markdown(f"{badge} {p.get('promise','')}", unsafe_allow_html=True)
        else:
            st.info("Brak obietnic. Najpierw uruchom etap 0.")
    
        # Competition, trends, similar
        exp = st.expander("🧪 Szczegóły analizy", expanded=False)
        with exp:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Konkurencja**")
                st.json(res.get("competition", {}))
                st.markdown("**Viral Score**")
                st.json(res.get("viral_score", {}))
            with c2:
                st.markdown("**Google Trends**")
                st.json(res.get("trends", {}))
                st.markdown("**External**")
                st.json(res.get("external_data", {}))
    
            st.markdown("**Podobne hity na kanale**")
            if res.get("similar_hits"):
                for h in res["similar_hits"][:8]:
                    st.markdown(f"- {h.get('title','')} | {h.get('views',0):,} views | {h.get('label','')}")
            else:
                st.caption("Brak lub nie wczytano.")
    
        # Save full result now
        col_s1, col_s2 = st.columns([1, 1])
        with col_s1:
            if st.button("💾 Zapisz pełny wynik do Vault", key="main_save_topic_full"):
                best_title = (res.get("selected_title") or {}).get("title") or f"Temat: {res.get('topic','')}"
                best_promise = (res.get("promises") or [{}])[0].get("promise", "")
                score = int(res.get("overall_score", res.get("overall_score_base", 0) or 0))
                vault.add(
                    title=best_title,
                    promise=best_promise,
                    score=score,
                    reason="Ocena tematu - zapis pełnego wyniku",
                    tags=["topic_mode"],
                    topic=res.get("topic",""),
                    payload=res
                )
                history.add({
                    "title": best_title,
                    "promise": best_promise,
                    "final_score": score,
                    "final_score_with_bonus": score,
                    "final_verdict": "PASS" if score >= 75 else "BORDER" if score >= 60 else "FAIL",
                    "data_score": 0,
                    "llm_score": 0,
                    "risk_penalty": 0,
                    "advanced_bonus": 0,
                    "advanced_insights": {},
                    "topic_mode": True,
                    "payload": res
                })
                st.success("✅ Zapisano (Vault + Historia).")
        with col_s2:
            if st.button("🗑️ Wyczyść wynik tematu", key="main_clear_topic"):
                st.session_state.topic_result_main = None
                st.session_state.topic_job_main = None
    
    st.divider()
    st.caption("Poniżej zostawiam tryb legacy (wpisujesz tytuł i obietnicę ręcznie).")
    
    

    with st.container():

        # Input
        col_input, col_settings = st.columns([3, 1])
        
        with col_input:
            title = st.text_input(
                "📹 Tytuł filmu",
                placeholder="np. Dlaczego ta katastrofa MUSIAŁA się wydarzyć?"
            )
            
            promise = st.text_area(
                "💬 Obietnica (opcjonalnie)",
                placeholder="Co widz dowie się z filmu? Jaka jest główna wartość?",
                height=80
            )
            
            # Generuj obietnice
            if title and ADVANCED_AVAILABLE and api_key:
                if st.button("✨ Generuj propozycje obietnic"):
                    with st.spinner("Generuję..."):
                        try:
                            from openai import OpenAI
                            client = OpenAI(api_key=api_key)
                            gen = PromiseGenerator(client)
                            promises = gen.generate_from_title(title, n=5)
                            
                            st.markdown("**Propozycje obietnic:**")
                            for i, p in enumerate(promises, 1):
                                prom = p.get("promise", p) if isinstance(p, dict) else p
                                st.markdown(f"{i}. {prom}")
                        except Exception as e:
                            st.error(f"Błąd: {e}")
        
        with col_settings:
            st.markdown("**⚙️ Ustawienia**")
            
            n_judges = st.slider(
                "Liczba sędziów LLM",
                1, 3, config.get("default_judges", 2),
                help=TOOLTIPS["judges"]
            )
            
            topn_examples = st.slider(
                "Podobne przykłady",
                3, 10, config.get("default_topn", 5),
                help=TOOLTIPS["topn"]
            )
            
            optimize_variants = st.checkbox(
                "Optymalizuj warianty",
                value=config.get("default_optimize_variants", False),
                help=TOOLTIPS["optimize"]
            )
        
        # Evaluate button
        evaluate_btn = st.button("🚀 OCEŃ POMYSŁ", type="primary", use_container_width=True)
        
        if evaluate_btn:
            if not api_key:
                st.error("❌ Ustaw OpenAI API Key w panelu bocznym")
            elif not title:
                st.error("❌ Wpisz tytuł filmu")
            elif merged_df is None or len(merged_df) < 5:
                st.error("❌ Załaduj dane kanału (min. 5 filmów)")
            else:
                with st.spinner("🔄 Analizuję pomysł... (może potrwać 15-30s)"):
                    try:
                        # Get data path
                        synced_file = CHANNEL_DATA_DIR / "synced_channel_data.csv"
                        data_path = str(synced_file) if synced_file.exists() else str(MERGED_DATA_FILE)
                        
                        # Initialize evaluator
                        evaluator = get_evaluator(api_key, data_path)
                        
                        # Evaluate
                        result = evaluator.evaluate(
                            title=title,
                            promise=promise,
                            topn=topn_examples,
                            n_judges=n_judges,
                            optimize=optimize_variants
                        )
                        
                        # Add advanced analytics
                        advanced_bonus = 0
                        advanced_insights = {}
                        
                        if ADVANCED_AVAILABLE:
                            try:
                                analytics = get_advanced_analytics(data_path, api_key)
                                if analytics:
                                    adv_result = analytics.analyze_idea(title, promise)
                                    advanced_insights = adv_result.get("analyses", {})
                                    advanced_bonus = adv_result.get("total_bonus", 0)
                                    
                                    result["advanced_bonus"] = advanced_bonus
                                    result["advanced_insights"] = advanced_insights
                                    result["final_score_with_bonus"] = min(100, max(0, result["final_score"] + advanced_bonus))
                            except Exception as e:
                                st.warning(f"⚠️ Advanced analytics error: {e}")
                        
                        # Save to history
                        eval_id = history.add(result)
                        
                        # SAVE TO SESSION STATE
                        st.session_state.last_result = result
                        st.session_state.last_title = title
                        st.session_state.last_promise = promise
                        
                    except Exception as e:
                        st.error(f"❌ Błąd oceny: {e}")
                        st.exception(e)
        
        # ========== DISPLAY RESULTS FROM SESSION STATE ==========
        # Wyświetla wyniki nawet po przełączeniu zakładek
        
        if st.session_state.last_result is not None:
            result = st.session_state.last_result
            
            st.divider()
            
            # Clear button
            col_clear, col_info = st.columns([1, 4])
            with col_clear:
                if st.button("🗑️ Wyczyść wyniki"):
                    st.session_state.last_result = None
                    st.session_state.last_title = ""
                    st.session_state.last_promise = ""
                    st.rerun()
            with col_info:
                st.caption(f"Ostatnia ocena: **{st.session_state.last_title[:50]}...**")
            
            # Verdict
            render_verdict_card(result)
            
            # Diagnosis
            render_diagnosis(result)
            
            # Two columns: dimensions + advanced
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📐 Wymiary oceny")
                render_dimensions(result)
            
            with col2:
                render_advanced_insights(result)
            
            st.divider()
            
            # Improvements
            render_improvements(result)
            
            # Hook suggestion
            hook = result.get("suggested_hook_angle", "")
            emotion = result.get("target_emotion", "")
            if hook or emotion:
                st.subheader("🎣 Sugestia hooka")
                if hook:
                    st.info(f"**Kąt:** {hook}")
                if emotion:
                    st.caption(f"**Docelowa emocja:** {emotion}")
            
            st.divider()
            
            # Variants with scores
            st.subheader("📝 Warianty")
            try:
                synced_file = CHANNEL_DATA_DIR / "synced_channel_data.csv"
                data_path = str(synced_file) if synced_file.exists() else str(MERGED_DATA_FILE)
                analytics_for_scores = get_advanced_analytics(data_path, api_key) if ADVANCED_AVAILABLE else None
                render_variants_with_scores(result, analytics=analytics_for_scores)
            except:
                render_variants_with_scores(result, analytics=None)
            
            st.divider()
            
            # Copy report
            render_copy_report(result)
            
            # Save to vault option
            with st.expander("💡 Zapisz do Idea Vault"):
                vault_reason = st.text_input("Dlaczego zapisujesz na później?", key="vault_reason")
                vault_tags = st.text_input("Tagi (przecinki)", key="vault_tags")
                remind = st.selectbox("Przypomnij gdy:", 
                    ["Nie przypominaj", "30 dni", "60 dni", "90 dni", "Temat zacznie trendować"],
                    key="vault_remind"
                )
                
                if st.button("💾 Zapisz do Vault"):
                    remind_map = {
                        "30 dni": "30_days",
                        "60 dni": "60_days",
                        "90 dni": "90_days",
                        "Temat zacznie trendować": "trending",
                    }
                    tags = [t.strip() for t in vault_tags.split(",") if t.strip()]
                    vault.add(
                        title=st.session_state.last_title,
                        promise=st.session_state.last_promise,
                        score=result.get("final_score", 0),
                        reason=vault_reason,
                        tags=tags,
                        remind_when=remind_map.get(remind)
                    )
                    st.success("✅ Zapisano do Vault!")
    
# =============================================================================
# TAB: PORÓWNAJ POMYSŁY
# =============================================================================

with tab_compare:
    st.header("⚖️ Porównaj pomysły")
    st.markdown("Wrzuć 2-5 pomysłów i zobacz który najlepszy")
    
    # Input for multiple ideas
    num_ideas = st.slider("Ile pomysłów porównać?", 2, 5, 3)
    
    ideas = []
    for i in range(num_ideas):
        with st.expander(f"💡 Pomysł {i+1}", expanded=(i < 2)):
            t = st.text_input(f"Tytuł {i+1}", key=f"comp_title_{i}")
            p = st.text_area(f"Obietnica {i+1}", key=f"comp_promise_{i}", height=60)
            if t:
                ideas.append({"title": t, "promise": p})
    
    if st.button("🏆 PORÓWNAJ", type="primary", disabled=len(ideas) < 2):
        if not api_key:
            st.error("❌ Ustaw API Key")

        elif merged_df is None:
            st.error("❌ Załaduj dane kanału")
        else:
            with st.spinner("Porównuję pomysły..."):
                try:
                    data_path = str(CHANNEL_DATA_DIR / "synced_channel_data.csv") if (CHANNEL_DATA_DIR / "synced_channel_data.csv").exists() else str(MERGED_DATA_FILE)
                    analytics = get_advanced_analytics(data_path, api_key)
                    
                    if analytics:
                        comparison = analytics.compare_ideas(ideas)
                        st.session_state.comparison_result = comparison
                    else:
                        st.error("Analytics niedostępne")
                        
                except Exception as e:
                    st.error(f"Błąd: {e}")
    
    # Display comparison results from session state
    if st.session_state.comparison_result is not None:
        comparison = st.session_state.comparison_result
        
        st.divider()
        
        col_clear, col_info = st.columns([1, 4])
        with col_clear:
            if st.button("🗑️ Wyczyść porównanie"):
                st.session_state.comparison_result = None
                st.rerun()
        
        # Summary
        st.markdown(f"### {comparison.get('comparison_summary', '')}")
        
        # Ranking
        st.subheader("🏆 Ranking")
        
        for item in comparison.get("ranking", []):
            rank = item["rank"]
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"#{rank}"
            
            score = item["score"]
            color = "#2d5a3d" if score >= 65 else "#4a4000" if score >= 50 else "#4a1a1a"
            
            st.markdown(f"""
            <div style="background: {color}; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;">
                <span style="font-size: 1.5rem;">{medal}</span>
                <strong>{item['title'][:50]}...</strong>
                <span style="float: right; font-size: 1.3rem; font-weight: bold;">{score:.0f}/100</span>
                <div style="opacity: 0.7; font-size: 0.9rem;">Bonus: {item['bonus']:+d}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    
    # A/B Title Tester
    st.subheader("🔀 A/B Title Tester")
    st.markdown("Porównaj 2 wersje tytułu")
    
    col1, col2 = st.columns(2)
    with col1:
        title_a = st.text_input("Tytuł A", key="ab_title_a")
    with col2:
        title_b = st.text_input("Tytuł B", key="ab_title_b")
    
    if st.button("🔍 Porównaj tytuły") and title_a and title_b:
        if merged_df is not None and ADVANCED_AVAILABLE:
            tester = ABTitleTester(merged_df)
            result = tester.compare(title_a, title_b)
            st.session_state.ab_result = result
        else:
            st.warning("Załaduj dane kanału aby użyć A/B testera")
    
    # Display A/B results from session state
    if st.session_state.ab_result is not None:
        result = st.session_state.ab_result
        winner = result["winner"]
        
        st.divider()
        
        col_clear, _ = st.columns([1, 4])
        with col_clear:
            if st.button("🗑️ Wyczyść A/B test"):
                st.session_state.ab_result = None
                st.rerun()
        
        col1, col2 = st.columns(2)
        
        with col1:
            is_winner = winner == "A"
            st.markdown(f"""
            <div style="background: {'#2d5a3d' if is_winner else '#3a3a3a'}; padding: 1rem; border-radius: 8px;">
                <div style="font-size: 1.5rem;">{'🏆' if is_winner else ''} Tytuł A</div>
                <div style="font-size: 2rem; font-weight: bold;">{result['title_a']['score']:.0f}</div>
            </div>
            """, unsafe_allow_html=True)
            
            for factor in result['title_a']['factors']:
                st.markdown(f"- {factor}")
        
        with col2:
            is_winner = winner == "B"
            st.markdown(f"""
            <div style="background: {'#2d5a3d' if is_winner else '#3a3a3a'}; padding: 1rem; border-radius: 8px;">
                <div style="font-size: 1.5rem;">{'🏆' if is_winner else ''} Tytuł B</div>
                <div style="font-size: 2rem; font-weight: bold;">{result['title_b']['score']:.0f}</div>
            </div>
            """, unsafe_allow_html=True)
            
            for factor in result['title_b']['factors']:
                st.markdown(f"- {factor}")
        
        st.info(f"💡 {result['recommendation']}")

# =============================================================================
# TAB: NARZĘDZIA
# =============================================================================

with tab_tools:
    st.header("🛠️ Narzędzia")
    
    tool_tabs = st.tabs([
        "🎯 Oceń TEMAT",
        "📉 Dlaczego wtopa?",
        "🕳️ Content Gaps",
        "📺 Analiza serii",
        "📅 Kalendarz",
        "🔔 Trend Alerts",
        "👀 Competitor Tracker"
    ])
    
    # ==========================================================================
    # NOWY: OCEŃ TEMAT (v4 feature)
    # ==========================================================================
    with tool_tabs[0]:
        st.subheader("🎯 Oceń TEMAT na film")
        st.markdown("""
        **NOWOŚĆ w v4!** Wpisz tylko **TEMAT** (np. "Operacja Northwoods") - AI wygeneruje:
        - 📝 Tytuły z ocenami i uzasadnieniami
        - 💬 Obietnice (hooki) z ocenami
        - 📊 Analiza konkurencji na YouTube
        - 🚀 Viral Score
        - 📅 Sezonowość i świeżość tematu
        """)
        
        if not TOPIC_ANALYZER_AVAILABLE:
            st.error("❌ Moduł topic_analyzer.py niedostępny. Sprawdź czy plik istnieje.")
        else:
            # Input
            topic_input = st.text_input(
                "🎬 Temat filmu",
                placeholder="np. Operacja Northwoods, Katastrofa w Czarnobylu, Sekta Jonestown...",
                help="Wpisz sam temat - AI wygeneruje tytuły i obietnice"
            )
            
            col_set1, col_set2 = st.columns(2)
            with col_set1:
                n_titles = st.slider("Liczba tytułów", 5, 15, 10, help="Ile propozycji tytułów wygenerować")
            with col_set2:
                n_promises = st.slider("Liczba obietnic", 3, 10, 5, help="Ile propozycji hooków wygenerować")
            
            # Evaluate button
            if st.button("🚀 ANALIZUJ TEMAT", type="primary", use_container_width=True, key="analyze_topic_btn"):
                if not topic_input:
                    st.warning("⚠️ Wpisz temat")
                elif not api_key:
                    st.error("❌ Ustaw API Key OpenAI w panelu bocznym")
                else:
                    with st.spinner("🔄 Analizuję temat... (może potrwać 30-60s)"):
                        try:
                            from openai import OpenAI
                            client = OpenAI(api_key=api_key)
                            
                            # Get channel data if available
                            channel_df = merged_df if merged_df is not None else None
                            
                            # Create evaluator and evaluate
                            evaluator = get_topic_evaluator(client, channel_df)
                            result = evaluator.evaluate(topic_input, n_titles=n_titles, n_promises=n_promises)
                            
                            # Get external data
                            try:
                                wiki = get_wiki_api()
                                news = get_news_checker()
                                seasonality = get_seasonality()
                                
                                result['external_data'] = {
                                    'wikipedia': wiki.get_topic_popularity(topic_input),
                                    'news': news.get_news_score(topic_input),
                                    'seasonality': seasonality.analyze_topic_seasonality(topic_input),
                                }
                            except Exception as ext_e:
                                result['external_data'] = {'error': str(ext_e)}
                            
                            # Store in session state
                            st.session_state['topic_result'] = result
                            
                        except Exception as e:
                            st.error(f"❌ Błąd: {e}")
                            st.exception(e)
            
            # Display results
            if 'topic_result' in st.session_state and st.session_state['topic_result']:
                result = st.session_state['topic_result']
                
                st.divider()
                
                # Header with scores
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.markdown(f"### 📊 Wynik: **{result.get('topic', '')}**")
                    st.markdown(result.get('recommendation', ''))
                
                with col2:
                    overall = result.get('overall_score', 0)
                    color = "#4CAF50" if overall >= 70 else "#FFC107" if overall >= 50 else "#f44336"
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, {color}33, {color}11); padding: 1rem; border-radius: 10px; text-align: center;">
                        <div style="font-size: 2rem; font-weight: bold; color: {color};">{overall}/100</div>
                        <div style="font-size: 0.8rem; color: #888;">Overall Score</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    viral = result.get('viral_score', {}).get('viral_score', 0)
                    vcolor = "#4CAF50" if viral >= 70 else "#FFC107" if viral >= 50 else "#f44336"
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, {vcolor}33, {vcolor}11); padding: 1rem; border-radius: 10px; text-align: center;">
                        <div style="font-size: 1.5rem; font-weight: bold; color: {vcolor};">{viral}/100</div>
                        <div style="font-size: 0.8rem; color: #888;">Viral Score</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.divider()
                
                # Titles section
                st.subheader("📝 Wygenerowane tytuły")
                st.caption("Kliknij na tytuł aby zobaczyć uzasadnienie. Najlepszy jest na górze.")
                
                titles = result.get('titles', [])
                selected_title_idx = st.session_state.get('selected_topic_title_idx', 0)
                
                for i, t in enumerate(titles):
                    col_t, col_s = st.columns([5, 1])
                    
                    with col_t:
                        is_selected = (i == selected_title_idx)
                        btn_label = f"{'✅ ' if is_selected else ''}{t['title']}"
                        if st.button(btn_label, key=f"topic_title_{i}", use_container_width=True):
                            st.session_state['selected_topic_title_idx'] = i
                            st.rerun()
                    
                    with col_s:
                        score = t.get('score', 0)
                        sc = "#4CAF50" if score >= 70 else "#FFC107" if score >= 50 else "#f44336"
                        st.markdown(f"<span style='color:{sc};font-weight:bold;font-size:1.2rem;'>{score}</span>", unsafe_allow_html=True)
                    
                    # Show reasoning if selected
                    if i == selected_title_idx:
                        with st.container():
                            st.markdown(f"""
                            <div style="background: #1a1a1a; padding: 0.8rem; border-radius: 6px; margin: 0.5rem 0; font-size: 0.85rem;">
                                <strong>Styl:</strong> {t.get('style', 'N/A')} | <strong>Źródło:</strong> {t.get('source', 'N/A')}<br>
                                <strong>Uzasadnienie:</strong> {t.get('reasoning', 'Brak')}
                            </div>
                            """, unsafe_allow_html=True)
                
                st.divider()
                
                # Promises section
                st.subheader("💬 Obietnice (hooki)")
                
                promises = result.get('promises', [])
                for i, p in enumerate(promises):
                    col_p, col_ps = st.columns([5, 1])
                    
                    with col_p:
                        st.markdown(f"**{i+1}.** {p.get('promise', '')}")
                        if p.get('reasoning'):
                            st.caption(p['reasoning'])
                    
                    with col_ps:
                        pscore = p.get('score', 0)
                        pc = "#4CAF50" if pscore >= 70 else "#FFC107" if pscore >= 50 else "#f44336"
                        st.markdown(f"<span style='color:{pc};font-weight:bold;'>{pscore}</span>", unsafe_allow_html=True)
                
                st.divider()
                
                # Additional analysis tabs
                st.subheader("🔍 Szczegółowe analizy")
                
                analysis_tabs = st.tabs(["📊 Konkurencja", "🚀 Viral", "🎬 Podobne", "🌐 External"])
                
                # Competition tab
                with analysis_tabs[0]:
                    comp = result.get('competition', {})
                    
                    if comp.get('error'):
                        st.warning(f"⚠️ {comp.get('error')}")
                    else:
                        sat = comp.get('saturation', 'UNKNOWN')
                        sat_colors = {'LOW': '#4CAF50', 'MEDIUM': '#FFC107', 'HIGH': '#f44336'}
                        sat_color = sat_colors.get(sat, '#888')
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**Nasycenie rynku:** <span style='color:{sat_color};font-weight:bold;'>{sat}</span>", unsafe_allow_html=True)
                            st.metric("Szansa", f"{comp.get('opportunity_score', 0)}/100")
                        with col2:
                            st.info(comp.get('recommendation', ''))
                        
                        top_vids = comp.get('top_videos', [])
                        if top_vids:
                            st.markdown("**Top konkurencyjne filmy:**")
                            for v in top_vids[:5]:
                                st.caption(f"• {v.get('title', '')[:50]}... - **{v.get('views', 0):,}** views ({v.get('channel', '')})")
                
                # Viral tab
                with analysis_tabs[1]:
                    viral_data = result.get('viral_score', {})
                    
                    st.metric("Viral Score", f"{viral_data.get('viral_score', 0)}/100")
                    st.markdown(f"**{viral_data.get('verdict', '')}**")
                    
                    factors = viral_data.get('factors', [])
                    if factors:
                        st.markdown("**Czynniki viralowe:**")
                        for f in factors:
                            st.caption(f"• {f.get('factor', '')}: {f.get('bonus', '')} - {', '.join(f.get('found', []))}")
                    
                    rec = viral_data.get('recommendation', '')
                    if rec:
                        st.info(rec)
                
                # Similar videos tab
                with analysis_tabs[2]:
                    similar = result.get('similar_hits', [])
                    
                    if similar:
                        st.markdown("**Podobne filmy na Twoim kanale:**")
                        for s in similar:
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"**{s.get('title', '')}**")
                                st.caption(s.get('insight', ''))
                            with col2:
                                st.metric("Views", f"{s.get('views', 0):,}")
                    else:
                        st.info("Brak podobnych filmów na kanale. Wgraj dane kanału aby zobaczyć porównanie.")
                
                # External data tab
                with analysis_tabs[3]:
                    ext = result.get('external_data', {})
                    
                    if ext.get('error'):
                        st.warning(f"⚠️ Dane zewnętrzne niedostępne: {ext.get('error')}")
                    else:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            wiki = ext.get('wikipedia', {})
                            st.markdown("**📚 Wikipedia**")
                            st.metric("Pageviews (30d)", f"{wiki.get('total_pageviews_30d', 0):,}")
                            trend = wiki.get('trend', 'UNKNOWN')
                            trend_emoji = '📈' if trend == 'RISING' else '📉' if trend == 'FALLING' else '➡️'
                            st.caption(f"Trend: {trend_emoji} {trend}")
                        
                        with col2:
                            news = ext.get('news', {})
                            st.markdown("**📰 News**")
                            st.metric("News Score", f"{news.get('news_score', 0)}/100")
                            st.caption(news.get('recommendation', '')[:80])
                            
                            headlines = news.get('recent_headlines', [])
                            if headlines:
                                with st.expander("Headlines"):
                                    for h in headlines[:3]:
                                        title = h['title'] if isinstance(h, dict) else h
                                        st.caption(f"• {title[:60]}...")
                        
                        with col3:
                            season = ext.get('seasonality', {})
                            st.markdown("**📅 Sezonowość**")
                            if season.get('has_seasonality'):
                                st.markdown(f"Peak: **{season.get('peak_month_name', '?')}**")
                                st.caption(season.get('reason', ''))
                            else:
                                st.caption("Temat evergreen")
                            st.info(season.get('recommendation', '')[:100])
                
                st.divider()
                
                # Save to vault
                col_save1, col_save2 = st.columns([2, 1])
                
                with col_save1:
                    topic_notes = st.text_area("Notatki (opcjonalne)", height=80, key="topic_save_notes")
                    
                    if st.button("💾 Zapisz do Vault", type="primary", key="save_topic_to_vault"):
                        vault = get_vault()
                        
                        # Prepare data for vault
                        vault_entry = {
                            'type': 'topic_evaluation',
                            'topic': result.get('topic', ''),
                            'best_title': result.get('selected_title', {}).get('title', ''),
                            'overall_score': result.get('overall_score', 0),
                            'viral_score': result.get('viral_score', {}).get('viral_score', 0),
                            'competition': result.get('competition', {}).get('saturation', ''),
                            'full_result': result,
                        }
                        
                        vault.add(
                            title=result.get('selected_title', {}).get('title', f"Temat: {result.get('topic', '')}"),
                            promise=result.get('promises', [{}])[0].get('promise', ''),
                            score=result.get('overall_score', 0),
                            notes=topic_notes,
                            topic=topic_input,
                            payload=vault_entry
                        )
                        st.success("✅ Zapisano do Vault!")
                
                with col_save2:
                    if st.button("🗑️ Wyczyść wyniki", key="clear_topic_results"):
                        st.session_state['topic_result'] = None
                        st.session_state['selected_topic_title_idx'] = 0
                        st.rerun()
    
    # --- WTOPA ANALYZER ---
    with tool_tabs[1]:
        st.subheader("📚 Dlaczego wtopa")
        st.markdown("Analizuj dlaczego film mógł mieć słabe wyniki i dostaj konkretne sugestie naprawy")

        if merged_df is not None and "title" in merged_df.columns:
            st.markdown("### 🎥 Wybierz film z kanału (auto-uzupełnianie)")
            df_w = merged_df.copy()
            if "views" not in df_w.columns:
                df_w["views"] = 0
            df_w = df_w.sort_values(by="views", ascending=False)
            opts = list(df_w.index)

            def _fmt(i):
                row = df_w.loc[i]
                return f"{str(row.get('title',''))[:80]} | {int(row.get('views',0) or 0):,} views"

            pick = st.selectbox("Film", options=opts, format_func=_fmt, key="wtopa_pick")
            if st.button("Załaduj dane filmu", key="wtopa_load_btn"):
                row = df_w.loc[pick]
                st.session_state["wtopa_title"] = str(row.get("title",""))
                st.session_state["wtopa_views"] = int(row.get("views",0) or 0)

                # heurystyka retencji (jeśli mamy)
                ret = None
                for col in ["retention", "avg_view_percentage", "average_view_percentage", "avg_view_duration_ratio"]:
                    if col in df_w.columns:
                        try:
                            ret = float(row.get(col))
                            if col == "avg_view_duration_ratio" and ret <= 1.5:
                                ret = ret * 100.0
                        except Exception:
                            ret = None
                        if ret is not None and ret > 0:
                            break
                if ret is not None:
                    st.session_state["wtopa_ret"] = float(max(0.0, min(100.0, ret)))

                st.rerun()

            st.divider()

        wtopa_title = st.text_input("Tytuł filmu", key="wtopa_title")
        wtopa_views = st.number_input("Wyświetlenia", min_value=0, value=0, key="wtopa_views")
        wtopa_retention = st.slider("Retencja (%)", 0.0, 100.0, 40.0, key="wtopa_ret")

        with st.expander("🎯 Kontekst", expanded=False):
            wtopa_target = st.text_input("Co miało się wydarzyć? (Twoje założenie)", placeholder="np. 50k views, 45% retencji")
            wtopa_notes = st.text_area("Dodatkowe notatki", height=80, placeholder="np. zmiana stylu, temat był ryzykowny, thumbnail A/B itp.")

        if st.button("🔎 Przeanalizuj", use_container_width=True, key="run_wtopa"):
            if not ADVANCED_AVAILABLE:
                st.error("Moduł advanced_analytics niedostępny.")
            else:
                with st.spinner("Analizuję..."):
                    analyzer = WhyItFloppedAnalyzer(merged_df, openai_client if api_key else None)
                    result = analyzer.analyze(wtopa_title, int(wtopa_views), float(wtopa_retention), {
                        "target": wtopa_target,
                        "notes": wtopa_notes
                    })

                st.session_state.last_wtopa = result
                st.success("✅ Gotowe!")

        if "last_wtopa" in st.session_state and st.session_state.last_wtopa:
            result = st.session_state.last_wtopa

            st.markdown("### 📉 Diagnoza")
            st.write(result.get("summary", ""))

            st.markdown("### 🛠️ Najbardziej prawdopodobne powody")
            for item in result.get("reasons", []):
                st.markdown(f"- **{item.get('reason','')}**")
                if item.get("detail"):
                    st.caption(item.get("detail"))

            st.markdown("### ✅ Co poprawić (konkret)")
            for fix in result.get("fixes", []):
                st.markdown(f"- {fix}")

            with st.expander("📦 Pełny wynik", expanded=False):
                st.json(result)

    with tool_tabs[2]:
        st.subheader("🕳️ Content Gap Finder")
        st.markdown("Tematy popularne w niszy, których jeszcze nie robiłeś")
        
        if st.button("🔍 Znajdź luki") and merged_df is not None and ADVANCED_AVAILABLE:
            finder = ContentGapFinder(merged_df)
            gaps = finder.find_gaps()
            
            st.divider()
            
            for gap in gaps[:15]:
                covered = gap["covered"]
                icon = "🟡" if covered else "🟢"
                
                st.markdown(f"""
                {icon} **{gap['topic'].title()}** - {gap['recommendation']}
                """)
                
                if not covered:
                    with st.expander("💡 Pomysły na filmy"):
                        suggestions = finder.suggest_ideas(gap['topic'])
                        for s in suggestions:
                            st.markdown(f"- {s}")
        elif merged_df is None:
            st.warning("Załaduj dane kanału")
    
    # --- SERIE ---
    with tool_tabs[3]:
        st.subheader("📺 Analiza serii")
        st.markdown("Które serie/tematy działają najlepiej")

        # Ręczne serie: wybierasz dokładnie filmy, które należą do serii
        if merged_df is not None and ADVANCED_AVAILABLE:
            series_mgr = get_series_manager()
            st.markdown("### ✍️ Ręczne serie (Twoja definicja)")

            # Create new series
            cs1, cs2 = st.columns([3, 1])
            with cs1:
                new_series_name = st.text_input("Nazwa serii", placeholder="np. 'Polskie Tajemnice'", key="new_series_name")
            with cs2:
                if st.button("Utwórz", key="create_series_btn") and new_series_name:
                    series_mgr.create_series(new_series_name.strip())
                    st.success("✅ Utworzono serię.")
                    st.rerun()

            series_list = series_mgr.list_series()
            if series_list:
                # pick series
                series_names = [s["name"] for s in series_list]
                chosen = st.selectbox("Wybierz serię", series_names, key="series_pick")
                series_obj = next((s for s in series_list if s["name"] == chosen), series_list[0])

                # video selection
                df_small = merged_df.copy()
                if "title" not in df_small.columns:
                    df_small["title"] = ""
                video_map = {f"{row.get('title','')[:70]} ({row.get('views',0):,} views)": row.get("video_id","") for _, row in df_small.iterrows() if row.get("video_id")}
                current_ids = series_mgr.get_series_videos(series_obj["id"])
                current_labels = [k for k, vid in video_map.items() if vid in current_ids]

                selected_labels = st.multiselect(
                    "Filmy w serii",
                    options=list(video_map.keys()),
                    default=current_labels,
                    key="series_video_select"
                )

                if st.button("💾 Zapisz serię", key="save_series_videos_btn"):
                    vids = [video_map[l] for l in selected_labels if l in video_map]
                    series_mgr.set_series_videos(series_obj["id"], vids)
                    st.success("✅ Zapisano.")
                    st.rerun()

                # show stats
                vids = series_mgr.get_series_videos(series_obj["id"])
                if vids:
                    sdf = merged_df[merged_df["video_id"].isin(vids)].copy()
                    episodes = len(sdf)
                    avg_views = int(sdf["views"].mean()) if "views" in sdf.columns and episodes else 0
                    total_views = int(sdf["views"].sum()) if "views" in sdf.columns and episodes else 0
                    st.info(f"Seria **{series_obj['name']}** | odcinki: **{episodes}** | avg views: **{avg_views:,}** | total: **{total_views:,}**")
                else:
                    st.caption("Seria nie ma jeszcze przypisanych filmów.")
            else:
                st.caption("Nie masz jeszcze ręcznych serii.")

        st.divider()
        
        if st.button("📊 Analizuj serie") and merged_df is not None and ADVANCED_AVAILABLE:
            analyzer = SeriesAnalyzer(merged_df)
            series = analyzer.get_series_performance()
            recs = analyzer.get_recommendations()
            
            st.divider()
            
            if recs:
                for rec in recs:
                    st.info(rec)
            
            if series:
                st.markdown("### 📈 Performance serii")
                
                for s in series[:10]:
                    trend_color = "#2d5a3d" if "Rośnie" in s['trend'] else "#5a2a2a" if "Spada" in s['trend'] else "#3a3a3a"
                    
                    st.markdown(f"""
                    <div style="background: {trend_color}; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;">
                        <strong>{s['name']}</strong> ({s['episodes']} odcinków)
                        <span style="float: right;">{s['trend']}</span>
                        <div>Avg views: {s['avg_views']:,}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Nie wykryto serii w danych")
        elif merged_df is None:
            st.warning("Załaduj dane kanału")
    
    # --- KALENDARZ ---
    with tool_tabs[4]:
        st.subheader("📅 Optymalny kalendarz publikacji")
        
        if st.button("📅 Generuj kalendarz") and merged_df is not None and ADVANCED_AVAILABLE:
            data_path = str(CHANNEL_DATA_DIR / "synced_channel_data.csv") if (CHANNEL_DATA_DIR / "synced_channel_data.csv").exists() else str(MERGED_DATA_FILE)
            analytics = get_advanced_analytics(data_path, api_key)
            
            if analytics:
                calendar = analytics.get_optimal_upload_calendar()
                
                st.divider()
                
                if calendar.get("best_day"):
                    st.success(f"🎯 Najlepszy dzień publikacji: **{calendar['best_day']}**")
                
                if calendar.get("recommendations"):
                    st.markdown("### 📋 Rekomendacje")
                    for rec in calendar["recommendations"]:
                        icon = "📅" if rec["type"] == "day" else "🎬"
                        st.markdown(f"{icon} **{rec['recommendation']}** - {rec['reason']}")
        elif merged_df is None:
            st.warning("Załaduj dane kanału")
    

        st.divider()
        st.subheader("🗓️ Plan contentowy na kilka filmów do przodu")
        st.caption("Generator tematów na kolejne tygodnie: miesza Twoje hity, luki contentowe i trendy. Wersja beta, ale już użyteczna.")

        pc1, pc2, pc3 = st.columns([1, 1, 2])
        with pc1:
            plan_weeks = st.slider("Tygodnie", 1, 8, 4, key="plan_weeks")
        with pc2:
            plan_per_week = st.slider("Filmy/tydzień", 1, 4, 1, key="plan_per_week")
        with pc3:
            plan_focus = st.text_input("Fokus (opcjonalnie): np. katastrofy, sekty, UFO", value="", key="plan_focus")

        if st.button("🧠 Wygeneruj plan", key="gen_content_plan"):
            if not api_key:
                st.error("❌ Brak OpenAI API Key. Dodaj klucz w sidebarze.")
            elif merged_df is None:
                st.warning("⚠️ Najpierw wczytaj dane kanału (zakładka: Dane).")
            else:
                # 1) Zbierz kontekst: hity, nisza, konkurencja, trendy
                niche_keywords = config.get("niche_keywords", [])
                comp_mgr = get_competitor_manager()
                competitors = comp_mgr.list_all()

                # Top hity z kanału (dla kotwic)
                top_hits = []
                try:
                    top_df = merged_df.sort_values("views", ascending=False).head(15)
                    top_hits = top_df["title"].astype(str).tolist()
                except Exception:
                    top_hits = []

                # Trendy (jeśli da się pobrać)
                trending_now = []
                try:
                    trend_disc = get_trend_discovery()
                    trending_now = trend_disc.find_trending_topics(niche_keywords, limit=10)
                except Exception:
                    trending_now = []

                n_items = int(plan_weeks * plan_per_week)
                n_items = max(1, min(12, n_items))

                # 2) LLM: zaproponuj tematy i kąty
                plan_items = []
                try:
                    client = OpenAI(api_key=api_key)

                    prompt = f"""Jesteś strategiem YouTube dla kanału dark documentary po polsku.
Zadanie: zaproponuj {n_items} tematów na kolejne tygodnie.
Każdy temat ma być krótki i konkretny (np. "Operacja Northwoods", "Katastrofa w Smoleńsku", "Czarna Plaga 1258 Samalas").
Dodaj też kąt (angle) i 1-2 zdania uzasadnienia (why) pod moją niszę.

Kontekst:
- Niszowe słowa klucze: {", ".join(niche_keywords) if niche_keywords else "brak"}
- Fokus użytkownika: {plan_focus or "brak"}
- Największe hity kanału (tytuły): {top_hits[:8] if top_hits else "brak danych"}
- Trendy teraz (jeśli są): {trending_now if trending_now else "brak"}

Zwróć WYŁĄCZNIE poprawny JSON w formacie:
{{
  "plan": [
    {{"week": 1, "topic": "...", "angle": "...", "why": "..."}}
  ]
}}
Bez komentarzy i bez markdown.
"""

                    resp = client.chat.completions.create(
                        model=config.get("openai_model", "gpt-4o-mini"),
                        messages=[
                            {"role": "system", "content": "Zwracaj tylko JSON. Bez markdown."},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.6,
                    )

                    raw = resp.choices[0].message.content.strip()
                    data = json.loads(raw)
                    plan_items = data.get("plan", [])[:n_items]
                except Exception as e:
                    st.warning(f"Nie udało się wygenerować planu z LLM: {e}")
                    plan_items = []

                if not plan_items:
                    st.info("Brak planu do pokazania. Spróbuj ponownie lub ustaw inny fokus.")
                else:
                    # 3) Szybka ocena każdego tematu (score + tytuł + viral)
                    eval_results = []
                    if TOPIC_ANALYZER_AVAILABLE:
                        try:
                            client = OpenAI(api_key=api_key)
                            topic_eval = get_topic_evaluator(client, merged_df)
                            for item in plan_items:
                                topic = (item.get("topic") or "").strip()
                                if not topic:
                                    continue
                                res = topic_eval.evaluate(topic, n_titles=6, n_promises=3)
                                eval_results.append({
                                    "week": item.get("week", 1),
                                    "topic": topic,
                                    "angle": item.get("angle", ""),
                                    "why": item.get("why", ""),
                                    "score": res.get("overall_score", 0),
                                    "recommendation": res.get("recommendation", ""),
                                    "best_title": (res.get("selected_title") or {}).get("title", topic),
                                    "viral": (res.get("viral_score") or {}).get("viral_score", 0),
                                    "opportunity": (res.get("competition") or {}).get("opportunity_score", 50),
                                    "payload": res,
                                })
                        except Exception as e:
                            st.warning(f"Ocena tematów nie zadziałała: {e}")

                    # 4) Prezentacja
                    by_week = {}
                    for item in plan_items:
                        w = int(item.get("week", 1))
                        by_week.setdefault(w, []).append(item)

                    st.markdown("#### Propozycje (plan)")
                    for w in sorted(by_week.keys()):
                        st.markdown(f"### Tydzień {w}")
                        week_items = [x for x in plan_items if int(x.get("week", 1)) == w]
                        for idx, item in enumerate(week_items, start=1):
                            topic = item.get("topic", "")
                            angle = item.get("angle", "")
                            why = item.get("why", "")

                            scored = next((r for r in eval_results if r["topic"] == topic), None)
                            title_line = scored["best_title"] if scored else topic
                            score_line = scored["score"] if scored else None

                            label = f"{idx}. {title_line}"
                            if score_line is not None:
                                label = f"{idx}. {title_line}  |  Score: {score_line}/100"

                            with st.expander(label, expanded=False):
                                st.markdown(f"**Temat:** {topic}")
                                if angle:
                                    st.markdown(f"**Kąt:** {angle}")
                                if why:
                                    st.markdown(f"**Dlaczego:** {why}")
                                if scored:
                                    st.markdown(scored.get("recommendation", ""))
                                    st.markdown(f"**Viral:** {scored.get('viral', 0)}/100  |  **Opportunity:** {scored.get('opportunity', 0)}/100")

                                    # szybkie top 3 obietnice
                                    promises = (scored["payload"].get("promises") or [])[:3]
                                    if promises:
                                        st.markdown("**Obietnice (top):**")
                                        for p in promises:
                                            st.markdown(f"- {p.get('promise','')} ({p.get('score',0)}/100)")

                                    if st.button("💾 Zapisz do Idea Vault", key=f"save_plan_{w}_{idx}"):
                                        best_promise = ""
                                        if promises:
                                            best_promise = promises[0].get("promise", "")
                                        vault.add(
                                            title=scored.get("best_title", topic),
                                            topic=topic,
                                            promise=best_promise,
                                            score=scored.get("score", 0),
                                            tags=["plan"],
                                            status="waiting",
                                            payload=scored.get("payload", {})
                                        )
                                        st.success("Zapisano do Idea Vault.")

                    st.divider()
                    if eval_results and st.button("💾 Zapisz CAŁY plan do Idea Vault", key="save_full_plan"):
                        saved = 0
                        for r in eval_results:
                            promises = (r["payload"].get("promises") or [])[:1]
                            best_promise = promises[0].get("promise", "") if promises else ""
                            ok = vault.add(
                                title=r.get("best_title", r["topic"]),
                                topic=r["topic"],
                                promise=best_promise,
                                score=r.get("score", 0),
                                tags=["plan"],
                                status="waiting",
                                payload=r.get("payload", {})
                            )
                            if ok:
                                saved += 1
                        st.success(f"Zapisano {saved} pomysłów.")
        # --- TREND ALERTS ---
    with tool_tabs[5]:
        st.subheader("🔔 Trend Alerts")
        st.markdown("Monitoruj tematy i otrzymuj powiadomienia gdy zaczną trendować")

        # ---------------------------------------------------------------------
        # Trend Discovery: co trenduje teraz w Twojej niszy
        # ---------------------------------------------------------------------
        st.markdown("### 📈 Trend Discovery: co trenduje teraz w niszy")
        st.caption("Klikasz, a apka zbiera propozycje tematów (powiązane frazy) i sprawdza je w Google Trends.")

        default_seeds = config.get("niche_keywords", ["tajemnice", "zagadki", "spiski", "ufo", "katastrofy"])
        seeds_text = st.text_area(
            "Słowa-klucze niszy (po przecinku)",
            value=", ".join(default_seeds),
            height=70,
            key="niche_seeds_text"
        )
        cseed1, cseed2 = st.columns([1, 1])
        with cseed1:
            if st.button("💾 Zapisz słowa-klucze", key="save_niche_seeds"):
                config.set("niche_keywords", [s.strip() for s in seeds_text.split(",") if s.strip()])
                config.save()
                st.success("✅ Zapisano.")
        with cseed2:
            discover_now = st.button("🔎 Szukaj trendów w niszy", key="discover_niche_trends")

        if discover_now:
            with st.spinner("Szukam trendów w niszy..."):
                try:
                    discovery = get_trend_discovery()
                    candidates = []
                    for seed in [s.strip() for s in seeds_text.split(",") if s.strip()][:8]:
                        rel = discovery.discover_related_topics(seed, limit=6) or []
                        for item in rel:
                            term = (item.get("topic") or "").strip()
                            if term:
                                candidates.append({
                                    "topic": term,
                                    "source_seed": seed,
                                    "relevance": item.get("relevance", 0)
                                })
                    # dedupe by topic, keep max relevance
                    dedup = {}
                    for c in candidates:
                        t = c["topic"].lower()
                        if t not in dedup or c.get("relevance",0) > dedup[t].get("relevance",0):
                            dedup[t] = c
                    top = sorted(dedup.values(), key=lambda x: x.get("relevance",0), reverse=True)[:20]

                    trend_map = {}
                    if ADVANCED_AVAILABLE and top:
                        ta = TrendsAnalyzer()
                        # batch in groups of 5
                        batch = []
                        for item in top:
                            batch.append(item["topic"])
                            if len(batch) == 5:
                                tr = ta.check_trend(batch)
                                if tr.get("status") == "OK":
                                    for kw, det in (tr.get("details") or {}).items():
                                        trend_map[kw] = det
                                batch = []
                        if batch:
                            tr = ta.check_trend(batch)
                            if tr.get("status") == "OK":
                                for kw, det in (tr.get("details") or {}).items():
                                    trend_map[kw] = det

                    rows = []
                    for item in top:
                        t = item["topic"]
                        det = trend_map.get(t, {})
                        overall = det.get("overall") or {}
                        rows.append({
                            "Temat": t,
                            "Seed": item["source_seed"],
                            "Relevance": item.get("relevance", 0),
                            "Trend score": overall.get("score", ""),
                            "Trend verdict": overall.get("verdict", ""),
                            "Trend msg": overall.get("message", "")
                        })
                    if rows:
                        st.dataframe(rows, use_container_width=True)
                    else:
                        st.info("Brak wyników. Dodaj inne seed keywords.")
                except Exception as e:
                    st.error(f"Błąd trend discovery: {e}")

        st.divider()

        # ---------------------------------------------------------------------
        # Monitorowane tematy (alerty)
        # ---------------------------------------------------------------------
        st.markdown("### ➕ Dodaj temat do monitorowania")
        col1, col2 = st.columns([3, 1])
        with col1:
            new_topic = st.text_input("Temat do monitorowania", placeholder="np. 'Cicada 3301'")
        with col2:
            threshold = st.number_input("Próg", min_value=0, max_value=100, value=70)

        if st.button("Dodaj", key="add_trend_alert"):
            if new_topic:
                alerts.add_topic(new_topic, threshold)
                st.success(f"✅ Dodano: {new_topic}")
                st.rerun()

        st.divider()

        st.markdown("### 📋 Monitorowane tematy")
        all_alerts = alerts.get_all()

        if all_alerts:
            for alert in all_alerts:
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                with col1:
                    st.markdown(f"**{alert['topic']}**")
                    if alert.get("last_check"):
                        st.caption(f"Ostatnio: {alert.get('last_interest', 0)}/100 | {alert.get('last_check', '')[:10]}")
                with col2:
                    st.write(f"Próg: {alert['threshold']}")
                with col3:
                    status = "🔥" if alert.get("is_trending") else "⏸️"
                    st.write(status)
                with col4:
                    if st.button("🗑️", key=f"del_alert_{alert['id']}"):
                        alerts.remove_topic(alert["id"])
                        st.rerun()

            if st.button("🔄 Sprawdź trendy teraz", key="check_trend_alerts_now"):
                if ADVANCED_AVAILABLE:
                    trends_analyzer = TrendsAnalyzer()
                    for alert in all_alerts:
                        result = trends_analyzer.check_trend([alert["topic"]])
                        if result.get("status") == "OK":
                            det = (result.get("details") or {}).get(alert["topic"], {})
                            overall = det.get("overall") or {}
                            current = overall.get("score", 0)  # fallback
                            is_trending = overall.get("verdict") in ["UP", "PEAK"]
                            alerts.update_check(alert["id"], current, is_trending)
                    st.success("✅ Sprawdzono!")
                    st.rerun()
        else:
            st.info("Brak monitorowanych tematów")

    # =============================================================================
    # TAB: ANALYTICS (DASHBOARD)
    # =============================================================================

    with tab_analytics:
        st.header("📊 Dashboard kanału")
    
        if merged_df is None:
            st.warning("⚠️ Załaduj dane kanału aby zobaczyć dashboard")
        else:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
        
            with col1:
                st.metric("📹 Filmów", len(merged_df))
        
            with col2:
                if "views" in merged_df.columns:
                    total_views = merged_df["views"].sum()
                    st.metric("👁️ Total views", f"{total_views:,.0f}")
        
            with col3:
                if "views" in merged_df.columns:
                    avg_views = merged_df["views"].mean()
                    st.metric("📊 Avg views", f"{avg_views:,.0f}")
        
            with col4:
                if "retention" in merged_df.columns and merged_df["retention"].notna().any():
                    avg_ret = merged_df["retention"].mean()
                    st.metric("⏱️ Avg retention", f"{avg_ret:.1f}%")
        
            st.divider()
        
            # Charts
            chart_col1, chart_col2 = st.columns(2)
        
            with chart_col1:
                st.subheader("📈 Views over time")
            
                if "published_at" in merged_df.columns or "publishedAt" in merged_df.columns:
                    date_col = "published_at" if "published_at" in merged_df.columns else "publishedAt"
                    df_chart = merged_df.copy()
                    df_chart["date"] = pd.to_datetime(df_chart[date_col], errors="coerce")
                    df_chart = df_chart.dropna(subset=["date"]).sort_values("date")
                
                    if len(df_chart) > 0 and "views" in df_chart.columns:
                        fig = px.line(df_chart, x="date", y="views", 
                                      title="Views per video")
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
        
            with chart_col2:
                st.subheader("🎯 Hit Rate")
            
                if "label" in merged_df.columns:
                    label_counts = merged_df["label"].value_counts()
                
                    fig = px.pie(values=label_counts.values, names=label_counts.index,
                                title="PASS / BORDER / FAIL",
                                color_discrete_map={"PASS": "#28a745", "BORDER": "#ffc107", "FAIL": "#dc3545"})
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
        
            st.divider()
        
            # DNA Analysis
            if ADVANCED_AVAILABLE:
                st.subheader("🧬 DNA Twojego kanału")
            
                if st.button("🔍 Analizuj DNA"):
                    data_path = str(CHANNEL_DATA_DIR / "synced_channel_data.csv") if (CHANNEL_DATA_DIR / "synced_channel_data.csv").exists() else str(MERGED_DATA_FILE)
                    analytics = get_advanced_analytics(data_path, api_key)
                
                    if analytics:
                        dna = analytics.get_packaging_dna()
                    
                        if "error" not in dna:
                            # Recommendations
                            if dna.get("recommendations"):
                                st.markdown("### 💡 Kluczowe wnioski")
                                for rec in dna["recommendations"]:
                                    st.info(rec)
                        
                            col1, col2 = st.columns(2)
                        
                            with col1:
                                # Trigger words
                                st.markdown("### ✅ Twoje trigger words")
                                triggers = dna.get("word_triggers", {}).get("trigger_words", [])
                                for t in triggers[:10]:
                                    st.markdown(f"- **{t['word']}** (lift: {t['lift']}x)")
                        
                            with col2:
                                # Avoid words
                                st.markdown("### ❌ Unikaj")
                                avoid = dna.get("word_triggers", {}).get("avoid_words", [])
                                for t in avoid[:10]:
                                    st.markdown(f"- {t['word']} (lift: {t['lift']}x)")
        
            st.divider()
        
            # Tracking accuracy
            st.subheader("🎯 Tracking Accuracy")
        
            tracking_stats = history.get_tracking_stats()
        
            if tracking_stats.get("total_tracked", 0) > 0:
                col1, col2, col3 = st.columns(3)
            
                with col1:
                    st.metric("Tracked filmów", tracking_stats["total_tracked"])
                with col2:
                    st.metric("Avg accuracy", f"{tracking_stats['avg_accuracy']}%")
                with col3:
                    st.metric("PASS accuracy", f"{tracking_stats['pass_accuracy']}%")
            else:
                st.info("Brak danych tracking. Po publikacji filmu dodaj rzeczywiste views w zakładce Historia.")
    # --- COMPETITOR TRACKER ---
    with tool_tabs[6]:
        st.subheader("👀 Competitor Tracker")
        st.caption("Dodaj kanały konkurencji i podejrzyj ich ostatnie uploady (z YouTube API lub fallbackiem).")

        if not COMPETITOR_TRACKER_AVAILABLE:
            st.error("Brak modułu competitor_tracker.py lub brak zależności. Sprawdź pliki.")
        else:
            comp_mgr = get_competitor_manager()
            competitors = comp_mgr.list_all()

            st.markdown("### ➕ Dodaj konkurenta")
            c1, c2, c3 = st.columns([2, 2, 1])
            with c1:
                comp_name = st.text_input("Nazwa (opcjonalnie)", placeholder="np. Kanał X", key="comp_add_name")
            with c2:
                comp_channel_id = st.text_input("Channel ID", placeholder="UCxxxxxxxxxxxxxxxx", key="comp_add_channel_id")
            with c3:
                if st.button("Dodaj", key="comp_add_btn"):
                    cid = comp_mgr.add(comp_name, comp_channel_id)
                    if cid:
                        st.success("✅ Dodano.")
                        st.rerun()
                    else:
                        st.warning("Podaj poprawny Channel ID.")

            st.divider()
            st.markdown("### 📋 Lista konkurentów")
            if competitors:
                for c in competitors:
                    cc1, cc2, cc3, cc4 = st.columns([2, 2, 2, 1])
                    with cc1:
                        st.markdown(f"**{c.get('name','')}**")
                        st.caption(c.get("channel_id",""))
                    with cc2:
                        st.caption(c.get("notes",""))
                    with cc3:
                        st.caption(f"Dodano: {c.get('added','')[:10]}")
                    with cc4:
                        if st.button("🗑️", key=f"comp_del_{c.get('id','')}"):
                            comp_mgr.remove(c.get("id",""))
                            st.rerun()
            else:
                st.info("Brak konkurentów. Dodaj pierwszego powyżej.")

            st.divider()
            st.markdown("### 🛰️ Ostatnie uploady")
            days = st.slider("Z jakiego okresu", 3, 60, 14, key="comp_days")
            max_per = st.slider("Max filmów na kanał", 1, 20, 8, key="comp_max_per")
            fetch_btn = st.button("Pobierz ostatnie filmy", key="comp_fetch_btn")

            if fetch_btn:
                yt_sync = get_youtube_sync()
                yt_client = None
                if GOOGLE_API_AVAILABLE and yt_sync.has_credentials():
                    ok, msg = yt_sync.authenticate()
                    if ok:
                        yt_client = yt_sync.youtube
                    else:
                        st.warning(msg)

                tracker = get_competitor_tracker(yt_client)
                uploads = tracker.fetch_recent_uploads(competitors, days=days, max_per_channel=max_per)

                # filter out errors separately
                errs = [u for u in uploads if u.get("error")]
                vids = [u for u in uploads if u.get("video_id")]

                if errs:
                    with st.expander("⚠️ Błędy", expanded=False):
                        st.json(errs)

                if vids:
                    # Sort by publishedAt if present
                    def _ts(x):
                        return x.get("publishedAt") or x.get("publishedTime") or ""
                    vids_sorted = sorted(vids, key=_ts, reverse=True)
                    st.dataframe(vids_sorted, use_container_width=True)
                else:
                    st.info("Brak filmów do wyświetlenia (albo brak danych z API).")
# =============================================================================
# TAB: HISTORIA
# =============================================================================

with tab_history:
    st.header("📜 Historia ocen")
    
    all_history = history.get_all()
    
    if not all_history:
        st.info("Brak ocen w historii. Oceń pierwszy pomysł!")
    else:
        # Search
        search_query = st.text_input("🔍 Szukaj w historii", key="history_search")
        
        filtered = all_history
        if search_query:
            filtered = history.search(search_query)
        
        st.caption(f"Pokazuję {len(filtered)} z {len(all_history)} ocen")
        
        for entry in filtered[:50]:
            verdict = entry.get("final_verdict", "BORDER")
            score = entry.get("final_score_with_bonus", entry.get("final_score", 0))
            
            color = "#2d5a3d" if verdict == "PASS" else "#4a4000" if verdict == "BORDER" else "#4a1a1a"
            emoji = "🟢" if verdict == "PASS" else "🟡" if verdict == "BORDER" else "🔴"
            
            with st.expander(f"{emoji} {entry.get('title', 'Bez tytułu')[:60]}... | {score:.0f}/100 | {entry.get('timestamp', '')[:10]}"):
                
                

                # Actions
                a1, a2, a3 = st.columns([1, 1, 2])
                with a1:
                    if st.button("🗑️ Usuń", key=f"hist_del_{entry.get('id','')}"):
                        history.delete(entry.get("id",""))
                        st.rerun()
                with a2:
                    if st.button("💾 Do Vault", key=f"hist_to_vault_{entry.get('id','')}"):
                        payload = entry.get("payload") or entry
                        vault.add(
                            title=entry.get("title",""),
                            promise=entry.get("promise",""),
                            score=int(entry.get("final_score_with_bonus", entry.get("final_score", 0)) or 0),
                            reason="Zapis z historii",
                            tags=["history_import"],
                            topic=(payload.get("topic") if isinstance(payload, dict) else ""),
                            payload=payload
                        )
                        st.success("✅ Zapisano do Vault.")

                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown(f"**Tytuł:** {entry.get('title', '')}")
                    st.markdown(f"**Obietnica:** {entry.get('promise', '') or 'Brak'}")
                    if entry.get("why"):
                        st.markdown(f"**Diagnoza:** {entry.get('why', '')}")

                    with st.expander("📦 Pełny wynik (payload)", expanded=False):
                        st.json(entry.get("payload") or {})

                    improvements = entry.get("improvements") or []
                    if improvements:
                        st.markdown("**Ulepszenia:**")
                        for imp in improvements[:3]:
                            st.markdown(f"- {imp}")

                with col2:
                    st.metric("Score", f"{score:.0f}/100")
                    st.metric("Data", f"{entry.get('data_score', 0):.0f}")
                    st.metric("LLM", f"{entry.get('llm_score', 0):.0f}")

                    # Tracking section
                    st.divider()
                    if entry.get("published"):
                        st.success(f"✅ Published: {entry.get('actual_views', 0):,} views")
                        if entry.get("prediction_accuracy"):
                            st.metric("Accuracy", f"{entry['prediction_accuracy']}%")
                    else:
                        st.markdown("**📊 Dodaj tracking:**")
                        actual_views = st.number_input(
                            "Rzeczywiste views",
                            min_value=0,
                            key=f"track_views_{entry['id']}"
                        )
                        if st.button("💾 Zapisz", key=f"track_btn_{entry['id']}"):
                            history.update_tracking(entry["id"], actual_views)
                            st.success("✅ Zapisano!")
                            st.rerun()
        # Export
        st.divider()
        if st.button("📥 Eksportuj do CSV"):
            csv = history.export_to_csv()
            st.download_button(
                "⬇️ Pobierz CSV",
                csv,
                "evaluation_history.csv",
                "text/csv"
            )

# =============================================================================
# TAB: IDEA VAULT
# =============================================================================

with tab_vault:
    st.header("💡 Idea Vault")
    st.markdown("Pomysły zapisane na później")
    
    # Reminders
    reminders = vault.check_reminders()
    if reminders:
        st.warning(f"🔔 Masz {len(reminders)} przypomnienie(a)!")
        
        for r in reminders:
            st.markdown(f"""
            <div class="warning-box">
                <strong>{r['title']}</strong><br>
                {r['reminder_reason']}
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
    
    # Filter
    status_filter = st.selectbox(
        "Filtruj:",
        ["Wszystkie", "Oczekujące", "Użyte", "Odrzucone"],
        key="vault_filter"
    )
    
    status_map = {
        "Wszystkie": None,
        "Oczekujące": "waiting",
        "Użyte": "used",
        "Odrzucone": "discarded"
    }
    
    all_ideas = vault.get_all(status=status_map.get(status_filter))
    
    if not all_ideas:
        st.info("Vault jest pusty. Zapisz pomysły podczas oceny!")
    else:
        for idea in all_ideas:
            status = idea.get("status", "waiting")
            status_emoji = "⏳" if status == "waiting" else "✅" if status == "used" else "❌"
            
            with st.expander(f"{status_emoji} {idea.get('title', '')[:50]}... | Score: {idea.get('score', 0)}"):
                st.markdown(f"**Tytuł:** {idea.get('title', '')}")
                st.markdown(f"**Obietnica:** {idea.get('promise', '') or 'Brak'}")
                st.markdown(f"**Powód zapisania:** {idea.get('reason', '') or 'Nie podano'}")

                with st.expander("📦 Pełny wynik (payload)", expanded=False):
                    st.json(idea.get("payload") or {})
                st.markdown(f"**Tagi:** {', '.join(idea.get('tags', [])) or 'Brak'}")
                st.markdown(f"**Dodano:** {idea.get('added', '')[:10]}")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("✅ Oznacz jako użyte", key=f"vault_used_{idea['id']}"):
                        vault.update_status(idea["id"], "used")
                        st.rerun()
                
                with col2:
                    if st.button("❌ Odrzuć", key=f"vault_discard_{idea['id']}"):
                        vault.update_status(idea["id"], "discarded")
                        st.rerun()
                
                with col3:
                    if st.button("🗑️ Usuń", key=f"vault_delete_{idea['id']}"):
                        vault.remove(idea["id"])
                        st.rerun()

# =============================================================================
# TAB: DANE
# =============================================================================

with tab_data:
    st.header("📁 Zarządzanie danymi")
    
    # YouTube Sync section
    st.subheader("📺 YouTube Sync")
    
    yt_sync = get_youtube_sync()
    
    if GOOGLE_API_AVAILABLE and yt_sync.has_credentials():
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Pełna synchronizacja", use_container_width=True):
                with st.spinner("Synchronizuję..."):
                    success, msg = yt_sync.authenticate()
                    if success:
                        df, sync_msg = yt_sync.sync_all(include_analytics=True, include_transcripts=False)
                        st.success(sync_msg)
                        st.rerun()
                    else:
                        st.error(msg)
        
        with col2:
            if st.button("📝 Sync z transkryptami", use_container_width=True):
                with st.spinner("Synchronizuję (z transkryptami - może potrwać)..."):
                    success, msg = yt_sync.authenticate()
                    if success:
                        df, sync_msg = yt_sync.sync_all(include_analytics=True, include_transcripts=True)
                        st.success(sync_msg)
                        st.rerun()
                    else:
                        st.error(msg)
    else:
        st.info("Skonfiguruj YouTube API aby używać sync")
        with st.expander("📖 Instrukcje"):
            st.markdown(yt_sync.setup_instructions())
    
    st.divider()
    
    # Manual CSV upload
    st.subheader("📤 Ręczny upload CSV")
    
    uploaded_files = st.file_uploader(
        "Przeciągnij pliki CSV",
        type=["csv"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        for file in uploaded_files:
            df = pd.read_csv(file)
            st.write(f"**{file.name}**: {len(df)} wierszy, kolumny: {', '.join(df.columns[:5])}")
        
        if st.button("💾 Zapisz i połącz dane"):
            CHANNEL_DATA_DIR.mkdir(exist_ok=True)
            
            all_dfs = []
            for file in uploaded_files:
                df = pd.read_csv(file)
                all_dfs.append(df)
            
            if all_dfs:
                # Simple merge by stacking
                merged = pd.concat(all_dfs, ignore_index=True)
                
                # Remove duplicates by title
                if "title" in merged.columns:
                    merged = merged.drop_duplicates(subset=["title"], keep="last")
                
                merged.to_csv(MERGED_DATA_FILE, index=False)
                st.success(f"✅ Zapisano {len(merged)} wierszy do {MERGED_DATA_FILE}")
                st.rerun()
    
    st.divider()
    
    # Current data preview
    st.subheader("📊 Aktualne dane")
    
    if merged_df is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Filmów", len(merged_df))
        with col2:
            if "views" in merged_df.columns:
                st.metric("Avg views", f"{merged_df['views'].mean():,.0f}")
        with col3:
            if "retention" in merged_df.columns and merged_df["retention"].notna().any():
                st.metric("Avg retention", f"{merged_df['retention'].mean():.1f}%")
        with col4:
            if "label" in merged_df.columns:
                pass_rate = (merged_df["label"] == "PASS").mean() * 100
                st.metric("PASS rate", f"{pass_rate:.0f}%")
        
        # Preview table
        display_cols = [c for c in ["title", "views", "retention", "label"] if c in merged_df.columns]
        if display_cols:
            st.dataframe(merged_df[display_cols].head(20), use_container_width=True)
        
        # Delete button
        if st.button("🗑️ Usuń wszystkie dane", type="secondary"):
            if MERGED_DATA_FILE.exists():
                MERGED_DATA_FILE.unlink()
            synced = CHANNEL_DATA_DIR / "synced_channel_data.csv"
            if synced.exists():
                synced.unlink()
            st.success("Dane usunięte")
            st.rerun()
    else:
        st.info("Brak danych. Użyj YouTube Sync lub wgraj CSV.")

# =============================================================================
# FOOTER
# =============================================================================

st.divider()
st.caption("YT Idea Evaluator Pro v3 | Made for Dawid 🎬")
