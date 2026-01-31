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
import hashlib
import calendar
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
    get_competitor_manager
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
        PromiseGenerator, ContentGapFinder,
        WtopaAnalyzer
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


# =============================================================================
# STYLE CSS (DARK MODE FRIENDLY)
# =============================================================================

st.markdown("""
<style>
/* Dark mode friendly colors */
.stAlert > div {
    color: inherit !important;
}

/* Layout helpers */
.section-card {
    background: #141414;
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
}
.section-card h4 {
    margin: 0 0 8px 0;
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
CACHE_FILE = Path("./app_data/cache_store.json")
CACHE_VERSION = "v1"

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

def validate_channel_dataframe(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Waliduje podstawowy format danych kanału."""
    required = {"title", "views"}
    recommended = {"retention", "label", "published_at"}
    issues = {
        "missing_required": [],
        "missing_recommended": [],
        "warnings": []
    }

    cols = set(df.columns)
    missing_required = sorted(required - cols)
    missing_recommended = sorted(recommended - cols)

    if missing_required:
        issues["missing_required"] = missing_required
    if missing_recommended:
        issues["missing_recommended"] = missing_recommended

    if "views" in df.columns:
        views_numeric = pd.to_numeric(df["views"], errors="coerce")
        invalid_views = views_numeric.isna().mean() * 100
        if invalid_views > 0:
            issues["warnings"].append(f"Kolumna 'views' ma {invalid_views:.1f}% niepoprawnych wartości.")

    if "retention" in df.columns:
        retention_numeric = pd.to_numeric(df["retention"], errors="coerce")
        invalid_retention = retention_numeric.isna().mean() * 100
        if invalid_retention > 0:
            issues["warnings"].append(f"Kolumna 'retention' ma {invalid_retention:.1f}% niepoprawnych wartości.")
        if retention_numeric.dropna().between(0, 100).mean() < 0.9:
            issues["warnings"].append("Wartości 'retention' powinny być w zakresie 0-100%.")

    if "title" in df.columns:
        empty_titles = df["title"].isna().mean() * 100
        if empty_titles > 0:
            issues["warnings"].append(f"Kolumna 'title' ma {empty_titles:.1f}% pustych wartości.")

    return issues

def _load_cache_store() -> Dict:
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_cache_store(cache: Dict) -> None:
    CACHE_FILE.parent.mkdir(exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False, default=str)

def _cache_get(namespace: str, key: str, ttl_seconds: int = None):
    cache = _load_cache_store()
    bucket = cache.get(namespace, {})
    entry = bucket.get(key)
    if not entry:
        return None
    if ttl_seconds is not None:
        ts = entry.get("ts")
        if not ts:
            return None
        try:
            age = (datetime.now() - datetime.fromisoformat(ts)).total_seconds()
        except Exception:
            return None
        if age > ttl_seconds:
            return None
    return entry.get("value")

def _cache_set(namespace: str, key: str, value) -> None:
    cache = _load_cache_store()
    cache.setdefault(namespace, {})[key] = {
        "ts": datetime.now().isoformat(),
        "value": value,
        "version": CACHE_VERSION
    }
    _save_cache_store(cache)

def _make_cache_key(payload: Dict) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def log_diagnostic(message: str, level: str = "info") -> None:
    st.session_state.setdefault("diagnostics", [])
    st.session_state["diagnostics"].append({
        "timestamp": datetime.now().isoformat(),
        "level": level,
        "message": message
    })

def get_cache_ttl(name: str, default_seconds: int) -> int:
    ttl_map = st.session_state.get("cache_ttl", {})
    return int(ttl_map.get(name, default_seconds))

def record_llm_call(kind: str, cached: bool = False) -> None:
    st.session_state.setdefault("llm_stats", {"calls": 0, "cached_hits": 0, "by_kind": {}})
    stats = st.session_state["llm_stats"]
    if cached:
        stats["cached_hits"] += 1
        return
    stats["calls"] += 1
    stats["by_kind"][kind] = stats["by_kind"].get(kind, 0) + 1

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

    # === STATUS MODUŁÓW ===
    st.subheader("🧩 Status modułów")
    module_status = {
        "Topic Analyzer": TOPIC_ANALYZER_AVAILABLE,
        "Advanced Analytics": ADVANCED_AVAILABLE,
        "Competitor Tracker": COMPETITOR_TRACKER_AVAILABLE,
        "YouTube API": GOOGLE_API_AVAILABLE,
    }
    for name, available in module_status.items():
        st.caption(f"{'✅' if available else '⚠️'} {name}")
    if not api_key:
        st.info("Tryb bez API: generowanie tytułów/obietnic działa z szablonów.")
    
    st.divider()
    
    # === YOUTUBE SYNC ===
    st.subheader("📺 YouTube Sync")
    
    yt_sync = get_youtube_sync()
    last_sync = yt_sync.get_last_sync_time()
    yt_api_key = config.get_youtube_api_key()
    yt_channel_id = config.get("channel_id", "")

    yt_api_key_input = st.text_input(
        "YouTube API Key (public)",
        value=yt_api_key,
        type="password",
        help="Klucz do publicznych zapytań YouTube Data API"
    )
    yt_channel_input = st.text_input(
        "Channel ID (public)",
        value=yt_channel_id,
        help="ID kanału do publicznego syncu (bez OAuth)"
    )
    if yt_api_key_input != yt_api_key:
        config.set_youtube_api_key(yt_api_key_input)
    if yt_channel_input != yt_channel_id:
        config.set("channel_id", yt_channel_input)
    yt_sync.set_api_key(yt_api_key_input)
    yt_sync.set_channel_id(yt_channel_input)
    
    if last_sync:
        st.caption(f"Ostatnia sync: {last_sync}")
    
    if not GOOGLE_API_AVAILABLE:
        st.warning("⚠️ Zainstaluj: `pip install google-api-python-client google-auth-oauthlib`")
    elif yt_sync.has_credentials():
        if st.button("🔄 Synchronizuj dane (OAuth)", use_container_width=True):
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
    else:
        if yt_sync.ensure_public_client() and yt_channel_input:
            st.info("Tryb publiczny: tylko dane z YouTube Data API (bez Analytics).")
            if st.button("🔄 Synchronizuj dane (public)", use_container_width=True):
                with st.spinner("Pobieram dane publiczne..."):
                    df, sync_msg = yt_sync.sync_all(include_analytics=False, include_transcripts=False)
                    st.success(sync_msg)
                    st.rerun()
        with st.expander("📖 Jak skonfigurować?"):
            st.markdown(yt_sync.setup_instructions())
    
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
    vault_ideas = len(vault.get_all(status="new"))
    tracked = len([e for e in history.get_all() if e.get("published")])
    
    st.metric("Ocen w historii", total_evals)
    st.metric("Pomysłów w Vault", vault_ideas)
    st.metric("Tracked filmów", tracked)

# =============================================================================
# MAIN TABS
# =============================================================================

tab_evaluate, tab_tools, tab_analytics, tab_history, tab_vault, tab_data, tab_diag = st.tabs([
    "🎯 Oceń pomysł",
    "🛠️ Narzędzia",
    "📊 Analytics",
    "📜 Historia",
    "💡 Idea Vault",
    "📁 Dane",
    "🧪 Diagnostyka"
])

# =============================================================================
# TAB: OCEŃ POMYSŁ
# =============================================================================

with tab_evaluate:
    st.header("🧭 Topic Workspace")
    st.caption("Jedno wejście: temat. Jeden wynik: tytuły, obietnice, trendy, konkurencja, podobne hity, viral score i timeline.")
    
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
            inc_competition = st.checkbox(
                "Konkurencja (YouTube)",
                value=True,
                key="main_inc_comp",
                help="Sprawdza nasycenie rynku i liczbę podobnych treści w YouTube."
            )
            inc_similar = st.checkbox(
                "Podobne hity (kanał)",
                value=True,
                key="main_inc_sim",
                help="Porównuje temat do Twoich historycznych hitów na kanale."
            )
        with c2:
            inc_trends = st.checkbox(
                "Google Trends",
                value=True,
                key="main_inc_trends",
                help="Ocena trendu wyszukiwań w Google dla tematu."
            )
            inc_external = st.checkbox(
                "Źródła zewnętrzne (Wiki/News)",
                value=True,
                key="main_inc_ext",
                help="Sprawdza świeżość i zainteresowanie z Wikipedii i newsów."
            )
        with c3:
            inc_viral = st.checkbox(
                "Viral Score",
                value=True,
                key="main_inc_viral",
                help="Predykcja wiralowości na podstawie tytułu/tematu."
            )
            inc_timeline = st.checkbox(
                "Performance Timeline",
                value=True,
                key="main_inc_timeline",
                help="Prognoza wyświetleń po 1/7/30 dniach."
            )

    with st.expander("🧪 Batch: oceń wiele tematów naraz", expanded=False):
        batch_raw = st.text_area(
            "Wklej listę tematów (1 temat = 1 linia)",
            height=120,
            key="batch_topics_input"
        )
        batch_limit = st.slider("Limit tematów", 1, 10, 5, key="batch_topics_limit")
        if st.button("⚡ Szybka ocena listy", key="batch_topics_run"):
            topics = [t.strip() for t in (batch_raw or "").splitlines() if t.strip()]
            topics = topics[:batch_limit]
            results = []
            for t in topics:
                job = {
                    "topic": t,
                    "n_titles": 3,
                    "n_promises": 3,
                    "api_key": api_key,
                    "inc_competition": inc_competition,
                    "inc_similar": inc_similar,
                    "inc_trends": inc_trends,
                    "inc_external": inc_external,
                    "inc_viral": inc_viral,
                    "inc_timeline": inc_timeline,
                    "result": {"topic": t, "timestamp": datetime.now().isoformat()}
                }
                for stg in [0, 1, 2, 5]:
                    job = _topic_stage_run(stg, job)
                res = job.get("result", {})
                results.append({
                    "topic": t,
                    "score": res.get("overall_score", res.get("overall_score_base", 0)),
                    "best_title": (res.get("selected_title") or {}).get("title", ""),
                    "recommendation": res.get("recommendation", "")
                })
            if results:
                st.session_state["batch_topic_results"] = results
            else:
                st.info("Brak tematów do oceny.")
        if st.session_state.get("batch_topic_results"):
            st.markdown("#### Wyniki batch")
            st.dataframe(st.session_state["batch_topic_results"], use_container_width=True)
    
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
    
        # Stage 0: trends + external
        if stage == 0:
            if job.get("inc_trends", True) and ADVANCED_AVAILABLE:
                cache_key = _make_cache_key({"topic": topic, "stage": "trends", "v": CACHE_VERSION})
                cached_trend = _cache_get("trends", cache_key, ttl_seconds=get_cache_ttl("trends", 6 * 3600))
                if cached_trend is None:
                    try:
                        ta = TrendsAnalyzer()
                        cached_trend = ta.check_trend([topic])
                        _cache_set("trends", cache_key, cached_trend)
                    except Exception as e:
                        cached_trend = {"status": "ERROR", "message": str(e)}
                        log_diagnostic(f"Trends error: {e}", "error")
                res["trends"] = cached_trend
            if job.get("inc_external", True):
                cache_key = _make_cache_key({"topic": topic, "stage": "external", "v": CACHE_VERSION})
                cached_external = _cache_get("external", cache_key, ttl_seconds=get_cache_ttl("external", 12 * 3600))
                if cached_external is None:
                    try:
                        wiki_api = get_wiki_api()
                        news = get_news_checker()
                        season = get_seasonality()
                        discovery = get_trend_discovery()
                        cached_external = {
                            "wikipedia": wiki_api.search_articles(topic, limit=3),
                            "wikipedia_stats": wiki_api.get_topic_popularity(topic),
                            "news": news.get_news_score(topic),
                            "seasonality": season.analyze_topic_seasonality(topic),
                            "trend_discovery": discovery.analyze_topic_complete(topic),
                        }
                        _cache_set("external", cache_key, cached_external)
                    except Exception as e:
                        cached_external = {"error": str(e)}
                        log_diagnostic(f"External sources error: {e}", "error")
                res["external_data"] = cached_external
            job["result"] = res
            job["stage_done"] = 0
            return job

        # Stage 1: competition
        if stage == 1:
            if job.get("inc_competition", True):
                cache_key = _make_cache_key({"topic": topic, "stage": "competition", "v": CACHE_VERSION})
                cached_comp = _cache_get("competition", cache_key, ttl_seconds=get_cache_ttl("competition", 6 * 3600))
                if cached_comp is None:
                    cached_comp = evaluator.competitor_analyzer.analyze(topic)
                    _cache_set("competition", cache_key, cached_comp)
                res["competition"] = cached_comp
            job["result"] = res
            job["stage_done"] = 1
            return job

        # Stage 2: similar hits
        if stage == 2:
            if job.get("inc_similar", True) and evaluator.similar_finder:
                cache_key = _make_cache_key({"topic": topic, "stage": "similar", "v": CACHE_VERSION})
                cached_similar = _cache_get("similar_hits", cache_key, ttl_seconds=get_cache_ttl("similar_hits", 6 * 3600))
                if cached_similar is None:
                    cached_similar = evaluator.similar_finder.find(topic, topic)
                    _cache_set("similar_hits", cache_key, cached_similar)
                res["similar_hits"] = cached_similar
            job["result"] = res
            job["stage_done"] = 2
            return job

        # Stage 3: titles
        if stage == 3:
            use_ai = bool(client)
            cache_key = _make_cache_key({
                "topic": topic,
                "stage": "titles",
                "n_titles": job.get("n_titles", 6),
                "use_ai": use_ai,
                "v": CACHE_VERSION
            })
            cached_titles = _cache_get("llm_titles", cache_key, ttl_seconds=get_cache_ttl("llm_titles", 24 * 3600))
            if cached_titles is None:
                cached_titles = evaluator.title_generator.generate(
                    topic,
                    n=job.get("n_titles", 6),
                    use_ai=use_ai
                )
                _cache_set("llm_titles", cache_key, cached_titles)
                record_llm_call("titles")
            else:
                record_llm_call("titles", cached=True)
            res["titles"] = cached_titles
            if res.get("titles"):
                res["selected_title"] = res["titles"][0]
            job["result"] = res
            job["stage_done"] = 3
            return job

        # Stage 4: promises
        if stage == 4:
            use_ai = bool(client)
            best_title = (res.get("selected_title") or {}).get("title") or topic
            cache_key = _make_cache_key({
                "topic": topic,
                "title": best_title,
                "stage": "promises",
                "n_promises": job.get("n_promises", 6),
                "use_ai": use_ai,
                "v": CACHE_VERSION
            })
            cached_promises = _cache_get("llm_promises", cache_key, ttl_seconds=get_cache_ttl("llm_promises", 24 * 3600))
            if cached_promises is None:
                cached_promises = evaluator.promise_generator.generate(
                    best_title,
                    topic,
                    n=job.get("n_promises", 6),
                    use_ai=use_ai
                )
                _cache_set("llm_promises", cache_key, cached_promises)
                record_llm_call("promises")
            else:
                record_llm_call("promises", cached=True)
            res["promises"] = cached_promises
            job["result"] = res
            job["stage_done"] = 4
            return job

        # Stage 5: summary
        if stage == 5:
            best_title = (res.get("selected_title") or {}).get("title") or topic
            if job.get("inc_viral", True):
                res["viral_score"] = evaluator.viral_predictor.predict(best_title, topic, res.get("competition", {}))

            title_score = int(res.get("selected_title", {}).get("score", 50))
            competition_score = int(res.get("competition", {}).get("opportunity_score", 50))
            viral_score = int(res.get("viral_score", {}).get("viral_score", 50))
            base_overall = int(title_score * 0.35 + competition_score * 0.30 + viral_score * 0.35)

            trend_bonus = 0
            if res.get("trends", {}).get("overall", {}).get("score") is not None:
                try:
                    overall_trend_score = int(res["trends"]["overall"]["score"])
                    trend_bonus = max(-10, min(10, int((overall_trend_score - 50) / 5)))
                except Exception:
                    trend_bonus = 0

            similar_bonus = 0
            if res.get("similar_hits"):
                try:
                    views_vals = [h.get("views", 0) for h in res["similar_hits"] if isinstance(h.get("views", 0), (int, float))]
                    views_vals = [v for v in views_vals if v > 0]
                    if views_vals:
                        median_views = int(sorted(views_vals)[len(views_vals)//2])
                        if merged_df is not None and "views" in merged_df.columns:
                            chan_med = int(merged_df["views"].median())
                            if median_views > chan_med:
                                similar_bonus = 5
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
            job["stage_done"] = 5
            job["done"] = True
            return job
    
        return job
    
    stage_names = [
        "Trendy i źródła",
        "Konkurencja",
        "Podobne hity",
        "Generacja tytułów",
        "Generacja obietnic",
        "Podsumowanie"
    ]

    stage_done = None
    if st.session_state.topic_job_main:
        stage_done = st.session_state.topic_job_main.get("stage_done")
    st.caption(f"Etapy: {' → '.join(stage_names)}")
    if stage_done is not None:
        st.caption(f"Aktualny etap: {stage_names[min(stage_done, len(stage_names)-1)]}")

    # Action buttons
    b1, b2, b3, b4 = st.columns([1, 1, 1, 1])
    with b1:
        start_step = st.button("🧩 Start krokowo", use_container_width=True, key="main_topic_step_start")
    with b2:
        full_run = st.button("🚀 Pełna ocena", use_container_width=True, key="main_topic_full")
    with b3:
        cont_step = st.button("➡️ Następny etap", use_container_width=True, key="main_topic_step_continue")
    with b4:
        stop_and_save = st.button("⏹️ STOP + Zapisz częściowo", use_container_width=True, key="main_topic_step_stop")

    st.markdown("### ⚡ Tryb jedno‑kliknięcie")
    if st.button("✅ Oceń + wygeneruj + zapisz do Vault", use_container_width=True, key="main_one_click"):
        if not topic_input_main:
            st.warning("Wpisz temat.")
        elif not api_key:
            st.error("❌ Brak OpenAI API Key. Dodaj klucz w sidebarze.")
        elif merged_df is None:
            st.warning("⚠️ Najpierw wczytaj dane kanału (zakładka: Dane).")
        else:
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
            with st.spinner("Oceniam temat i zapisuję..."):
                for stg in [0, 1, 2, 3, 4, 5]:
                    job = _topic_stage_run(stg, job)
            st.session_state.topic_job_main = job
            st.session_state.topic_result_main = job.get("result")
            res = st.session_state.topic_result_main or {}
            best_title = (res.get("selected_title") or {}).get("title") or f"Temat: {res.get('topic','')}"
            best_promise = (res.get("promises") or [{}])[0].get("promise", "")
            score = int(res.get("overall_score", res.get("overall_score_base", 0) or 0))
            vault.add(
                title=best_title,
                promise=best_promise,
                score=score,
                reason="Ocena tematu - zapis z trybu jedno‑kliknięcia",
                tags=["topic_mode", "one_click"],
                topic=res.get("topic",""),
                payload=res,
                status="new"
            )
            st.success("✅ Gotowe — zapisane w Vault.")

    b7, b8 = st.columns([1, 1])
    with b7:
        quick_preview = st.button("⚡ Szybka ocena (bez LLM)", use_container_width=True, key="main_quick_preview")
    with b8:
        clear_preview = st.button("🧹 Wyczyść podgląd", use_container_width=True, key="main_clear_preview")

    b5, b6 = st.columns([1, 1])
    with b5:
        if st.button("🔄 Wygeneruj nowe tytuły", use_container_width=True, key="main_regen_titles"):
            if st.session_state.topic_job_main:
                st.session_state.topic_job_main["stage_done"] = 2
                st.session_state.topic_result_main = st.session_state.topic_job_main.get("result")
            st.session_state.topic_job_main = st.session_state.topic_job_main or {
                "topic": topic_input_main.strip(),
                "n_titles": n_titles_main,
                "n_promises": n_promises_main,
                "api_key": api_key,
                "result": {"topic": topic_input_main.strip(), "timestamp": datetime.now().isoformat()}
            }
            st.session_state.topic_job_main["n_titles"] = n_titles_main
            st.session_state.topic_job_main = _topic_stage_run(3, st.session_state.topic_job_main)
            st.session_state.topic_result_main = st.session_state.topic_job_main.get("result")
    with b6:
        if st.button("🔄 Wygeneruj nowe obietnice", use_container_width=True, key="main_regen_promises"):
            if st.session_state.topic_job_main:
                st.session_state.topic_job_main["stage_done"] = 3
                st.session_state.topic_result_main = st.session_state.topic_job_main.get("result")
            st.session_state.topic_job_main = st.session_state.topic_job_main or {
                "topic": topic_input_main.strip(),
                "n_titles": n_titles_main,
                "n_promises": n_promises_main,
                "api_key": api_key,
                "result": {"topic": topic_input_main.strip(), "timestamp": datetime.now().isoformat()}
            }
            st.session_state.topic_job_main["n_promises"] = n_promises_main
            st.session_state.topic_job_main = _topic_stage_run(4, st.session_state.topic_job_main)
            st.session_state.topic_result_main = st.session_state.topic_job_main.get("result")
    
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

    if quick_preview and topic_input_main:
        job = {
            "topic": topic_input_main.strip(),
            "n_titles": 0,
            "n_promises": 0,
            "api_key": "",
            "inc_competition": inc_competition,
            "inc_similar": inc_similar,
            "inc_trends": inc_trends,
            "inc_external": inc_external,
            "inc_viral": inc_viral,
            "inc_timeline": inc_timeline,
            "result": {"topic": topic_input_main.strip(), "timestamp": datetime.now().isoformat()}
        }
        for stg in [0, 1, 2, 5]:
            job = _topic_stage_run(stg, job)
        st.session_state.topic_job_main = job
        st.session_state.topic_result_main = job.get("result")
        st.info("✅ Szybka ocena zakończona (bez LLM).")

    if clear_preview:
        st.session_state.topic_result_main = None
        st.session_state.topic_job_main = None
    
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
            for stg in [0, 1, 2, 3, 4, 5]:
                job = _topic_stage_run(stg, job)
        st.session_state.topic_job_main = job
        st.session_state.topic_result_main = job.get("result")
    
    # Continue step-by-step
    if cont_step and st.session_state.topic_job_main:
        job = st.session_state.topic_job_main
        stage = int(job.get("stage_done", 0)) + 1
        stage = min(stage, 5)
        with st.spinner(f"Etap {stage+1}/6..."):
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
        stage_note = ""
        if st.session_state.topic_job_main:
            stage_note = f"Etap: {st.session_state.topic_job_main.get('stage_done')}"
    
        vault.add(
            title=best_title,
            promise=best_promise,
            score=score,
            reason="Ocena tematu (zatrzymana) - zapis pełnego wyniku",
            tags=["topic_mode", "partial"],
            topic=res.get("topic",""),
            payload=res,
            status="new",
            notes=stage_note
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
            "payload": res,
            "tags": ["topic_mode", "partial"],
            "status": "new"
        })
    
        st.success("✅ Zapisano (Vault + Historia).")
        st.session_state.topic_job_main = None
    
    # Display topic result
    if st.session_state.topic_result_main:
        res = st.session_state.topic_result_main
    
        st.divider()
        st.subheader("Wynik oceny tematu")

        score_val = int(res.get("overall_score", res.get("overall_score_base", 0) or 0))
        viral_val = int(res.get("viral_score", {}).get("viral_score", 0))
        score_color = "#4CAF50" if score_val >= 70 else "#FFC107" if score_val >= 50 else "#f44336"
        viral_color = "#4CAF50" if viral_val >= 70 else "#FFC107" if viral_val >= 50 else "#f44336"
        verdict = "PASS" if score_val >= 75 else "BORDER" if score_val >= 60 else "FAIL"
        verdict_color = "#2d5a3d" if verdict == "PASS" else "#4a4000" if verdict == "BORDER" else "#4a1a1a"

        col_score, col_viral = st.columns([1, 1])
        with col_score:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {score_color}33, {score_color}11); padding: 1rem; border-radius: 10px; text-align: center;">
                <div style="font-size: 2rem; font-weight: bold; color: {score_color};">{score_val}/100</div>
                <div style="font-size: 0.8rem; color: #888;">Overall Score</div>
            </div>
            """, unsafe_allow_html=True)
        with col_viral:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {viral_color}33, {viral_color}11); padding: 1rem; border-radius: 10px; text-align: center;">
                <div style="font-size: 1.5rem; font-weight: bold; color: {viral_color};">{viral_val}/100</div>
                <div style="font-size: 0.8rem; color: #888;">Viral Score</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown(
            f"<div class='section-card' style='background:{verdict_color};'>"
            f"<h4>Werdykt: {verdict}</h4>"
            f"<div>Rekomendacja: {'publikuj' if verdict == 'PASS' else 'popraw i testuj' if verdict == 'BORDER' else 'przemyśl temat'}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        if res.get("recommendation"):
            st.info(res.get("recommendation"))
        if not api_key:
            st.warning("Tryb bez API: tytuły/obietnice z szablonów, bez LLM.")

        comp = res.get("competition", {})
        trend = res.get("trends", {}).get("overall", {})

        st.markdown("#### ✅ Najważniejsze wnioski")
        takeaway_cols = st.columns(4)
        with takeaway_cols[0]:
            st.metric("Overall", f"{score_val}/100")
        with takeaway_cols[1]:
            st.metric("Viral", f"{viral_val}/100")
        with takeaway_cols[2]:
            st.metric("Trend", f"{trend.get('score', 0) if trend else 0}")
        with takeaway_cols[3]:
            st.metric("Opportunity", f"{comp.get('opportunity_score', 0)}")

        # Timeline
        if res.get("performance_timeline"):
            tl = res["performance_timeline"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Predykcja views 1 dzień", f"{tl.get('day_1',0):,}")
            c2.metric("Predykcja views 7 dni", f"{tl.get('day_7',0):,}")
            c3.metric("Predykcja views 30 dni", f"{tl.get('day_30',0):,}")
        else:
            st.warning("Brak danych do timeline — używam trybu uproszczonego.")

        # Quick signals
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Konkurencja (opportunity)", f"{comp.get('opportunity_score', 0)}")
        with c2:
            st.metric("Viral Score", f"{viral_val}")
        with c3:
            st.metric("Trend Score", f"{trend.get('score', 0) if trend else 0}")

        if merged_df is None or "views" not in merged_df.columns:
            st.info("⚠️ Brak danych kanału (views). Wynik jest w trybie uproszczonym.")

        st.markdown("### Dlaczego taki wynik?")
        base_overall = int(res.get("overall_score_base", 0))
        trend_bonus = int(res.get("trend_bonus", 0))
        similar_bonus = int(res.get("similar_bonus", 0))
        total_calc = max(0, min(100, base_overall + trend_bonus + similar_bonus))

        explainer_cols = st.columns(4)
        with explainer_cols[0]:
            st.metric("Base score", f"{base_overall}/100")
        with explainer_cols[1]:
            st.metric("Trend bonus", f"{trend_bonus:+d}")
        with explainer_cols[2]:
            st.metric("Similar bonus", f"{similar_bonus:+d}")
        with explainer_cols[3]:
            st.metric("Final", f"{total_calc}/100")

        st.progress(min(1.0, total_calc / 100.0))
        if res.get("competition"):
            st.caption(f"Konkurencja: {res['competition'].get('saturation','?')} | Opportunity: {res['competition'].get('opportunity_score','?')}")

        st.markdown("#### 🧾 Szybkie uzasadnienie (3 punkty)")
        reasons = [
            f"Base score: {base_overall}/100 (siła tytułu + sygnały bazowe)",
            f"Trend bonus: {trend_bonus:+d} (Trends/External)",
            f"Similar bonus: {similar_bonus:+d} (dopasowanie do hitów kanału)",
        ]
        for r in reasons:
            st.markdown(f"- {r}")

        st.markdown("#### 📊 Udział składników (wizualnie)")
        base_pct = max(0, min(100, base_overall))
        trend_pct = max(0, min(100, abs(trend_bonus) * 5))
        similar_pct = max(0, min(100, abs(similar_bonus) * 5))
        st.markdown(
            f"<div class='section-card'>"
            f"<div>Base</div>"
            f"<div style='background:#222;border-radius:6px;height:10px;'><div style='width:{base_pct}%;height:10px;background:#4CAF50;border-radius:6px;'></div></div>"
            f"<div style='margin-top:6px;'>Trend</div>"
            f"<div style='background:#222;border-radius:6px;height:10px;'><div style='width:{trend_pct}%;height:10px;background:#FFC107;border-radius:6px;'></div></div>"
            f"<div style='margin-top:6px;'>Similar</div>"
            f"<div style='background:#222;border-radius:6px;height:10px;'><div style='width:{similar_pct}%;height:10px;background:#90EE90;border-radius:6px;'></div></div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        if res.get("risk_flags"):
            st.markdown("#### 🚩 Czerwone flagi")
            st.warning(" | ".join(res.get("risk_flags", [])))

        st.markdown("### 🚀 Szybkie akcje")
        qa1, qa2, qa3 = st.columns(3)
        with qa1:
            if st.button("🔄 Nowe tytuły", key="quick_regen_titles"):
                if st.session_state.topic_job_main:
                    st.session_state.topic_job_main["stage_done"] = 2
                    st.session_state.topic_result_main = st.session_state.topic_job_main.get("result")
                st.session_state.topic_job_main = st.session_state.topic_job_main or {
                    "topic": res.get("topic", ""),
                    "n_titles": n_titles_main,
                    "n_promises": n_promises_main,
                    "api_key": api_key,
                    "result": {"topic": res.get("topic", ""), "timestamp": datetime.now().isoformat()}
                }
                st.session_state.topic_job_main["n_titles"] = n_titles_main
                st.session_state.topic_job_main = _topic_stage_run(3, st.session_state.topic_job_main)
                st.session_state.topic_result_main = st.session_state.topic_job_main.get("result")
                st.rerun()
        with qa2:
            if st.button("🔄 Nowe obietnice", key="quick_regen_promises"):
                if st.session_state.topic_job_main:
                    st.session_state.topic_job_main["stage_done"] = 3
                    st.session_state.topic_result_main = st.session_state.topic_job_main.get("result")
                st.session_state.topic_job_main = st.session_state.topic_job_main or {
                    "topic": res.get("topic", ""),
                    "n_titles": n_titles_main,
                    "n_promises": n_promises_main,
                    "api_key": api_key,
                    "result": {"topic": res.get("topic", ""), "timestamp": datetime.now().isoformat()}
                }
                st.session_state.topic_job_main["n_promises"] = n_promises_main
                st.session_state.topic_job_main = _topic_stage_run(4, st.session_state.topic_job_main)
                st.session_state.topic_result_main = st.session_state.topic_job_main.get("result")
                st.rerun()
        with qa3:
            if st.button("💾 Zapisz wynik do Vault", key="quick_save_topic"):
                best_title = (res.get("selected_title") or {}).get("title") or f"Temat: {res.get('topic','')}"
                best_promise = (res.get("promises") or [{}])[0].get("promise", "")
                score = int(res.get("overall_score", res.get("overall_score_base", 0) or 0))
                vault.add(
                    title=best_title,
                    promise=best_promise,
                    score=score,
                    reason="Ocena tematu - szybki zapis",
                    tags=["topic_mode"],
                    topic=res.get("topic",""),
                    payload=res,
                    status="new"
                )
                st.success("✅ Zapisano do Vault.")
    
        # Titles with tooltips
        st.markdown("### Proponowane tytuły")
        if res.get("titles"):
            titles = res["titles"]
            top_titles = titles[:3]
            st.markdown("**Top 3 (na start):**")
            for t in top_titles:
                reason = t.get("reasoning") or t.get("reason") or t.get("calculated_reasoning", "")
                short_reason = f"{reason.split('.')[0]}." if reason else ""
                badge = _score_badge(int(t.get("score", 0)), reason)
                st.markdown(f"{badge} <b>{t.get('title','')}</b> — {short_reason}", unsafe_allow_html=True)
            title_opts = [t.get("title","") for t in titles]
            selected_title_str = st.radio(
                "Wybierz tytuł do dalszej oceny",
                title_opts,
                index=0,
                key="main_selected_title",
                horizontal=False,
            )
            selected_obj = next((t for t in titles if t.get("title") == selected_title_str), titles[0])
            res["selected_title"] = selected_obj

            reason = selected_obj.get("reasoning") or selected_obj.get("reason") or selected_obj.get("calculated_reasoning", "")
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.markdown(f"**Wybrany tytuł:** {selected_obj.get('title','')}")
            if reason:
                st.markdown(reason)
            st.caption(f"Źródło: {selected_obj.get('source','?')} | Styl: {selected_obj.get('style','?')}")
            st.markdown("</div>", unsafe_allow_html=True)

            with st.expander("Pełna lista tytułów", expanded=False):
                for t in titles:
                    reason = t.get("reasoning") or t.get("reason") or t.get("calculated_reasoning", "")
                    badge = _score_badge(int(t.get("score", 0)), reason)
                    st.markdown(f"{badge} <b>{t.get('title','')}</b>", unsafe_allow_html=True)
                    if reason:
                        st.caption(reason)

            st.markdown("#### ✍️ Szybkie przepisanie tytułu")
            rewrite_style = st.selectbox(
                "Styl przepisu",
                ["bardziej konkretny", "bardziej tajemniczy", "krótszy", "bardziej emocjonalny"],
                key="rewrite_style",
            )
            if st.button("⚡ Przepisz tytuł", key="rewrite_title"):
                if not api_key:
                    st.warning("Dodaj OpenAI API Key aby przepisać tytuł.")
                else:
                    try:
                        from openai import OpenAI
                        client = OpenAI(api_key=api_key)
                        prompt = f"Napisz 1 wariant tytułu w stylu: {rewrite_style}. Oryginał: {selected_title_str}"
                        resp = client.chat.completions.create(
                            model=config.get("openai_model", "gpt-4o-mini"),
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.7,
                        )
                        rewritten = resp.choices[0].message.content.strip()
                        st.success(rewritten)
                    except Exception as e:
                        st.warning(f"Nie udało się przepisać tytułu: {e}")

            st.markdown("#### ⚖️ Porównaj 2 tytuły (szybko)")
            cmp1, cmp2 = st.columns(2)
            with cmp1:
                title_a = st.text_input("Tytuł A", value=title_opts[0], key="compare_title_a")
            with cmp2:
                title_b = st.text_input("Tytuł B", value=title_opts[1] if len(title_opts) > 1 else "", key="compare_title_b")
            if st.button("Porównaj tytuły", key="compare_titles"):
                if not api_key:
                    st.warning("Dodaj OpenAI API Key aby porównać tytuły.")
                else:
                    try:
                        from openai import OpenAI
                        client = OpenAI(api_key=api_key)
                        prompt = f"Porównaj dwa tytuły i wskaż lepszy. A: {title_a} B: {title_b}. Zwróć krótki werdykt."
                        resp = client.chat.completions.create(
                            model=config.get("openai_model", "gpt-4o-mini"),
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.3,
                        )
                        verdict = resp.choices[0].message.content.strip()
                        st.info(verdict)
                    except Exception as e:
                        st.warning(f"Nie udało się porównać tytułów: {e}")
        else:
            st.warning("Brak wygenerowanych tytułów.")
    
        # Promises
        st.markdown("### Proponowane obietnice (hook/promise)")
        if res.get("promises"):
            promises = res["promises"]
            top_promises = promises[:3]
            st.markdown("**Top 3 (na start):**")
            for p in top_promises:
                reason = p.get("reasoning") or p.get("reason") or ""
                short_reason = f"{reason.split('.')[0]}." if reason else ""
                badge = _score_badge(int(p.get("score", 0)), reason)
                st.markdown(f"{badge} {p.get('promise','')} — {short_reason}", unsafe_allow_html=True)
            promise_opts = [p.get("promise","") for p in promises]
            chosen_promise = st.radio(
                "Wybierz obietnicę",
                promise_opts,
                index=0,
                key="main_selected_promise",
            )
            selected_promise_obj = next((p for p in promises if p.get("promise") == chosen_promise), promises[0])
            reason = selected_promise_obj.get("reasoning") or selected_promise_obj.get("reason") or ""
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.markdown(f"**Wybrana obietnica:** {selected_promise_obj.get('promise','')}")
            if reason:
                st.markdown(reason)
            st.caption(f"Źródło: {selected_promise_obj.get('source','?')}")
            st.markdown("</div>", unsafe_allow_html=True)

            with st.expander("Pełna lista obietnic", expanded=False):
                for p in promises:
                    reason = p.get("reasoning") or p.get("reason") or ""
                    badge = _score_badge(int(p.get("score", 0)), reason)
                    st.markdown(f"{badge} {p.get('promise','')}", unsafe_allow_html=True)
                    if reason:
                        st.caption(reason)

            st.markdown("#### 🧪 Szybka ocena hooka")
            hook_text = st.text_area("Wklej hook (2-3 zdania)", key="hook_quick_text", height=80)
            if st.button("Oceń hook", key="hook_quick_btn"):
                if not api_key:
                    st.warning("Dodaj OpenAI API Key aby ocenić hook.")
                else:
                    try:
                        from openai import OpenAI
                        client = OpenAI(api_key=api_key)
                        prompt = f"Oceń hook (0-100) i podaj 1 zdanie uzasadnienia. Hook: {hook_text}"
                        resp = client.chat.completions.create(
                            model=config.get("openai_model", "gpt-4o-mini"),
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.4,
                        )
                        st.info(resp.choices[0].message.content.strip())
                    except Exception as e:
                        st.warning(f"Nie udało się ocenić hooka: {e}")
        else:
            st.info("Brak obietnic. Uruchom etap generacji obietnic.")
    
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
                recent_warning = None
                for h in res["similar_hits"][:8]:
                    st.markdown(f"- {h.get('title','')} | {h.get('views',0):,} views | {h.get('label','')}")
                    pub = h.get("published_at") or h.get("publishedAt")
                    if pub:
                        try:
                            pub_dt = pd.to_datetime(pub, errors="coerce")
                            if pd.notna(pub_dt) and (datetime.now() - pub_dt).days <= 30:
                                recent_warning = h.get("title", "")
                        except Exception:
                            pass
                if recent_warning:
                    st.warning(f"⚠️ Ten temat może kanibalizować świeży film: {recent_warning}")
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
                    payload=res,
                    status="new"
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
                    "payload": res,
                    "tags": ["topic_mode"],
                    "status": "new"
                })
                st.success("✅ Zapisano (Vault + Historia).")
        with col_s2:
            if st.button("🗑️ Wyczyść wynik tematu", key="main_clear_topic"):
                st.session_state.topic_result_main = None
                st.session_state.topic_job_main = None
    
    
# =============================================================================
# TAB: NARZĘDZIA
# =============================================================================

with tab_tools:
    st.header("🛠️ Narzędzia")
    
    tool_tabs = st.tabs([
        "📉 Dlaczego wtopa?",
        "🕳️ Content Gaps",
        "📅 Kalendarz",
        "🔔 Trend Alerts",
        "👀 Competitor Tracker"
    ])
    
    # --- WTOPA ANALYZER ---
    with tool_tabs[0]:
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

        st.session_state.setdefault("wtopa_title", "")
        st.session_state.setdefault("wtopa_views", 0)
        st.session_state.setdefault("wtopa_ret", 40.0)

        wtopa_title = st.text_input("Tytuł filmu", key="wtopa_title")
        wtopa_views = st.number_input("Wyświetlenia", min_value=0, key="wtopa_views")
        wtopa_retention = st.slider("Retencja (%)", 0.0, 100.0, key="wtopa_ret")

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

    with tool_tabs[1]:
        st.subheader("🕳️ Content Gap Finder")
        st.markdown("Tematy popularne w niszy, których jeszcze nie robiłeś")
        
        if st.button("🔍 Znajdź luki") and merged_df is not None and ADVANCED_AVAILABLE:
            finder = ContentGapFinder(merged_df)
            gaps = finder.find_gaps()
            
            st.divider()
            
            channel_median_views = None
            channel_avg_views = None
            channel_avg_retention = None
            if "views" in merged_df.columns:
                channel_median_views = int(merged_df["views"].median())
                channel_avg_views = int(merged_df["views"].mean())
            if "retention" in merged_df.columns and merged_df["retention"].notna().any():
                channel_avg_retention = float(merged_df["retention"].mean())

            for gap in gaps[:15]:
                covered = gap["covered"]
                icon = "🟡" if covered else "🟢"
                topic = gap["topic"]
                topic_mask = merged_df["title"].astype(str).str.lower().str.contains(topic)
                topic_df = merged_df[topic_mask]
                topic_count = int(topic_df.shape[0])
                topic_avg_views = int(topic_df["views"].mean()) if topic_count and "views" in topic_df.columns else None
                topic_median_views = int(topic_df["views"].median()) if topic_count and "views" in topic_df.columns else None
                topic_avg_retention = float(topic_df["retention"].mean()) if topic_count and "retention" in topic_df.columns and topic_df["retention"].notna().any() else None

                st.markdown(f"{icon} **{topic.title()}** - {gap['recommendation']}")

                stat_lines = []
                if channel_median_views is not None:
                    stat_lines.append(f"Mediana kanału: **{channel_median_views:,}** views")
                if channel_avg_views is not None:
                    stat_lines.append(f"Średnia kanału: **{channel_avg_views:,}** views")
                if topic_count:
                    if topic_avg_views is not None:
                        stat_lines.append(f"Średnia dla tematu: **{topic_avg_views:,}** views")
                    if topic_median_views is not None:
                        stat_lines.append(f"Mediana dla tematu: **{topic_median_views:,}** views")
                    stat_lines.append(f"Liczba filmów na temat: **{topic_count}**")
                    if topic_avg_retention is not None:
                        stat_lines.append(f"Średnia retencja tematu: **{topic_avg_retention:.1f}%**")
                else:
                    stat_lines.append("Brak filmów na ten temat na kanale.")

                if stat_lines:
                    st.caption(" | ".join(stat_lines))

                if not covered:
                    with st.expander("💡 Pomysły na filmy"):
                        suggestions = finder.suggest_ideas(topic)
                        for s in suggestions:
                            st.markdown(f"- {s}")
        elif merged_df is None:
            st.warning("Załaduj dane kanału")
    
    # --- KALENDARZ ---
    with tool_tabs[2]:
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
                    trending_now = trend_disc.discover_trending(niche_keywords)[:10]
                except Exception:
                    trending_now = []

                n_items = int(plan_weeks * plan_per_week)
                n_items = max(1, min(12, n_items))

                # 2) LLM: zaproponuj tematy i kąty
                plan_items = []
                use_llm = False
                if api_key:
                    try:
                        from openai import OpenAI
                        use_llm = True
                    except Exception as e:
                        st.warning(f"Nie udało się zainicjalizować OpenAI: {e}")

                if use_llm:
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
                    angles = [
                        "Ukryta prawda", "Timeline zdarzeń", "Śledztwo krok po kroku",
                        "Konsekwencje i stawka", "Najbardziej niewygodny fakt"
                    ]
                    candidate_topics = []
                    seen = set()

                    for item in trending_now:
                        topic = str(item.get("topic", "")).strip()
                        if topic and topic.lower() not in seen:
                            seen.add(topic.lower())
                            candidate_topics.append({
                                "topic": topic,
                                "why": item.get("recommendation") or f"Trendy teraz (score: {item.get('score', 0)})",
                                "source": "trend"
                            })

                    gap_finder = ContentGapFinder(merged_df)
                    gap_topics = [g for g in gap_finder.find_gaps() if not g.get("covered")]
                    for g in gap_topics:
                        topic = g.get("topic", "").strip()
                        if topic and topic.lower() not in seen:
                            seen.add(topic.lower())
                            candidate_topics.append({
                                "topic": topic,
                                "why": "Brak filmów na kanale + temat niszowy",
                                "source": "gap"
                            })

                    for hit in top_hits[:5]:
                        topic = str(hit).strip()
                        if topic and topic.lower() not in seen:
                            seen.add(topic.lower())
                            candidate_topics.append({
                                "topic": topic,
                                "why": "Podobny do historycznych hitów kanału",
                                "source": "hit"
                            })

                    for kw in niche_keywords[:5]:
                        topic = str(kw).strip()
                        if topic and topic.lower() not in seen:
                            seen.add(topic.lower())
                            candidate_topics.append({
                                "topic": topic,
                                "why": "Niszowe słowo kluczowe z konfiguracji",
                                "source": "niche"
                            })

                    plan_items = []
                    for idx, item in enumerate(candidate_topics[:n_items], start=1):
                        plan_items.append({
                            "week": ((idx - 1) // plan_per_week) + 1,
                            "topic": item["topic"],
                            "angle": angles[(idx - 1) % len(angles)],
                            "why": item["why"],
                            "source": item["source"]
                        })

                if not plan_items:
                    st.info("Brak planu do pokazania. Spróbuj ponownie lub ustaw inny fokus.")
                else:
                    # 3) Szybka ocena każdego tematu (score + tytuł + viral)
                    eval_results = []
                    if TOPIC_ANALYZER_AVAILABLE:
                        try:
                            from openai import OpenAI
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

                    def _render_month_calendar(items: List[Dict], weeks: int, per_week: int) -> None:
                        today = datetime.now().date()
                        month_matrix = calendar.monthcalendar(today.year, today.month)
                        month_name = today.strftime("%B %Y")
                        days_header = ["Pn", "Wt", "Śr", "Cz", "Pt", "Sb", "Nd"]

                        day_assignments: Dict[int, List[Dict]] = {}
                        overflow_items: List[Dict] = []

                        item_idx = 0
                        for item in items:
                            week_no = int(item.get("week", 1))
                            if week_no < 1:
                                week_no = 1
                            if week_no > len(month_matrix):
                                overflow_items.append(item)
                                continue
                            week_row = month_matrix[week_no - 1]
                            day_indices = [idx for idx, day in enumerate(week_row) if day > 0]
                            if not day_indices:
                                overflow_items.append(item)
                                continue
                            bucket = day_indices[item_idx % len(day_indices)]
                            day_num = week_row[bucket]
                            day_assignments.setdefault(day_num, []).append(item)
                            item_idx += 1

                        def _cell(day_num: int) -> str:
                            if day_num == 0:
                                return "<td class='cal-cell empty'></td>"
                            items_html = ""
                            for itm in day_assignments.get(day_num, []):
                                title = itm.get("topic", "")
                                angle = itm.get("angle", "")
                                items_html += f"<div class='cal-item'><strong>{title}</strong><span>{angle}</span></div>"
                            return f"<td class='cal-cell'><div class='cal-date'>{day_num}</div>{items_html}</td>"

                        rows_html = ""
                        for week_row in month_matrix:
                            row_cells = "".join(_cell(day) for day in week_row)
                            rows_html += f"<tr>{row_cells}</tr>"

                        calendar_html = f"""
                        <style>
                        .cal-wrap {{
                            border: 1px solid #333;
                            border-radius: 12px;
                            overflow: hidden;
                            background: #111;
                        }}
                        .cal-header {{
                            padding: 12px 16px;
                            font-weight: 600;
                            font-size: 1.1rem;
                            background: #1b1b1b;
                            border-bottom: 1px solid #2a2a2a;
                        }}
                        .cal-table {{
                            width: 100%;
                            border-collapse: collapse;
                        }}
                        .cal-table th {{
                            padding: 8px;
                            font-weight: 600;
                            color: #bbb;
                            border-bottom: 1px solid #222;
                            background: #161616;
                            text-align: center;
                        }}
                        .cal-cell {{
                            vertical-align: top;
                            padding: 8px;
                            min-height: 100px;
                            border-right: 1px solid #222;
                            border-bottom: 1px solid #222;
                        }}
                        .cal-cell.empty {{
                            background: #0d0d0d;
                        }}
                        .cal-date {{
                            font-size: 0.85rem;
                            color: #888;
                            margin-bottom: 6px;
                        }}
                        .cal-item {{
                            background: #202020;
                            border: 1px solid #2f2f2f;
                            border-radius: 8px;
                            padding: 6px 8px;
                            margin-bottom: 6px;
                            font-size: 0.8rem;
                        }}
                        .cal-item strong {{
                            display: block;
                            color: #f0f0f0;
                        }}
                        .cal-item span {{
                            display: block;
                            color: #9a9a9a;
                            font-size: 0.75rem;
                        }}
                        </style>
                        <div class='cal-wrap'>
                          <div class='cal-header'>📅 {month_name} — Plan na {weeks} tygodni ({per_week} / tydzień)</div>
                          <table class='cal-table'>
                            <thead>
                              <tr>{''.join([f'<th>{d}</th>' for d in days_header])}</tr>
                            </thead>
                            <tbody>
                              {rows_html}
                            </tbody>
                          </table>
                        </div>
                        """
                        st.markdown(calendar_html, unsafe_allow_html=True)
                        if overflow_items:
                            st.warning("Część tematów wykracza poza bieżący miesiąc — poniżej lista dodatkowa.")
                            for itm in overflow_items:
                                st.markdown(f"- {itm.get('topic', '')} (tydzień {itm.get('week', '?')})")

                    st.markdown("#### Widok kalendarza (miesiąc)")
                    _render_month_calendar(plan_items, plan_weeks, plan_per_week)

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
                                if item.get("source"):
                                    st.caption(f"Źródło pomysłu: {item.get('source')}")
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
                                            status="new",
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
                                status="new",
                                payload=r.get("payload", {})
                            )
                            if ok:
                                saved += 1
                        st.success(f"Zapisano {saved} pomysłów.")
    # --- TREND ALERTS ---
    with tool_tabs[3]:
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
                    seeds = [s.strip() for s in seeds_text.split(",") if s.strip()][:8]
                    top = discovery.discover_trending(seeds) if seeds else discovery.discover_trending()

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
                        t = item.get("topic", "")
                        det = trend_map.get(t, {})
                        overall = det.get("overall") or {}
                        rows.append({
                            "Temat": t,
                            "Źródło": item.get("source", ""),
                            "Score": item.get("score", 0),
                            "Rekomendacja": item.get("recommendation", ""),
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

            st.subheader("🔮 Prediction — prognoza rozwoju kanału")
            st.caption("Heurystyczna prognoza na bazie ostatnich publikacji i średnich wyników.")

            date_col = None
            for col in ["published_at", "publishedAt", "date", "published"]:
                if col in merged_df.columns:
                    date_col = col
                    break

            df_pred = merged_df.copy()
            if date_col:
                df_pred["date"] = pd.to_datetime(df_pred[date_col], errors="coerce")
                df_pred = df_pred.dropna(subset=["date"]).sort_values("date")

            if "views" not in df_pred.columns or df_pred.empty:
                st.info("Brak wystarczających danych (views/published_at) do prognozy.")
            else:
                planned_uploads = st.slider("Planowane publikacje / miesiąc", 1, 12, 4, key="pred_uploads")

                recent_df = df_pred.copy()
                if date_col:
                    cutoff = datetime.now() - timedelta(days=30)
                    recent_df = recent_df[recent_df["date"] >= cutoff]
                if recent_df.empty:
                    recent_df = df_pred.tail(10)

                avg_recent_views = recent_df["views"].mean()
                avg_all_views = df_pred["views"].mean()
                growth_pct = 0.0
                if avg_all_views:
                    growth_pct = (avg_recent_views / avg_all_views - 1.0) * 100.0

                projected_month_views = int(avg_recent_views * planned_uploads)
                projected_next_total = int(df_pred["views"].sum() + projected_month_views)

                p1, p2, p3, p4 = st.columns(4)
                with p1:
                    st.metric("Śr. views (ostatnie publikacje)", f"{avg_recent_views:,.0f}")
                with p2:
                    st.metric("Trend vs średnia kanału", f"{growth_pct:+.1f}%")
                with p3:
                    st.metric("Prognoza views / miesiąc", f"{projected_month_views:,.0f}")
                with p4:
                    st.metric("Prognoza total views", f"{projected_next_total:,.0f}")

                st.caption("Prognoza bazuje na średniej z ostatnich publikacji i zakładanej liczbie filmów w miesiącu.")
        
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
    with tool_tabs[4]:
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
            fetch_btn = st.button("Pobierz ostatnie filmy", key="comp_fetch_btn", disabled=not competitors)

            if fetch_btn:
                if not competitors:
                    st.warning("Dodaj co najmniej jednego konkurenta, aby pobrać uploady.")
                    st.stop()

                yt_sync = get_youtube_sync()
                yt_client = None
                source_hint = "fallback"
                if GOOGLE_API_AVAILABLE and yt_sync.has_credentials():
                    ok, msg = yt_sync.authenticate()
                    if ok:
                        yt_client = yt_sync.youtube
                        source_hint = "oauth"
                    else:
                        st.warning(msg)

                if not yt_client and GOOGLE_API_AVAILABLE:
                    api_key = config.get("youtube_api_key", "")
                    if api_key:
                        yt_sync.set_api_key(api_key)
                        if yt_sync.ensure_public_client():
                            yt_client = yt_sync.youtube
                            source_hint = "api_key"
                    else:
                        st.info("Ustaw YouTube API Key w konfiguracji, aby użyć trybu publicznego bez OAuth.")

                tracker = get_competitor_tracker(yt_client)
                uploads = tracker.fetch_recent_uploads(competitors, days=days, max_per_channel=max_per)

                # filter out errors separately
                errs = [u for u in uploads if u.get("error")]
                vids = [u for u in uploads if u.get("video_id")]

                if errs:
                    with st.expander("⚠️ Błędy", expanded=False):
                        st.json(errs)

                if vids:
                    st.caption(f"Źródło danych: {source_hint}")
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
                            payload=payload,
                            status="new"
                        )
                        st.success("✅ Zapisano do Vault.")

                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown(f"**Tytuł:** {entry.get('title', '')}")
                    st.markdown(f"**Obietnica:** {entry.get('promise', '') or 'Brak'}")
                    if entry.get("status"):
                        st.markdown(f"**Status:** {entry.get('status')}")
                    if entry.get("tags"):
                        st.markdown(f"**Tagi:** {', '.join(entry.get('tags', []))}")
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
        col_export1, col_export2 = st.columns(2)
        with col_export1:
            if st.button("📥 Eksportuj do CSV"):
                csv = history.export_to_csv()
                st.download_button(
                    "⬇️ Pobierz CSV",
                    csv,
                    "evaluation_history.csv",
                    "text/csv"
                )
        with col_export2:
            if st.button("📥 Eksportuj do JSON"):
                json_payload = history.export_to_json()
                st.download_button(
                    "⬇️ Pobierz JSON",
                    json_payload,
                    "evaluation_history.json",
                    "application/json"
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
        ["Wszystkie", "Nowe", "Shortlisted", "Scripted", "Użyte", "Odrzucone"],
        key="vault_filter"
    )
    
    status_map = {
        "Wszystkie": None,
        "Nowe": "new",
        "Shortlisted": "shortlisted",
        "Scripted": "scripted",
        "Użyte": "used",
        "Odrzucone": "discarded"
    }
    
    all_ideas = vault.get_all(status=status_map.get(status_filter))
    all_tags = sorted({t for idea in all_ideas for t in idea.get("tags", [])})
    selected_tags = st.multiselect("Filtruj po tagach", options=all_tags, default=[], key="vault_tag_filter")
    if selected_tags:
        all_ideas = [i for i in all_ideas if set(i.get("tags", [])) & set(selected_tags)]
    selected_ids = set()
    
    if not all_ideas:
        st.info("Vault jest pusty. Zapisz pomysły podczas oceny!")
    else:
        st.markdown("### ⚙️ Akcje zbiorcze")
        bulk_status = st.selectbox(
            "Ustaw status dla zaznaczonych",
            ["new", "shortlisted", "scripted", "used", "discarded"],
            key="vault_bulk_status"
        )
        if st.button("✅ Zastosuj status do zaznaczonych", key="vault_bulk_apply"):
            for idea in all_ideas:
                if idea.get("id") in st.session_state.get("vault_selected_ids", []):
                    vault.update_metadata(idea["id"], status=bulk_status)
            st.success("Zaktualizowano zaznaczone wpisy.")
            st.rerun()
        if st.button("🗑️ Usuń zaznaczone", key="vault_bulk_delete"):
            for idea in all_ideas:
                if idea.get("id") in st.session_state.get("vault_selected_ids", []):
                    vault.remove(idea["id"])
            st.success("Usunięto zaznaczone wpisy.")
            st.rerun()

        for idea in all_ideas:
            status = idea.get("status", "new")
            if status == "waiting":
                status = "new"
            status_emoji = "🆕" if status == "new" else "⭐" if status == "shortlisted" else "📝" if status == "scripted" else "✅" if status == "used" else "❌"
            
            selected = st.checkbox(
                "Zaznacz",
                key=f"vault_select_{idea['id']}"
            )
            if selected:
                selected_ids.add(idea["id"])

            with st.expander(f"{status_emoji} {idea.get('title', '')[:50]}... | Score: {idea.get('score', 0)}"):
                st.markdown(f"**Tytuł:** {idea.get('title', '')}")
                st.markdown(f"**Obietnica:** {idea.get('promise', '') or 'Brak'}")
                st.markdown(f"**Powód zapisania:** {idea.get('reason', '') or 'Nie podano'}")

                with st.expander("📦 Pełny wynik (payload)", expanded=False):
                    st.json(idea.get("payload") or {})
                st.markdown(f"**Tagi:** {', '.join(idea.get('tags', [])) or 'Brak'}")
                st.markdown(f"**Dodano:** {idea.get('added', '')[:10]}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    new_status = st.selectbox(
                        "Status",
                        ["new", "shortlisted", "scripted", "used", "discarded"],
                        index=["new", "shortlisted", "scripted", "used", "discarded"].index(status if status in ["new", "shortlisted", "scripted", "used", "discarded"] else "new"),
                        key=f"vault_status_{idea['id']}"
                    )
                    new_tags = st.text_input(
                        "Tagi (oddziel przecinkami)",
                        value=", ".join(idea.get("tags", [])),
                        key=f"vault_tags_{idea['id']}"
                    )
                    new_notes = st.text_area(
                        "Notatki",
                        value=idea.get("notes", ""),
                        key=f"vault_notes_{idea['id']}"
                    )
                    if st.button("💾 Zapisz zmiany", key=f"vault_save_{idea['id']}"):
                        tags_list = [t.strip() for t in new_tags.split(",") if t.strip()]
                        vault.update_metadata(idea["id"], tags=tags_list, status=new_status, notes=new_notes)
                        st.success("✅ Zaktualizowano.")
                        st.rerun()
                
                with col2:
                    if st.button("🗑️ Usuń", key=f"vault_delete_{idea['id']}"):
                        vault.remove(idea["id"])
                        st.rerun()

    st.session_state["vault_selected_ids"] = list(selected_ids)

    st.divider()
    if all_ideas:
        vault_json = json.dumps(all_ideas, indent=2, ensure_ascii=False, default=str)
        st.download_button(
            "⬇️ Pobierz Vault (JSON)",
            vault_json,
            "idea_vault.json",
            "application/json"
        )
        try:
            vault_df = pd.DataFrame(all_ideas)
            st.download_button(
                "⬇️ Pobierz Vault (CSV)",
                vault_df.to_csv(index=False),
                "idea_vault.csv",
                "text/csv"
            )
        except Exception:
            pass

# =============================================================================
# TAB: DANE
# =============================================================================

with tab_data:
    st.header("📁 Zarządzanie danymi")
    
    # YouTube Sync section
    st.subheader("📺 YouTube Sync")
    
    yt_sync = get_youtube_sync()
    yt_sync.set_api_key(config.get_youtube_api_key())
    yt_sync.set_channel_id(config.get("channel_id", ""))
    
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
        if yt_sync.ensure_public_client() and config.get("channel_id"):
            if st.button("🔄 Publiczny sync (bez Analytics)", key="public_sync_data"):
                with st.spinner("Pobieram dane publiczne..."):
                    df, sync_msg = yt_sync.sync_all(include_analytics=False, include_transcripts=False)
                    st.success(sync_msg)
                    st.rerun()
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
        validations = []
        for file in uploaded_files:
            df = pd.read_csv(file)
            st.write(f"**{file.name}**: {len(df)} wierszy, kolumny: {', '.join(df.columns[:5])}")
            issues = validate_channel_dataframe(df)
            validations.append(issues)
            if issues["missing_required"]:
                st.error(f"Brak wymaganych kolumn: {', '.join(issues['missing_required'])}")
            if issues["missing_recommended"]:
                st.warning(f"Brak rekomendowanych kolumn: {', '.join(issues['missing_recommended'])}")
            for warning in issues["warnings"]:
                st.warning(warning)
        
        if st.button("💾 Zapisz i połącz dane"):
            if any(v["missing_required"] for v in validations):
                st.error("Nie można zapisać danych: brakuje wymaganych kolumn (title, views).")
                st.stop()
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
        issues = validate_channel_dataframe(merged_df)
        if issues["missing_required"]:
            st.error(f"Brak wymaganych kolumn: {', '.join(issues['missing_required'])}")
        if issues["missing_recommended"]:
            st.warning(f"Brak rekomendowanych kolumn: {', '.join(issues['missing_recommended'])}")
        for warning in issues["warnings"]:
            st.warning(warning)

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
# TAB: DIAGNOSTYKA
# =============================================================================

with tab_diag:
    st.header("🧪 Diagnostyka")
    st.caption("Stan modułów, dane, cache i ostatnie błędy.")

    st.subheader("Status modułów")
    diag_modules = {
        "Topic Analyzer": TOPIC_ANALYZER_AVAILABLE,
        "Advanced Analytics": ADVANCED_AVAILABLE,
        "Competitor Tracker": COMPETITOR_TRACKER_AVAILABLE,
        "YouTube API": GOOGLE_API_AVAILABLE,
    }
    for name, available in diag_modules.items():
        st.markdown(f"{'✅' if available else '⚠️'} {name}")

    st.subheader("Dane kanału")
    if merged_df is not None:
        issues = validate_channel_dataframe(merged_df)
        if issues["missing_required"]:
            st.error(f"Brak wymaganych kolumn: {', '.join(issues['missing_required'])}")
        if issues["missing_recommended"]:
            st.warning(f"Brak rekomendowanych kolumn: {', '.join(issues['missing_recommended'])}")
        for warning in issues["warnings"]:
            st.warning(warning)
    else:
        st.info("Brak danych kanału.")

    st.subheader("Sync i cache")
    yt_sync = get_youtube_sync()
    last_sync = yt_sync.get_last_sync_time()
    st.caption(f"Ostatnia synchronizacja: {last_sync or 'brak'}")

    cache_store = _load_cache_store()
    cache_entries = sum(len(v) for v in cache_store.values()) if cache_store else 0
    st.caption(f"Cache entries: {cache_entries}")

    st.subheader("Ustawienia cache (TTL)")
    st.session_state.setdefault("cache_ttl", {})
    c1, c2 = st.columns(2)
    with c1:
        st.session_state["cache_ttl"]["trends"] = st.slider("TTL trendów (h)", 1, 48, 6)
        st.session_state["cache_ttl"]["external"] = st.slider("TTL external (h)", 1, 72, 12)
        st.session_state["cache_ttl"]["competition"] = st.slider("TTL konkurencji (h)", 1, 48, 6)
    with c2:
        st.session_state["cache_ttl"]["similar_hits"] = st.slider("TTL podobnych hitów (h)", 1, 48, 6)
        st.session_state["cache_ttl"]["llm_titles"] = st.slider("TTL tytułów (h)", 1, 168, 24)
        st.session_state["cache_ttl"]["llm_promises"] = st.slider("TTL obietnic (h)", 1, 168, 24)

    st.subheader("Statystyki LLM")
    llm_stats = st.session_state.get("llm_stats", {})
    if llm_stats:
        st.json(llm_stats)
    else:
        st.info("Brak danych o wywołaniach LLM.")

    st.subheader("Ostatnie zdarzenia")
    diagnostics = st.session_state.get("diagnostics", [])
    if diagnostics:
        st.json(diagnostics[-50:])
    else:
        st.info("Brak zapisanych zdarzeń diagnostycznych.")

    if diagnostics:
        diag_json = json.dumps(diagnostics, indent=2, ensure_ascii=False, default=str)
        st.download_button(
            "⬇️ Pobierz logi diagnostyczne (JSON)",
            diag_json,
            "diagnostics.json",
            "application/json"
        )

# =============================================================================
# FOOTER
# =============================================================================

st.divider()
st.caption("YT Idea Evaluator Pro v3 | Made for Dawid 🎬")
