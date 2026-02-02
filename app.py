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
from typing import Any, Optional, Dict, List, Tuple
from types import SimpleNamespace
import plotly.graph_objects as go
import plotly.express as px

from ui.components import (
    copy_to_clipboard_button,
    render_advanced_insights,
    render_copy_report,
    render_diagnosis,
    render_dimensions,
    render_improvements,
    render_variants_with_scores,
    render_verdict_card,
)
from ui.sidebar import render_sidebar
from ui.styles import inject_styles
from ui.tab_analytics import render_analytics_tab
from ui.tab_data import render_data_tab
from ui.tab_diagnostics import render_diagnostics_tab
from ui.tab_evaluate import render_evaluate_tab
from ui.tab_history import render_history_tab
from ui.tab_tools import render_tools_tab
from ui.tab_vault import render_vault_tab
from ui.tooltips import TOOLTIPS, show_tooltip

# Import modułów v3
from yt_idea_evaluator_pro_v2 import YTIdeaEvaluatorV2, Config, format_result
from config_manager import (
    AppConfig, EvaluationHistory, IdeaVault, TrendAlerts,
    get_config, get_history, get_vault, get_alerts,
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

try:
    import google.generativeai as genai
    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    GOOGLE_GENAI_AVAILABLE = False

LLM_PROVIDER_LABELS = {
    "openai": "OpenAI",
    "google": "Google AI Studio (Gemini)",
}



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

def init_session_state():
    """
    Inicjalizuje wszystkie klucze session state z domyślnymi wartościami.
    Centralne miejsce definicji stanu aplikacji.
    """
    defaults = {
        # Diagnostyka i statystyki
        "diagnostics": [],
        "llm_stats": {"calls": 0, "cached_hits": 0, "by_kind": {}},

        # Cache TTL (godziny)
        "cache_ttl": {
            "trends": 6,
            "external": 12,
            "competition": 6,
            "similar_hits": 6,
            "llm_titles": 24,
            "llm_promises": 24,
        },

        # Topic evaluation
        "topic_job_main": None,
        "topic_result_main": None,
        "batch_topic_results": None,

        # Narzędzia
        "last_wtopa": None,
        "wtopa_title": "",
        "wtopa_views": 0,
        "wtopa_ret": 0.0,

        # Idea Vault
        "vault_selected_ids": [],

        # Status API
        "openai_api_status": {"tested": False, "success": False, "message": ""},
        "google_api_status": {"tested": False, "success": False, "message": ""},
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


# Inicjalizuj session state przy starcie
init_session_state()


# =============================================================================
# STYLE CSS (DARK MODE FRIENDLY)
# =============================================================================

inject_styles()

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
SCRIPT_SYNC_DIR = Path("./script_sync")
CACHE_VERSION = "v1"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_merged_data() -> Optional[pd.DataFrame]:
    """Ładuje połączone dane kanału z normalizacją kolumn"""
    from config_manager import normalize_dataframe_columns

    # Najpierw sprawdź synced data
    synced_file = CHANNEL_DATA_DIR / "synced_channel_data.csv"
    if synced_file.exists():
        df = pd.read_csv(synced_file)
        return normalize_dataframe_columns(df)

    if MERGED_DATA_FILE.exists():
        df = pd.read_csv(MERGED_DATA_FILE)
        return normalize_dataframe_columns(df)
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

def load_manual_scripts(directory: Path) -> List[Dict[str, str]]:
    """Ładuje skrypty przygotowane ręcznie z katalogu."""
    if not directory.exists():
        return []

    scripts = []
    for ext in (".txt", ".md"):
        for path in sorted(directory.glob(f"*{ext}")):
            try:
                content = path.read_text(encoding="utf-8")
            except Exception:
                continue
            scripts.append({"name": path.name, "content": content})
    return scripts

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

class GoogleAIStudioClient:
    def __init__(self, api_key: str, model: str = "auto"):
        if not GOOGLE_GENAI_AVAILABLE:
            raise RuntimeError("Brak biblioteki google-generativeai.")
        if not api_key:
            raise RuntimeError("Brak Google AI Studio API key.")
        genai.configure(api_key=api_key)
        self._genai = genai
        self.model_name = _resolve_google_model(api_key, model)
        self.chat = self.Chat(self)

    class Chat:
        def __init__(self, outer):
            self.completions = GoogleAIStudioClient.Chat.Completions(outer)

        class Completions:
            def __init__(self, outer):
                self._outer = outer

            def create(self, model=None, messages=None, temperature=0.7, max_tokens=1024, **kwargs):
                prompt = _build_prompt_from_messages(messages or [])
                model_name = model or self._outer.model_name
                llm = self._outer._genai.GenerativeModel(model_name)
                generation_config = self._outer._genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
                response = llm.generate_content(prompt, generation_config=generation_config)
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=response.text))]
                )

def _build_prompt_from_messages(messages: List[Dict[str, str]]) -> str:
    prompt_parts = []
    for msg in messages:
        role = msg.get("role", "user").upper()
        content = msg.get("content", "")
        prompt_parts.append(f"{role}:\n{content}")
    return "\n\n".join(prompt_parts).strip()

def _get_openai_model_list(api_key: str) -> List[str]:
    fallback_order = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4.1",
        "gpt-4-turbo",
        "gpt-4",
        "gpt-3.5-turbo",
    ]
    if not api_key:
        return fallback_order
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        models = client.models.list()
        available = {model.id for model in models.data if model.id}
        ordered = [model_id for model_id in fallback_order if model_id in available]
        ordered.extend(sorted(model_id for model_id in available if model_id not in ordered))
        return ordered or fallback_order
    except Exception:
        return fallback_order


def _resolve_openai_model(api_key: str, requested: str) -> str:
    requested = (requested or "").strip()
    if requested and requested.lower() not in {"auto", "latest"}:
        return requested
    fallback_order = _get_openai_model_list(api_key)
    if not api_key:
        return fallback_order[0]
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        models = client.models.list()
        available = {model.id for model in models.data if model.id}
        for candidate in fallback_order:
            if candidate in available:
                return candidate
    except Exception:
        pass
    return fallback_order[0]


def _get_google_model_list(api_key: str) -> List[str]:
    fallback_order = [
        "gemini-1.5-pro-latest",
        "gemini-1.5-pro",
        "gemini-1.5-flash-latest",
        "gemini-1.5-flash",
        "gemini-1.0-pro",
    ]
    if not api_key or not GOOGLE_GENAI_AVAILABLE:
        return fallback_order
    try:
        genai.configure(api_key=api_key)
        available = {
            model.name.replace("models/", "")
            for model in genai.list_models()
            if getattr(model, "name", "")
        }
        ordered = [model_id for model_id in fallback_order if model_id in available]
        ordered.extend(sorted(model_id for model_id in available if model_id not in ordered))
        return ordered or fallback_order
    except Exception:
        return fallback_order


def _resolve_google_model(api_key: str, requested: str) -> str:
    requested = (requested or "").strip()
    if requested and requested.lower() not in {"auto", "latest"}:
        return requested
    fallback_order = _get_google_model_list(api_key)
    if not api_key or not GOOGLE_GENAI_AVAILABLE:
        return fallback_order[0]
    try:
        genai.configure(api_key=api_key)
        available = {
            model.name.replace("models/", "")
            for model in genai.list_models()
            if getattr(model, "name", "")
        }
        for candidate in fallback_order:
            if candidate in available:
                return candidate
    except Exception:
        pass
    return fallback_order[0]


def get_llm_settings() -> Tuple[str, str, str]:
    """
    Zwraca (provider, api_key, model) dla aktualnie wybranego dostawcy LLM.
    Uwzględnia stan włączenia/wyłączenia danego providera.
    """
    provider = config.get("llm_provider", "openai")
    if provider == "google":
        if not config.get("google_enabled", True):
            # Google wyłączony, zwróć pusty klucz
            return provider, "", _resolve_google_model("", config.get("google_model", "auto"))
        api_key = config.get_google_api_key()
        model = _resolve_google_model(api_key, config.get("google_model", "auto"))
    else:
        if not config.get("openai_enabled", True):
            # OpenAI wyłączony, zwróć pusty klucz
            return provider, "", _resolve_openai_model("", config.get("openai_model", "auto"))
        api_key = config.get_api_key()
        model = _resolve_openai_model(api_key, config.get("openai_model", "auto"))
    return provider, api_key, model

def get_llm_client(provider: str, api_key: str, model: str):
    if not api_key:
        return None
    if provider == "google":
        if not GOOGLE_GENAI_AVAILABLE:
            return None
        return GoogleAIStudioClient(api_key, model=model)
    from openai import OpenAI
    # Dodaj timeout do klienta OpenAI (30s connect, 60s read)
    return OpenAI(api_key=api_key, timeout=60.0)


def safe_llm_call(func, *args, max_retries: int = 2, **kwargs) -> Tuple[Any, Optional[str]]:
    """
    Bezpieczne wywołanie funkcji LLM z retry i obsługą błędów.

    Returns:
        Tuple[result, error_message] - jeśli error_message is None, wywołanie się powiodło
    """
    import time

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            result = func(*args, **kwargs)
            return result, None
        except Exception as e:
            last_error = e
            error_msg = str(e).lower()

            # Błędy które nie mają sensu retryować
            if any(x in error_msg for x in ["invalid_api_key", "incorrect api key", "authentication"]):
                return None, "Nieprawidłowy klucz API"
            if "insufficient_quota" in error_msg:
                return None, "Brak środków na koncie API"

            # Błędy które warto retryować
            if any(x in error_msg for x in ["rate limit", "rate_limit", "too many requests"]):
                if attempt < max_retries:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                return None, "Przekroczono limit zapytań. Spróbuj ponownie za chwilę."

            if any(x in error_msg for x in ["timeout", "connection", "network"]):
                if attempt < max_retries:
                    time.sleep(1)
                    continue
                return None, "Problem z połączeniem. Sprawdź internet."

            # Inne błędy
            if attempt < max_retries:
                time.sleep(1)
                continue

    return None, f"Błąd API: {str(last_error)[:100]}"


def test_openai_connection(api_key: str, model: str = "auto") -> Tuple[bool, str]:
    """
    Testuje połączenie z OpenAI API.
    Zwraca (success, message).
    """
    if not api_key:
        return False, "Brak klucza API"
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        model = _resolve_openai_model(api_key, model)
        # Prosty test - lista modeli lub krótki request
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Test connection. Reply with: OK"}],
            max_tokens=5,
            temperature=0
        )
        if response.choices and response.choices[0].message.content:
            return True, "Połączono z OpenAI"
        return False, "Brak odpowiedzi z API"
    except Exception as e:
        error_msg = str(e)
        if "Incorrect API key" in error_msg or "invalid_api_key" in error_msg:
            return False, "Nieprawidłowy klucz API"
        elif "Rate limit" in error_msg:
            return False, "Przekroczono limit zapytań"
        elif "insufficient_quota" in error_msg:
            return False, "Brak środków na koncie"
        else:
            return False, f"Błąd: {error_msg[:50]}"


def test_google_connection(api_key: str, model: str = "auto") -> Tuple[bool, str]:
    """
    Testuje połączenie z Google AI Studio (Gemini) API.
    Zwraca (success, message).
    """
    if not api_key:
        return False, "Brak klucza API"
    if not GOOGLE_GENAI_AVAILABLE:
        return False, "Brak biblioteki google-generativeai"
    try:
        genai.configure(api_key=api_key)
        model = _resolve_google_model(api_key, model)
        llm = genai.GenerativeModel(model)
        response = llm.generate_content(
            "Test connection. Reply with: OK",
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=5,
                temperature=0
            )
        )
        if response.text:
            return True, "Połączono z Google AI"
        return False, "Brak odpowiedzi z API"
    except Exception as e:
        error_msg = str(e)
        if "API_KEY_INVALID" in error_msg or "INVALID_ARGUMENT" in error_msg:
            return False, "Nieprawidłowy klucz API"
        elif "RESOURCE_EXHAUSTED" in error_msg:
            return False, "Przekroczono limit zapytań"
        else:
            return False, f"Błąd: {error_msg[:50]}"


def build_topic_job(topic: str, api_key: str, provider: str, model: str, **kwargs) -> Dict:
    job = {
        "topic": topic,
        "api_key": api_key,
        "provider": provider,
        "model": model,
    }
    job.update(kwargs)
    return job

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
def get_evaluator(api_key: str, data_path: str):
    """Cache'owany evaluator. Zwraca (evaluator, error_msg) lub (None, error_msg)."""
    evaluator = YTIdeaEvaluatorV2()

    if not evaluator.initialize(api_key):
        return None, evaluator.get_init_error() or "Nie udało się zainicjalizować evaluatora"

    try:
        evaluator.load_data(data_path)
        evaluator.build_embeddings()
        evaluator.train_models()
        return evaluator, None
    except Exception as e:
        return None, f"Błąd ładowania danych: {str(e)}"

@st.cache_resource
def get_advanced_analytics(data_path: str, provider: str, model: str, _api_key: str = None):
    """Cache'owane advanced analytics"""
    if not ADVANCED_AVAILABLE:
        return None

    client = get_llm_client(provider, _api_key, model) if _api_key else None
    
    analytics = AdvancedAnalytics(openai_client=client)
    df = pd.read_csv(data_path)
    analytics.load_data(df)
    return analytics

# =============================================================================
# SIDEBAR
# =============================================================================

merged_df = load_merged_data()
llm_provider, api_key, llm_model = render_sidebar(
    config=config,
    history=history,
    vault=vault,
    merged_df=merged_df,
    get_youtube_sync=get_youtube_sync,
    show_tooltip=show_tooltip,
    llm_provider_labels=LLM_PROVIDER_LABELS,
    topic_analyzer_available=TOPIC_ANALYZER_AVAILABLE,
    advanced_available=ADVANCED_AVAILABLE,
    competitor_tracker_available=COMPETITOR_TRACKER_AVAILABLE,
    google_api_available=GOOGLE_API_AVAILABLE,
    google_genai_available=GOOGLE_GENAI_AVAILABLE,
    get_openai_model_list=_get_openai_model_list,
    get_google_model_list=_get_google_model_list,
    test_openai_connection=test_openai_connection,
    test_google_connection=test_google_connection,
    get_llm_settings=get_llm_settings,
)

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
# =============================================================================
# TAB: OCEŃ POMYSŁ
# =============================================================================

with tab_evaluate:
    render_evaluate_tab(
        merged_df=merged_df,
        llm_provider=llm_provider,
        api_key=api_key,
        llm_model=llm_model,
        vault=vault,
        history=history,
        build_topic_job=build_topic_job,
        resolve_google_model=_resolve_google_model,
        resolve_openai_model=_resolve_openai_model,
        get_llm_client=get_llm_client,
        get_topic_evaluator=get_topic_evaluator,
        cache_get=_cache_get,
        cache_set=_cache_set,
        make_cache_key=_make_cache_key,
        get_cache_ttl=get_cache_ttl,
        record_llm_call=record_llm_call,
        log_diagnostic=log_diagnostic,
        get_wiki_api=get_wiki_api,
        get_news_checker=get_news_checker,
        get_seasonality=get_seasonality,
        get_trend_discovery=get_trend_discovery,
        trends_analyzer_cls=TrendsAnalyzer,
        topic_analyzer_available=TOPIC_ANALYZER_AVAILABLE,
        advanced_available=ADVANCED_AVAILABLE,
        llm_provider_labels=LLM_PROVIDER_LABELS,
        cache_version=CACHE_VERSION,
    )

# =============================================================================
# =============================================================================
# TAB: NARZĘDZIA
# =============================================================================

with tab_tools:
    render_tools_tab(
        merged_df=merged_df,
        config=config,
        vault=vault,
        alerts=alerts,
        llm_provider=llm_provider,
        api_key=api_key,
        llm_model=llm_model,
        advanced_available=ADVANCED_AVAILABLE,
        competitor_tracker_available=COMPETITOR_TRACKER_AVAILABLE,
        google_api_available=GOOGLE_API_AVAILABLE,
        get_llm_client=get_llm_client,
        get_youtube_sync=get_youtube_sync,
        get_competitor_tracker=get_competitor_tracker,
        get_competitor_manager=get_competitor_manager,
        get_trend_discovery=get_trend_discovery,
        trends_analyzer_cls=TrendsAnalyzer,
        content_gap_finder_cls=ContentGapFinder,
        wtopa_analyzer_cls=WtopaAnalyzer,
        hook_analyzer_cls=HookAnalyzer,
        competition_scanner_cls=CompetitionScanner,
        packaging_dna_cls=PackagingDNA,
        timing_predictor_cls=TimingPredictor,
        promise_generator_cls=PromiseGenerator,
        llm_provider_labels=LLM_PROVIDER_LABELS,
    )

# =============================================================================
# TAB: ANALYTICS
# =============================================================================

with tab_analytics:
    render_analytics_tab(
        merged_df=merged_df,
        history=history,
        advanced_available=ADVANCED_AVAILABLE,
        get_advanced_analytics=get_advanced_analytics,
        llm_provider=llm_provider,
        llm_model=llm_model,
        api_key=api_key,
        channel_data_dir=CHANNEL_DATA_DIR,
        merged_data_file=MERGED_DATA_FILE,
    )

# =============================================================================
# =============================================================================
# TAB: HISTORIA
# =============================================================================

with tab_history:
    render_history_tab(history, vault)

# =============================================================================
# =============================================================================
# TAB: IDEA VAULT
# =============================================================================

with tab_vault:
    render_vault_tab(vault)

# =============================================================================
# =============================================================================
# TAB: DANE
# =============================================================================

with tab_data:
    render_data_tab(
        config=config,
        merged_df=merged_df,
        validate_channel_dataframe=validate_channel_dataframe,
        load_manual_scripts=load_manual_scripts,
        get_youtube_sync=get_youtube_sync,
        channel_data_dir=CHANNEL_DATA_DIR,
        merged_data_file=MERGED_DATA_FILE,
        script_sync_dir=SCRIPT_SYNC_DIR,
        google_api_available=GOOGLE_API_AVAILABLE,
    )

# =============================================================================
# =============================================================================
# TAB: DIAGNOSTYKA
# =============================================================================

with tab_diag:
    render_diagnostics_tab(
        merged_df=merged_df,
        show_tooltip=show_tooltip,
        validate_channel_dataframe=validate_channel_dataframe,
        load_cache_store=_load_cache_store,
        get_youtube_sync=get_youtube_sync,
        topic_analyzer_available=TOPIC_ANALYZER_AVAILABLE,
        advanced_available=ADVANCED_AVAILABLE,
        competitor_tracker_available=COMPETITOR_TRACKER_AVAILABLE,
        google_api_available=GOOGLE_API_AVAILABLE,
        google_genai_available=GOOGLE_GENAI_AVAILABLE,
    )

# =============================================================================
# FOOTER
# =============================================================================

st.divider()
st.caption("YT Idea Evaluator Pro v3 | Made for Dawid 🎬")
