"""Diagnostics tab rendering."""

import json
from typing import Callable

import pandas as pd
import streamlit as st


def render_diagnostics_tab(
    *,
    merged_df: pd.DataFrame | None,
    show_tooltip: Callable[[str], str],
    validate_channel_dataframe: Callable[[pd.DataFrame], dict],
    load_cache_store: Callable[[], dict],
    get_youtube_sync: Callable,
    topic_analyzer_available: bool,
    advanced_available: bool,
    competitor_tracker_available: bool,
    google_api_available: bool,
    google_genai_available: bool,
) -> None:
    """Render the diagnostics tab."""
    st.header("🧪 Diagnostyka")
    st.caption("Stan modułów, dane, cache i ostatnie błędy.")

    st.subheader("Status modułów")
    diag_modules = {
        "Topic Analyzer": topic_analyzer_available,
        "Advanced Analytics": advanced_available,
        "Competitor Tracker": competitor_tracker_available,
        "YouTube API": google_api_available,
        "Google AI Studio": google_genai_available,
    }
    for name, available in diag_modules.items():
        st.markdown(f"{'✅' if available else '⚠️'} {name}")

    st.subheader("Dane kanału")
    if merged_df is not None:
        present_cols = set(merged_df.columns)
        required_cols = {"title", "views"}
        recommended_cols = {"retention", "label", "published_at"}
        missing_required = sorted(required_cols - present_cols)
        missing_recommended = sorted(recommended_cols - present_cols)

        st.markdown("**Kompletność danych**")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Wymagane", f"{len(required_cols) - len(missing_required)}/{len(required_cols)}")
        with c2:
            st.metric(
                "Rekomendowane",
                f"{len(recommended_cols) - len(missing_recommended)}/{len(recommended_cols)}",
            )
        with c3:
            st.metric("Kolumny razem", f"{len(present_cols)}")

        issues = validate_channel_dataframe(merged_df)
        if issues["missing_required"]:
            st.error(f"Brak wymaganych kolumn: {', '.join(issues['missing_required'])}")
        if issues["missing_recommended"]:
            st.warning(f"Brak rekomendowanych kolumn: {', '.join(issues['missing_recommended'])}")
        for warning in issues["warnings"]:
            st.warning(warning)

        if missing_required or missing_recommended:
            st.markdown("**Jak uzupełnić brakujące dane?**")
            if "title" in missing_required:
                st.info(show_tooltip("channel_title"))
            if "views" in missing_required:
                st.info(show_tooltip("channel_views"))
            if "retention" in missing_recommended:
                st.info(show_tooltip("channel_retention"))
            if "label" in missing_recommended:
                st.info(show_tooltip("channel_label"))
            if "published_at" in missing_recommended:
                st.info(show_tooltip("channel_published_at"))
    else:
        st.info("Brak danych kanału.")

    st.subheader("Sugestie usprawnień analizy")
    tips = []
    if merged_df is None:
        tips.append("Dodaj dane kanału (CSV lub YouTube Sync), aby uruchomić pełne analizy.")
    else:
        if "retention" not in merged_df.columns:
            tips.append("Dodaj retencję, aby lepiej oceniać hook i viral score.")
        if "label" not in merged_df.columns:
            tips.append("Dodaj etykiety PASS/BORDER/FAIL — poprawia dopasowanie do wzorców hitów.")
        if "published_at" not in merged_df.columns:
            tips.append("Dodaj daty publikacji, aby analizy trendu i sezonowości były dokładniejsze.")
    if not google_api_available:
        tips.append("Zainstaluj biblioteki YouTube API, aby korzystać z pełnego syncu.")
    if not topic_analyzer_available:
        tips.append("Włącz moduł Topic Analyzer, aby uruchomić pełny tryb oceny tematu.")
    if not advanced_available:
        tips.append("Włącz Advanced Analytics, aby dostać analizę hooka, DNA i competition scan.")
    tips.append("Jeśli masz transkrypty, użyj syncu z transkryptami — poprawia analizę hooka i retention.")
    tips.append("Dodaj więcej filmów historycznych (min. 30-50), aby model ML był stabilniejszy.")

    if tips:
        for tip in tips:
            st.markdown(f"- {tip}")

    st.subheader("Sync i cache")
    yt_sync = get_youtube_sync()
    last_sync = yt_sync.get_last_sync_time()
    st.caption(f"Ostatnia synchronizacja: {last_sync or 'brak'}")

    cache_store = load_cache_store()
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
            "application/json",
        )
