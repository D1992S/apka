"""Tools tab rendering."""

from __future__ import annotations

from datetime import datetime
import re
from typing import Callable

import pandas as pd
import streamlit as st


def render_tools_tab(
    *,
    merged_df: pd.DataFrame | None,
    config,
    vault,
    alerts,
    llm_provider: str,
    api_key: str,
    llm_model: str,
    advanced_available: bool,
    competitor_tracker_available: bool,
    google_api_available: bool,
    get_llm_client: Callable,
    get_youtube_sync: Callable,
    get_competitor_tracker: Callable,
    get_competitor_manager: Callable,
    get_trend_discovery: Callable,
    trends_analyzer_cls: Callable,
    content_gap_finder_cls: Callable,
    wtopa_analyzer_cls: Callable,
    hook_analyzer_cls: Callable,
    competition_scanner_cls: Callable,
    packaging_dna_cls: Callable,
    timing_predictor_cls: Callable,
    promise_generator_cls: Callable,
    llm_provider_labels: dict,
) -> None:
    """Render the tools tab."""
    st.header("🛠️ Narzędzia")

    tool_tabs = st.tabs(
        [
            "📉 Dlaczego wtopa?",
            "🕳️ Content Gaps",
            "📅 Kalendarz",
            "🔔 Trend Alerts",
            "👀 Competitor Tracker",
        ]
    )

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

            def _safe_int(value, default: int = 0) -> int:
                if value is None or (isinstance(value, float) and pd.isna(value)):
                    return default
                try:
                    if isinstance(value, str) and not value.strip():
                        return default
                    return int(value)
                except (TypeError, ValueError):
                    return default

            def _fmt(idx):
                row = df_w.loc[idx]
                views = _safe_int(row.get("views", 0))
                return f"{str(row.get('title',''))[:80]} | {views:,} views"

            pick = st.selectbox("Film", options=opts, format_func=_fmt, key="wtopa_pick")
            if st.button("Załaduj dane filmu", key="wtopa_load_btn"):
                row = df_w.loc[pick]
                st.session_state["wtopa_title"] = str(row.get("title", ""))
                st.session_state["wtopa_views"] = _safe_int(row.get("views", 0))

                ret = None
                for col in [
                    "retention",
                    "avg_view_percentage",
                    "average_view_percentage",
                    "avg_view_duration_ratio",
                ]:
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
            wtopa_target = st.text_input(
                "Co miało się wydarzyć? (Twoje założenie)",
                placeholder="np. 50k views, 45% retencji",
            )
            wtopa_notes = st.text_area(
                "Dodatkowe notatki",
                height=80,
                placeholder="np. zmiana stylu, temat był ryzykowny, thumbnail A/B itp.",
            )

        if st.button("🔎 Przeanalizuj", use_container_width=True, key="run_wtopa"):
            if not advanced_available:
                st.error("Moduł advanced_analytics niedostępny.")
            else:
                with st.spinner("Analizuję..."):
                    analyzer = wtopa_analyzer_cls(
                        merged_df, get_llm_client(llm_provider, api_key, llm_model) if api_key else None
                    )
                    result = analyzer.analyze(
                        wtopa_title,
                        int(wtopa_views),
                        float(wtopa_retention),
                        {"target": wtopa_target, "notes": wtopa_notes},
                    )
                    st.session_state["last_wtopa"] = result

        if st.session_state.get("last_wtopa"):
            res = st.session_state["last_wtopa"]
            st.markdown("### 🧠 Wynik analizy")
            st.info(res.get("summary", ""))

            if res.get("insights"):
                st.markdown("### 🔍 Insights")
                for insight in res["insights"]:
                    st.markdown(f"- {insight}")

            if res.get("suggestions"):
                st.markdown("### ✅ Sugestie naprawy")
                for sugg in res["suggestions"]:
                    st.markdown(f"- {sugg}")

    # --- CONTENT GAPS ---
    with tool_tabs[1]:
        st.subheader("🕳️ Content Gaps")
        st.markdown("Znajdź luki w treściach Twojego kanału")

        if merged_df is None or "title" not in merged_df.columns:
            st.warning("Brak danych kanału. Załaduj CSV w zakładce Dane.")
        else:
            if st.button("🔍 Znajdź luki", key="run_gap_finder"):
                with st.spinner("Szukam luk w treściach..."):
                    finder = content_gap_finder_cls(merged_df)
                    gaps = finder.find_gaps()
                    st.session_state["content_gaps"] = gaps

            gaps = st.session_state.get("content_gaps", [])
            if gaps:
                for gap in gaps:
                    st.markdown(f"- {gap}")
            else:
                st.info("Brak wyników. Kliknij przycisk aby uruchomić analizę.")

    # --- KALENDARZ ---
    with tool_tabs[2]:
        st.subheader("📅 Kalendarz publikacji")
        st.markdown("Automatyczny plan publikacji na podstawie wyników")

        plan_topics = st.text_area(
            "Lista tematów (1 temat = 1 linia)",
            height=150,
            placeholder="np. Operacja Northwoods\nCicada 3301\nZagadkowe katastrofy",
            key="calendar_topics",
        )
        weeks = st.slider("Ile tygodni planu", 1, 8, 4)

        if st.button("🧠 Generuj plan", key="generate_calendar"):
            topics = [t.strip() for t in plan_topics.splitlines() if t.strip()]
            if not topics:
                st.warning("Dodaj przynajmniej 1 temat.")
            else:
                with st.spinner("Tworzę plan..."):
                    plan_items = []
                    for idx, topic in enumerate(topics):
                        plan_items.append(
                            {
                                "topic": topic,
                                "week": idx % weeks + 1,
                                "angle": "",
                                "why": "",
                                "source": "manual",
                            }
                        )
                    st.session_state["calendar_plan"] = plan_items

        plan_items = st.session_state.get("calendar_plan", [])
        if plan_items:
            st.markdown("### 🗓️ Plan publikacji")
            for week in range(1, weeks + 1):
                st.markdown(f"### Tydzień {week}")
                week_items = [x for x in plan_items if int(x.get("week", 1)) == week]
                for idx, item in enumerate(week_items, start=1):
                    topic = item.get("topic", "")
                    angle = item.get("angle", "")
                    why = item.get("why", "")

                    with st.expander(f"{idx}. {topic}", expanded=False):
                        st.markdown(f"**Temat:** {topic}")
                        if angle:
                            st.markdown(f"**Kąt:** {angle}")
                        if why:
                            st.markdown(f"**Dlaczego:** {why}")
                        if item.get("source"):
                            st.caption(f"Źródło pomysłu: {item.get('source')}")

    # --- TREND ALERTS ---
    with tool_tabs[3]:
        st.subheader("🔔 Trend Alerts")
        st.markdown("Monitoruj tematy i otrzymuj powiadomienia gdy zaczną trendować")

        st.markdown("### 📈 Trend Discovery: co trenduje teraz w niszy")
        st.caption(
            "Klikasz, a apka zbiera propozycje tematów (powiązane frazy) i sprawdza je w Google Trends."
        )

        default_seeds = config.get("niche_keywords", ["tajemnice", "zagadki", "spiski", "ufo", "katastrofy"])
        seeds_text = st.text_area(
            "Słowa-klucze niszy (po przecinku)",
            value=", ".join(default_seeds),
            height=70,
            key="niche_seeds_text",
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
                    if advanced_available and top:
                        ta = trends_analyzer_cls()
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
                        topic = item.get("topic", "")
                        det = trend_map.get(topic, {})
                        overall = det.get("overall") or {}
                        rows.append(
                            {
                                "Temat": topic,
                                "Źródło": item.get("source", ""),
                                "Score": item.get("score", 0),
                                "Rekomendacja": item.get("recommendation", ""),
                                "Trend score": overall.get("score", ""),
                                "Trend verdict": overall.get("verdict", ""),
                                "Trend msg": overall.get("message", ""),
                            }
                        )
                    if rows:
                        st.dataframe(rows, use_container_width=True)
                    else:
                        st.info("Brak wyników. Dodaj inne seed keywords.")
                except Exception as exc:
                    st.error(f"Błąd trend discovery: {exc}")

        st.divider()

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
                        st.caption(
                            f"Ostatnio: {alert.get('last_interest', 0)}/100 | {alert.get('last_check', '')[:10]}"
                        )
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
                if advanced_available:
                    trends_analyzer = trends_analyzer_cls()

                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    total_alerts = len(all_alerts)

                    for idx, alert in enumerate(all_alerts):
                        status_text.text(
                            f"Sprawdzam {idx + 1}/{total_alerts}: {alert['topic'][:30]}..."
                        )
                        progress_bar.progress((idx + 1) / max(total_alerts, 1))

                        result = trends_analyzer.check_trend([alert["topic"]])
                        if result.get("status") == "OK":
                            det = (result.get("details") or {}).get(alert["topic"], {})
                            overall = det.get("overall") or {}
                            current = overall.get("score", 0)
                            is_trending = overall.get("verdict") in ["UP", "PEAK"]
                            alerts.update_check(alert["id"], current, is_trending)

                    progress_bar.empty()
                    status_text.empty()
                    st.success("✅ Sprawdzono!")
                    st.rerun()
        else:
            st.info("Brak monitorowanych tematów")

    # --- COMPETITOR TRACKER ---
    with tool_tabs[4]:
        st.subheader("👀 Competitor Tracker")
        st.caption(
            "Dodaj kanały konkurencji i podejrzyj ich ostatnie uploady (z YouTube API lub fallbackiem)."
        )

        if not competitor_tracker_available:
            st.error("Brak modułu competitor_tracker.py lub brak zależności. Sprawdź pliki.")
        else:
            comp_mgr = get_competitor_manager()
            competitors = comp_mgr.list_all()

            st.markdown("### ➕ Dodaj konkurenta")
            c1, c2, c3 = st.columns([2, 2, 1])
            with c1:
                comp_name = st.text_input("Nazwa (opcjonalnie)", placeholder="np. Kanał X", key="comp_add_name")
            with c2:
                comp_channel_id = st.text_input(
                    "Channel ID", placeholder="UCxxxxxxxxxxxxxxxx", key="comp_add_channel_id"
                )
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
                for comp in competitors:
                    cc1, cc2, cc3, cc4 = st.columns([2, 2, 2, 1])
                    with cc1:
                        st.markdown(f"**{comp.get('name','')}**")
                        st.caption(comp.get("channel_id", ""))
                    with cc2:
                        st.caption(comp.get("notes", ""))
                    with cc3:
                        st.caption(f"Dodano: {comp.get('added','')[:10]}")
                    with cc4:
                        if st.button("🗑️", key=f"comp_del_{comp.get('id','')}"):
                            comp_mgr.remove(comp.get("id", ""))
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
                if google_api_available and yt_sync.has_credentials():
                    ok, msg = yt_sync.authenticate()
                    if ok:
                        yt_client = yt_sync.youtube
                        source_hint = "oauth"
                    else:
                        st.warning(msg)

                if not yt_client and google_api_available:
                    yt_api_key = config.get("youtube_api_key", "")
                    if yt_api_key:
                        yt_sync.set_api_key(yt_api_key)
                        if yt_sync.ensure_public_client():
                            yt_client = yt_sync.youtube
                            source_hint = "api_key"
                    else:
                        st.info("Ustaw YouTube API Key w konfiguracji, aby użyć trybu publicznego bez OAuth.")

                tracker = get_competitor_tracker(yt_client)
                uploads = tracker.fetch_recent_uploads(competitors, days=days, max_per_channel=max_per)

                errs = [u for u in uploads if u.get("error")]
                vids = [u for u in uploads if u.get("video_id")]

                if errs:
                    with st.expander("⚠️ Błędy", expanded=False):
                        st.json(errs)

                if vids:
                    st.caption(f"Źródło danych: {source_hint}")

                    def _ts(item):
                        return item.get("publishedAt") or item.get("publishedTime") or ""

                    vids_sorted = sorted(vids, key=_ts, reverse=True)
                    saved_count = comp_mgr.upsert_videos_from_fetch(vids_sorted)
                    if saved_count:
                        st.success(f"✅ Zapisano {saved_count} filmów do listy kanałów.")
                    st.dataframe(vids_sorted, use_container_width=True)
                else:
                    st.info("Brak filmów do wyświetlenia (albo brak danych z API).")

            st.divider()
            st.markdown("### 🎞️ Zapisane filmy konkurencji")

            def _extract_video_id(raw_value: str) -> str:
                raw_value = (raw_value or "").strip()
                if not raw_value:
                    return ""
                if "youtu" not in raw_value:
                    return raw_value
                patterns = [
                    r"v=([a-zA-Z0-9_-]{6,})",
                    r"youtu\.be/([a-zA-Z0-9_-]{6,})",
                    r"shorts/([a-zA-Z0-9_-]{6,})",
                ]
                for pattern in patterns:
                    match = re.search(pattern, raw_value)
                    if match:
                        return match.group(1)
                return raw_value

            if competitors:
                comp_options = {c["id"]: f"{c.get('name','')} ({c.get('channel_id','')})" for c in competitors}
                selected_competitor_id = st.selectbox(
                    "Wybierz kanał",
                    options=list(comp_options.keys()),
                    format_func=lambda cid: comp_options.get(cid, cid),
                    key="comp_videos_pick",
                )

                add_col1, add_col2, add_col3, add_col4 = st.columns([2, 2, 2, 1])
                with add_col1:
                    manual_video = st.text_input(
                        "Video ID lub URL",
                        placeholder="https://www.youtube.com/watch?v=...",
                        key="comp_video_manual_id",
                    )
                with add_col2:
                    manual_title = st.text_input(
                        "Tytuł (opcjonalnie)",
                        placeholder="Tytuł filmu",
                        key="comp_video_manual_title",
                    )
                with add_col3:
                    manual_date = st.text_input(
                        "Data publikacji (opcjonalnie)",
                        placeholder="2024-01-15",
                        key="comp_video_manual_date",
                    )
                with add_col4:
                    if st.button("Dodaj film", key="comp_video_manual_add"):
                        video_id = _extract_video_id(manual_video)
                        if not video_id:
                            st.warning("Podaj poprawny Video ID lub URL.")
                        else:
                            ok = comp_mgr.add_video(
                                competitor_id=selected_competitor_id,
                                video_id=video_id,
                                title=manual_title,
                                published_at=manual_date,
                                source="manual",
                            )
                            if ok:
                                st.success("✅ Dodano film.")
                                st.rerun()
                            else:
                                st.warning("Nie udało się dodać filmu.")

                current_videos = comp_mgr.list_videos(selected_competitor_id)
                if current_videos:
                    def _video_ts(item):
                        return item.get("published_at") or ""

                    for video in sorted(current_videos, key=_video_ts, reverse=True):
                        vcol1, vcol2, vcol3, vcol4 = st.columns([3, 2, 2, 1])
                        with vcol1:
                            st.markdown(f"**{video.get('title') or video.get('video_id')}**")
                            if video.get("url"):
                                st.caption(video.get("url"))
                        with vcol2:
                            st.caption(video.get("published_at", ""))
                        with vcol3:
                            st.caption(video.get("source", ""))
                        with vcol4:
                            if st.button("🗑️", key=f"comp_video_del_{selected_competitor_id}_{video.get('video_id')}"):
                                comp_mgr.remove_video(selected_competitor_id, video.get("video_id", ""))
                                st.rerun()
                else:
                    st.info("Brak zapisanych filmów dla wybranego kanału.")
            else:
                st.info("Dodaj kanał konkurencji, aby zarządzać jego filmami.")
