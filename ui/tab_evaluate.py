"""Evaluate tab rendering."""

from __future__ import annotations

from datetime import datetime
from typing import Callable, Dict

import pandas as pd
import streamlit as st


def render_evaluate_tab(
    *,
    merged_df: pd.DataFrame | None,
    llm_provider: str,
    api_key: str,
    llm_model: str,
    vault,
    history,
    build_topic_job: Callable,
    resolve_google_model: Callable[[str, str], str],
    resolve_openai_model: Callable[[str, str], str],
    get_llm_client: Callable,
    get_topic_evaluator: Callable,
    cache_get: Callable,
    cache_set: Callable,
    make_cache_key: Callable[[dict], str],
    get_cache_ttl: Callable[[str, int], int],
    record_llm_call: Callable[[str, bool], None],
    log_diagnostic: Callable[[str, str], None],
    get_wiki_api: Callable,
    get_news_checker: Callable,
    get_seasonality: Callable,
    get_trend_discovery: Callable,
    trends_analyzer_cls: Callable,
    topic_analyzer_available: bool,
    advanced_available: bool,
    llm_provider_labels: dict,
    cache_version: str,
) -> None:
    """Render the evaluate tab."""
    st.header("🧭 Topic Workspace")
    st.caption(
        "Jedno wejście: temat. Jeden wynik: tytuły, obietnice, trendy, konkurencja, podobne hity, viral score i timeline."
    )

    if "topic_job_main" not in st.session_state:
        st.session_state.topic_job_main = None
    if "topic_result_main" not in st.session_state:
        st.session_state.topic_result_main = None

    col_t1, col_t2, col_t3 = st.columns([3, 1, 1])
    with col_t1:
        topic_input_main = st.text_input(
            "🧠 Temat / hasło",
            placeholder="np. Operacja Northwoods",
            key="main_topic_input",
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
                help="Sprawdza nasycenie rynku i liczbę podobnych treści w YouTube.",
            )
            inc_similar = st.checkbox(
                "Podobne hity (kanał)",
                value=True,
                key="main_inc_sim",
                help="Porównuje temat do Twoich historycznych hitów na kanale.",
            )
        with c2:
            inc_trends = st.checkbox(
                "Google Trends",
                value=True,
                key="main_inc_trends",
                help="Ocena trendu wyszukiwań w Google dla tematu.",
            )
            inc_external = st.checkbox(
                "Źródła zewnętrzne (Wiki/News)",
                value=True,
                key="main_inc_ext",
                help="Sprawdza świeżość i zainteresowanie z Wikipedii i newsów.",
            )
        with c3:
            inc_viral = st.checkbox(
                "Viral Score",
                value=True,
                key="main_inc_viral",
                help="Predykcja wiralowości na podstawie tytułu/tematu.",
            )
            inc_timeline = st.checkbox(
                "Performance Timeline",
                value=True,
                key="main_inc_timeline",
                help="Prognoza wyświetleń po 1/7/30 dniach.",
            )

    with st.expander("🧪 Batch: oceń wiele tematów naraz", expanded=False):
        batch_raw = st.text_area(
            "Wklej listę tematów (1 temat = 1 linia)",
            height=120,
            key="batch_topics_input",
        )
        batch_limit = st.slider("Limit tematów", 1, 10, 5, key="batch_topics_limit")
        if st.button("⚡ Szybka ocena listy", key="batch_topics_run"):
            topics = [t.strip() for t in (batch_raw or "").splitlines() if t.strip()]
            topics = topics[:batch_limit]
            results = []
            for topic in topics:
                job = build_topic_job(
                    topic,
                    api_key,
                    llm_provider,
                    llm_model,
                    n_titles=3,
                    n_promises=3,
                    inc_competition=inc_competition,
                    inc_similar=inc_similar,
                    inc_trends=inc_trends,
                    inc_external=inc_external,
                    inc_viral=inc_viral,
                    inc_timeline=inc_timeline,
                    result={"topic": topic, "timestamp": datetime.now().isoformat()},
                )
                for stg in [0, 1, 2, 5]:
                    job = _topic_stage_run(stg, job)
                res = job.get("result", {})
                results.append(
                    {
                        "topic": topic,
                        "score": res.get("overall_score", res.get("overall_score_base", 0)),
                        "best_title": (res.get("selected_title") or {}).get("title", ""),
                        "recommendation": res.get("recommendation", ""),
                    }
                )
            if results:
                st.session_state["batch_topic_results"] = results
            else:
                st.info("Brak tematów do oceny.")
        if st.session_state.get("batch_topic_results"):
            st.markdown("#### Wyniki batch")
            st.dataframe(st.session_state["batch_topic_results"], use_container_width=True)

    def _score_badge(score: int, tooltip: str) -> str:
        safe_tip = (
            (tooltip or "")
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )
        return f"<span class='score-badge' title=\"{safe_tip}\">{score}</span>"

    def _estimate_timeline(final_score: int, similar_hits: list) -> dict:
        try:
            views_candidates = [
                h.get("views", 0)
                for h in (similar_hits or [])
                if isinstance(h.get("views", 0), (int, float))
            ]
            views_candidates = [v for v in views_candidates if v and v > 0]
            if views_candidates:
                base_30 = int(sorted(views_candidates)[len(views_candidates) // 2])
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
            base_30 = int(10000 * (0.5 + final_score / 100.0))

        day1 = int(base_30 * 0.18)
        day7 = int(base_30 * 0.58)
        day30 = int(base_30)
        return {
            "day_1": day1,
            "day_7": day7,
            "day_30": day30,
            "basis": "similar_hits" if similar_hits else "channel_distribution",
        }

    def _topic_stage_run(stage: int, job: dict) -> dict:
        """Uruchamia jeden etap oceny tematu, zapisuje wynik w job['result']."""
        topic = job.get("topic", "")
        if not topic:
            return job

        api_key_local = job.get("api_key", "") or ""
        provider_local = job.get("provider", "openai")
        model_local = job.get("model", "auto")
        if provider_local == "google":
            model_local = resolve_google_model(api_key_local, model_local)
        else:
            model_local = resolve_openai_model(api_key_local, model_local)
        client = get_llm_client(provider_local, api_key_local, model_local)

        if not topic_analyzer_available:
            job["error"] = "Topic analyzer niedostępny. Sprawdź pliki i requirements."
            return job

        evaluator = get_topic_evaluator(client, merged_df)
        res = job.get("result", {"topic": topic, "timestamp": datetime.now().isoformat()})

        if stage == 0:
            if job.get("inc_trends", True) and advanced_available:
                cache_key = make_cache_key({"topic": topic, "stage": "trends", "v": cache_version})
                cached_trend = cache_get(
                    "trends", cache_key, ttl_seconds=get_cache_ttl("trends", 6 * 3600)
                )
                if cached_trend is None:
                    try:
                        ta = trends_analyzer_cls()
                        cached_trend = ta.check_trend([topic])
                        cache_set("trends", cache_key, cached_trend)
                    except Exception as exc:
                        cached_trend = {"status": "ERROR", "message": str(exc)}
                        log_diagnostic(f"Trends error: {exc}", "error")
                res["trends"] = cached_trend
            if job.get("inc_external", True):
                cache_key = make_cache_key({"topic": topic, "stage": "external", "v": cache_version})
                cached_external = cache_get(
                    "external",
                    cache_key,
                    ttl_seconds=get_cache_ttl("external", 12 * 3600),
                )
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
                        cache_set("external", cache_key, cached_external)
                    except Exception as exc:
                        cached_external = {"error": str(exc)}
                        log_diagnostic(f"External sources error: {exc}", "error")
                res["external_data"] = cached_external
            job["result"] = res
            job["stage_done"] = 0
            return job

        if stage == 1:
            if job.get("inc_competition", True):
                cache_key = make_cache_key({"topic": topic, "stage": "competition", "v": cache_version})
                cached_comp = cache_get(
                    "competition", cache_key, ttl_seconds=get_cache_ttl("competition", 6 * 3600)
                )
                if cached_comp is None:
                    cached_comp = evaluator.competitor_analyzer.analyze(topic)
                    cache_set("competition", cache_key, cached_comp)
                res["competition"] = cached_comp
            job["result"] = res
            job["stage_done"] = 1
            return job

        if stage == 2:
            if job.get("inc_similar", True) and evaluator.similar_finder:
                cache_key = make_cache_key({"topic": topic, "stage": "similar", "v": cache_version})
                cached_similar = cache_get(
                    "similar_hits", cache_key, ttl_seconds=get_cache_ttl("similar_hits", 6 * 3600)
                )
                if cached_similar is None:
                    cached_similar = evaluator.similar_finder.find(topic, topic)
                    cache_set("similar_hits", cache_key, cached_similar)
                res["similar_hits"] = cached_similar
            job["result"] = res
            job["stage_done"] = 2
            return job

        if stage == 3:
            use_ai = bool(client)
            cache_key = make_cache_key(
                {
                    "topic": topic,
                    "stage": "titles",
                    "n_titles": job.get("n_titles", 6),
                    "use_ai": use_ai,
                    "v": cache_version,
                }
            )
            cached_titles = cache_get(
                "llm_titles", cache_key, ttl_seconds=get_cache_ttl("llm_titles", 24 * 3600)
            )
            if cached_titles is None:
                cached_titles = evaluator.title_generator.generate(
                    topic,
                    n=job.get("n_titles", 6),
                    use_ai=use_ai,
                )
                cache_set("llm_titles", cache_key, cached_titles)
                record_llm_call("titles")
            else:
                record_llm_call("titles", cached=True)
            res["titles"] = cached_titles
            if res.get("titles"):
                res["selected_title"] = res["titles"][0]
            job["result"] = res
            job["stage_done"] = 3
            return job

        if stage == 4:
            use_ai = bool(client)
            best_title = (res.get("selected_title") or {}).get("title") or topic
            cache_key = make_cache_key(
                {
                    "topic": topic,
                    "title": best_title,
                    "stage": "promises",
                    "n_promises": job.get("n_promises", 6),
                    "use_ai": use_ai,
                    "v": cache_version,
                }
            )
            cached_promises = cache_get(
                "llm_promises", cache_key, ttl_seconds=get_cache_ttl("llm_promises", 24 * 3600)
            )
            if cached_promises is None:
                cached_promises = evaluator.promise_generator.generate(
                    best_title,
                    topic,
                    n=job.get("n_promises", 6),
                    use_ai=use_ai,
                )
                cache_set("llm_promises", cache_key, cached_promises)
                record_llm_call("promises")
            else:
                record_llm_call("promises", cached=True)
            res["promises"] = cached_promises
            job["result"] = res
            job["stage_done"] = 4
            return job

        if stage == 5:
            best_title = (res.get("selected_title") or {}).get("title") or topic
            if job.get("inc_viral", True):
                res["viral_score"] = evaluator.viral_predictor.predict(
                    best_title, topic, res.get("competition", {})
                )

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
                    views_vals = [
                        h.get("views", 0)
                        for h in res["similar_hits"]
                        if isinstance(h.get("views", 0), (int, float))
                    ]
                    views_vals = [v for v in views_vals if v > 0]
                    if views_vals:
                        median_views = int(sorted(views_vals)[len(views_vals) // 2])
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
        "Podsumowanie",
    ]

    stage_done = None
    if st.session_state.topic_job_main:
        stage_done = st.session_state.topic_job_main.get("stage_done")
    st.caption(f"Etapy: {' → '.join(stage_names)}")
    if stage_done is not None:
        st.caption(f"Aktualny etap: {stage_names[min(stage_done, len(stage_names) - 1)]}")

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
            st.error(
                f"❌ Brak API Key ({llm_provider_labels.get(llm_provider, 'LLM')}). Dodaj klucz w sidebarze."
            )
        elif merged_df is None:
            st.warning("⚠️ Najpierw wczytaj dane kanału (zakładka: Dane).")
        else:
            job = build_topic_job(
                topic_input_main.strip(),
                api_key,
                llm_provider,
                llm_model,
                n_titles=n_titles_main,
                n_promises=n_promises_main,
                inc_competition=inc_competition,
                inc_similar=inc_similar,
                inc_trends=inc_trends,
                inc_external=inc_external,
                inc_viral=inc_viral,
                inc_timeline=inc_timeline,
                result={"topic": topic_input_main.strip(), "timestamp": datetime.now().isoformat()},
            )
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
                topic=res.get("topic", ""),
                payload=res,
                status="new",
            )
            history.add(
                {
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
                    "tags": ["topic_mode", "one_click"],
                    "status": "new",
                }
            )
            st.success("✅ Zapisano (Vault + Historia).")

    if start_step:
        if not topic_input_main:
            st.warning("Wpisz temat.")
        elif not api_key:
            st.error(
                f"❌ Brak API Key ({llm_provider_labels.get(llm_provider, 'LLM')}). Dodaj klucz w sidebarze."
            )
        else:
            job = build_topic_job(
                topic_input_main.strip(),
                api_key,
                llm_provider,
                llm_model,
                n_titles=n_titles_main,
                n_promises=n_promises_main,
                inc_competition=inc_competition,
                inc_similar=inc_similar,
                inc_trends=inc_trends,
                inc_external=inc_external,
                inc_viral=inc_viral,
                inc_timeline=inc_timeline,
                result={"topic": topic_input_main.strip(), "timestamp": datetime.now().isoformat()},
            )
            st.session_state.topic_job_main = _topic_stage_run(0, job)
            st.session_state.topic_result_main = st.session_state.topic_job_main.get("result")
            st.rerun()

    if full_run:
        if not topic_input_main:
            st.warning("Wpisz temat.")
        elif not api_key:
            st.error(
                f"❌ Brak API Key ({llm_provider_labels.get(llm_provider, 'LLM')}). Dodaj klucz w sidebarze."
            )
        elif merged_df is None:
            st.warning("⚠️ Najpierw wczytaj dane kanału (zakładka: Dane).")
        else:
            job = build_topic_job(
                topic_input_main.strip(),
                api_key,
                llm_provider,
                llm_model,
                n_titles=n_titles_main,
                n_promises=n_promises_main,
                inc_competition=inc_competition,
                inc_similar=inc_similar,
                inc_trends=inc_trends,
                inc_external=inc_external,
                inc_viral=inc_viral,
                inc_timeline=inc_timeline,
                result={"topic": topic_input_main.strip(), "timestamp": datetime.now().isoformat()},
            )
            with st.spinner("Oceniam temat..."):
                for stg in [0, 1, 2, 3, 4, 5]:
                    job = _topic_stage_run(stg, job)
            st.session_state.topic_job_main = job
            st.session_state.topic_result_main = job.get("result")
            st.rerun()

    if cont_step:
        if not st.session_state.topic_job_main:
            st.warning("Najpierw użyj Start krokowo.")
        else:
            job = st.session_state.topic_job_main
            stage_done = job.get("stage_done", -1)
            next_stage = min(stage_done + 1, 5)
            st.session_state.topic_job_main = _topic_stage_run(next_stage, job)
            st.session_state.topic_result_main = st.session_state.topic_job_main.get("result")
            st.rerun()

    if stop_and_save:
        st.success("Zapisano stan częściowy.")

    if st.session_state.topic_result_main:
        res = st.session_state.topic_result_main or {}

        st.markdown("### 📌 Podsumowanie")
        final_score = res.get("overall_score", res.get("overall_score_base", 0))
        trend_bonus = int(res.get("trend_bonus", 0))
        similar_bonus = int(res.get("similar_bonus", 0))
        base_overall = int(res.get("overall_score_base", 0))
        st.markdown(
            f"**Ocena końcowa:** {final_score}/100  |  Base: {base_overall}  |  Trend: {trend_bonus:+d}  |  Similar: {similar_bonus:+d}"
        )

        if res.get("recommendation"):
            st.markdown("**Rekomendacja:**")
            st.info(res.get("recommendation", ""))

        st.markdown("#### 🧾 Szybkie uzasadnienie (3 punkty)")
        reasons = [
            f"Base score: {base_overall}/100 (siła tytułu + sygnały bazowe)",
            f"Trend bonus: {trend_bonus:+d} (Trends/External)",
            f"Similar bonus: {similar_bonus:+d} (dopasowanie do hitów kanału)",
        ]
        for reason in reasons:
            st.markdown(f"- {reason}")

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
                    "result": {"topic": res.get("topic", ""), "timestamp": datetime.now().isoformat()},
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
                    "result": {"topic": res.get("topic", ""), "timestamp": datetime.now().isoformat()},
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
                    topic=res.get("topic", ""),
                    payload=res,
                    status="new",
                )
                st.success("✅ Zapisano do Vault.")

        st.markdown("### Proponowane tytuły")
        if res.get("titles"):
            titles = res["titles"]
            top_titles = titles[:3]
            st.markdown("**Top 3 (na start):**")
            for title in top_titles:
                reason = title.get("reasoning") or title.get("reason") or title.get("calculated_reasoning", "")
                short_reason = f"{reason.split('.')[0]}." if reason else ""
                badge = _score_badge(int(title.get("score", 0)), reason)
                st.markdown(
                    f"{badge} <b>{title.get('title','')}</b> — {short_reason}",
                    unsafe_allow_html=True,
                )
            title_opts = [title.get("title", "") for title in titles]
            selected_title_str = st.radio(
                "Wybierz tytuł do dalszej oceny",
                title_opts,
                index=0,
                key="main_selected_title",
                horizontal=False,
            )
            selected_obj = next((title for title in titles if title.get("title") == selected_title_str), titles[0])
            res["selected_title"] = selected_obj

            reason = selected_obj.get("reasoning") or selected_obj.get("reason") or selected_obj.get("calculated_reasoning", "")
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.markdown(f"**Wybrany tytuł:** {selected_obj.get('title','')}")
            if reason:
                st.markdown(reason)
            st.caption(
                f"Źródło: {selected_obj.get('source','?')} | Styl: {selected_obj.get('style','?')}"
            )
            st.markdown("</div>", unsafe_allow_html=True)

            with st.expander("Pełna lista tytułów", expanded=False):
                for title in titles:
                    reason = title.get("reasoning") or title.get("reason") or title.get("calculated_reasoning", "")
                    badge = _score_badge(int(title.get("score", 0)), reason)
                    st.markdown(f"{badge} <b>{title.get('title','')}</b>", unsafe_allow_html=True)
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
                    st.warning(
                        f"Dodaj API Key ({llm_provider_labels.get(llm_provider, 'LLM')}) aby przepisać tytuł."
                    )
                else:
                    try:
                        client = get_llm_client(llm_provider, api_key, llm_model)
                        if not client:
                            st.warning(
                                f"Nie udało się zainicjalizować {llm_provider_labels.get(llm_provider, 'LLM')}."
                            )
                            raise RuntimeError("Brak klienta LLM.")
                        prompt = (
                            f"Napisz 1 wariant tytułu w stylu: {rewrite_style}. Oryginał: {selected_title_str}"
                        )
                        resp = client.chat.completions.create(
                            model=llm_model,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.7,
                        )
                        rewritten = resp.choices[0].message.content.strip()
                        st.success(rewritten)
                    except Exception as exc:
                        st.warning(f"Nie udało się przepisać tytułu: {exc}")

            st.markdown("#### ⚖️ Porównaj 2 tytuły (szybko)")
            cmp1, cmp2 = st.columns(2)
            with cmp1:
                title_a = st.text_input("Tytuł A", value=title_opts[0], key="compare_title_a")
            with cmp2:
                title_b = st.text_input(
                    "Tytuł B",
                    value=title_opts[1] if len(title_opts) > 1 else "",
                    key="compare_title_b",
                )
            if st.button("Porównaj tytuły", key="compare_titles"):
                if not api_key:
                    st.warning(
                        f"Dodaj API Key ({llm_provider_labels.get(llm_provider, 'LLM')}) aby porównać tytuły."
                    )
                else:
                    try:
                        client = get_llm_client(llm_provider, api_key, llm_model)
                        if not client:
                            st.warning(
                                f"Nie udało się zainicjalizować {llm_provider_labels.get(llm_provider, 'LLM')}."
                            )
                            raise RuntimeError("Brak klienta LLM.")
                        prompt = (
                            f"Porównaj dwa tytuły i wskaż lepszy. A: {title_a} B: {title_b}. Zwróć krótki werdykt."
                        )
                        resp = client.chat.completions.create(
                            model=llm_model,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.3,
                        )
                        verdict = resp.choices[0].message.content.strip()
                        st.info(verdict)
                    except Exception as exc:
                        st.warning(f"Nie udało się porównać tytułów: {exc}")
        else:
            st.warning("Brak wygenerowanych tytułów.")

        st.markdown("### Proponowane obietnice (hook/promise)")
        if res.get("promises"):
            promises = res["promises"]
            top_promises = promises[:3]
            st.markdown("**Top 3 (na start):**")
            for promise in top_promises:
                reason = promise.get("reasoning") or promise.get("reason") or ""
                short_reason = f"{reason.split('.')[0]}." if reason else ""
                badge = _score_badge(int(promise.get("score", 0)), reason)
                st.markdown(
                    f"{badge} {promise.get('promise','')} — {short_reason}",
                    unsafe_allow_html=True,
                )
            promise_opts = [promise.get("promise", "") for promise in promises]
            chosen_promise = st.radio(
                "Wybierz obietnicę",
                promise_opts,
                index=0,
                key="main_selected_promise",
            )
            selected_promise_obj = next(
                (promise for promise in promises if promise.get("promise") == chosen_promise),
                promises[0],
            )
            reason = selected_promise_obj.get("reasoning") or selected_promise_obj.get("reason") or ""
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.markdown(f"**Wybrana obietnica:** {selected_promise_obj.get('promise','')}")
            if reason:
                st.markdown(reason)
            st.caption(f"Źródło: {selected_promise_obj.get('source','?')}")
            st.markdown("</div>", unsafe_allow_html=True)

            with st.expander("Pełna lista obietnic", expanded=False):
                for promise in promises:
                    reason = promise.get("reasoning") or promise.get("reason") or ""
                    badge = _score_badge(int(promise.get("score", 0)), reason)
                    st.markdown(f"{badge} {promise.get('promise','')}", unsafe_allow_html=True)
                    if reason:
                        st.caption(reason)

            st.markdown("#### 🧪 Szybka ocena hooka")
            hook_text = st.text_area("Wklej hook (2-3 zdania)", key="hook_quick_text", height=80)
            if st.button("Oceń hook", key="hook_quick_btn"):
                if not api_key:
                    st.warning(
                        f"Dodaj API Key ({llm_provider_labels.get(llm_provider, 'LLM')}) aby ocenić hook."
                    )
                else:
                    try:
                        client = get_llm_client(llm_provider, api_key, llm_model)
                        if not client:
                            st.warning(
                                f"Nie udało się zainicjalizować {llm_provider_labels.get(llm_provider, 'LLM')}."
                            )
                            raise RuntimeError("Brak klienta LLM.")
                        prompt = f"Oceń hook (0-100) i podaj 1 zdanie uzasadnienia. Hook: {hook_text}"
                        resp = client.chat.completions.create(
                            model=llm_model,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.4,
                        )
                        st.info(resp.choices[0].message.content.strip())
                    except Exception as exc:
                        st.warning(f"Nie udało się ocenić hooka: {exc}")
        else:
            st.info("Brak obietnic. Uruchom etap generacji obietnic.")

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
                for hit in res["similar_hits"][:8]:
                    st.markdown(
                        f"- {hit.get('title','')} | {hit.get('views',0):,} views | {hit.get('label','')}"
                    )
                    pub = hit.get("published_at") or hit.get("publishedAt")
                    if pub:
                        try:
                            pub_dt = pd.to_datetime(pub, errors="coerce")
                            if pd.notna(pub_dt) and (datetime.now() - pub_dt).days <= 30:
                                recent_warning = hit.get("title", "")
                        except Exception:
                            pass
                if recent_warning:
                    st.warning(f"⚠️ Ten temat może kanibalizować świeży film: {recent_warning}")
            else:
                st.caption("Brak lub nie wczytano.")

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
                    topic=res.get("topic", ""),
                    payload=res,
                    status="new",
                )
                history.add(
                    {
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
                        "status": "new",
                    }
                )
                st.success("✅ Zapisano (Vault + Historia).")
        with col_s2:
            if st.button("🗑️ Wyczyść wynik tematu", key="main_clear_topic"):
                st.session_state.topic_result_main = None
                st.session_state.topic_job_main = None
