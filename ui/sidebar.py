"""Sidebar rendering for the Streamlit app."""

from typing import Callable, Tuple

import pandas as pd
import streamlit as st


def render_sidebar(
    *,
    config,
    history,
    vault,
    merged_df: pd.DataFrame | None,
    get_youtube_sync: Callable,
    show_tooltip: Callable[[str], str],
    llm_provider_labels: dict,
    topic_analyzer_available: bool,
    advanced_available: bool,
    competitor_tracker_available: bool,
    google_api_available: bool,
    google_genai_available: bool,
    get_openai_model_list: Callable[[str], list],
    get_google_model_list: Callable[[str], list],
    test_openai_connection: Callable[[str, str], Tuple[bool, str]],
    test_google_connection: Callable[[str, str], Tuple[bool, str]],
    get_llm_settings: Callable[[], Tuple[str, str, str]],
) -> Tuple[str, str, str]:
    """Render the sidebar and return active LLM settings."""
    with st.sidebar:
        st.title("🎬 YT Evaluator Pro")
        st.caption("v3.0 - Kompletna edycja")

        st.divider()

        # === API KEY ===
        st.subheader("🔑 LLM API")

        current_provider = config.get("llm_provider", "openai")
        provider_choice = st.radio(
            "Aktywny dostawca",
            list(llm_provider_labels.keys()),
            format_func=lambda key: llm_provider_labels[key],
            index=list(llm_provider_labels.keys()).index(current_provider),
            help="Wybierz którego dostawcy LLM chcesz używać",
        )
        if provider_choice != current_provider:
            config.set("llm_provider", provider_choice)
            st.rerun()

        # --- OpenAI Section ---
        st.markdown("---")
        openai_col1, openai_col2 = st.columns([3, 1])
        with openai_col1:
            st.markdown("**OpenAI**")
        with openai_col2:
            openai_enabled = st.toggle(
                "ON",
                value=config.get("openai_enabled", True),
                key="openai_toggle",
                help="Włącz/wyłącz OpenAI API",
            )
            if openai_enabled != config.get("openai_enabled", True):
                config.set("openai_enabled", openai_enabled)

        if openai_enabled:
            saved_openai_key = config.get_api_key()
            openai_api_key = st.text_input(
                "OpenAI API Key",
                value=saved_openai_key,
                type="password",
                help="Twój klucz OpenAI API",
                key="openai_api_key_input",
            )
            openai_model = config.get("openai_model", "auto")
            openai_models = ["auto"] + get_openai_model_list(openai_api_key)
            if openai_model not in openai_models:
                openai_models.append(openai_model)
            openai_selected = st.selectbox(
                "Model OpenAI",
                options=openai_models,
                index=openai_models.index(openai_model),
                key="openai_model_input",
                help="Wpisz nazwę modelu lub użyj 'auto' aby dobrać najnowszy dostępny.",
            )
            if openai_selected != openai_model:
                config.set("openai_model", openai_selected)
            if openai_api_key != saved_openai_key:
                if st.button("💾 Zapisz klucz OpenAI", key="save_openai"):
                    config.set_api_key(openai_api_key)
                    st.success("✅ Zapisano!")
                    st.rerun()

            # Status połączenia OpenAI
            openai_status_key = "openai_connection_status"
            if openai_status_key not in st.session_state:
                st.session_state[openai_status_key] = {
                    "tested": False,
                    "success": False,
                    "message": "",
                }

            col_status, col_test = st.columns([2, 1])
            with col_test:
                if st.button("🔌 Test", key="test_openai", use_container_width=True):
                    with st.spinner("Testuję..."):
                        success, msg = test_openai_connection(openai_api_key, openai_model)
                        st.session_state[openai_status_key] = {
                            "tested": True,
                            "success": success,
                            "message": msg,
                        }
            with col_status:
                status = st.session_state[openai_status_key]
                if status["tested"]:
                    if status["success"]:
                        st.markdown(f"🟢 **{status['message']}**")
                    else:
                        st.markdown(f"🔴 **{status['message']}**")
                elif openai_api_key:
                    st.markdown("⚪ *Kliknij Test*")
                else:
                    st.markdown("⚪ *Brak klucza*")
        else:
            st.caption("OpenAI wyłączony")

        # --- Google AI Studio Section ---
        st.markdown("---")
        google_col1, google_col2 = st.columns([3, 1])
        with google_col1:
            st.markdown("**Google AI Studio**")
        with google_col2:
            google_enabled = st.toggle(
                "ON",
                value=config.get("google_enabled", True),
                key="google_toggle",
                help="Włącz/wyłącz Google AI Studio (Gemini)",
            )
            if google_enabled != config.get("google_enabled", True):
                config.set("google_enabled", google_enabled)

        if google_enabled:
            if not google_genai_available:
                st.warning(
                    "⚠️ Brak biblioteki google-generativeai. Zainstaluj: `pip install google-generativeai`"
                )

            saved_google_key = config.get_google_api_key()
            google_api_key = st.text_input(
                "Google AI Studio API Key",
                value=saved_google_key,
                type="password",
                help="Twój klucz Google AI Studio (Gemini)",
                key="google_ai_key_input",
            )
            google_model = config.get("google_model", "auto")
            google_models = ["auto"] + get_google_model_list(google_api_key)
            if google_model not in google_models:
                google_models.append(google_model)
            google_selected = st.selectbox(
                "Model Gemini",
                options=google_models,
                index=google_models.index(google_model),
                key="google_model_input",
                help="Wpisz nazwę modelu lub użyj 'auto' aby dobrać najnowszy dostępny.",
            )
            if google_selected != google_model:
                config.set("google_model", google_selected)
            if google_api_key != saved_google_key:
                if st.button("💾 Zapisz klucz Google", key="save_google"):
                    config.set_google_api_key(google_api_key)
                    st.success("✅ Zapisano!")
                    st.rerun()

            # Status połączenia Google
            google_status_key = "google_connection_status"
            if google_status_key not in st.session_state:
                st.session_state[google_status_key] = {
                    "tested": False,
                    "success": False,
                    "message": "",
                }

            col_status_g, col_test_g = st.columns([2, 1])
            with col_test_g:
                test_disabled = not google_genai_available
                if st.button(
                    "🔌 Test",
                    key="test_google",
                    use_container_width=True,
                    disabled=test_disabled,
                ):
                    with st.spinner("Testuję..."):
                        success, msg = test_google_connection(google_api_key, google_model)
                        st.session_state[google_status_key] = {
                            "tested": True,
                            "success": success,
                            "message": msg,
                        }
            with col_status_g:
                status = st.session_state[google_status_key]
                if not google_genai_available:
                    st.markdown("⚪ *Zainstaluj bibliotekę*")
                elif status["tested"]:
                    if status["success"]:
                        st.markdown(f"🟢 **{status['message']}**")
                    else:
                        st.markdown(f"🔴 **{status['message']}**")
                elif google_api_key:
                    st.markdown("⚪ *Kliknij Test*")
                else:
                    st.markdown("⚪ *Brak klucza*")
        else:
            st.caption("Google AI Studio wyłączony")

        # Podsumowanie aktywnego providera
        st.markdown("---")
        if provider_choice == "openai":
            api_key = config.get_api_key() if openai_enabled else ""
        else:
            api_key = config.get_google_api_key() if google_enabled else ""

        if api_key:
            st.success(f"✅ Aktywny: {llm_provider_labels[provider_choice]}")
        else:
            st.warning(f"⚠️ Brak klucza dla: {llm_provider_labels[provider_choice]}")

        st.divider()

        # === STATUS MODUŁÓW ===
        st.subheader("🧩 Status modułów")
        module_status = {
            "Topic Analyzer": topic_analyzer_available,
            "Advanced Analytics": advanced_available,
            "Competitor Tracker": competitor_tracker_available,
            "YouTube API": google_api_available,
            "Google AI Studio": google_genai_available,
        }
        for name, available in module_status.items():
            st.caption(f"{'✅' if available else '⚠️'} {name}")
        if not api_key:
            st.info(
                f"Tryb bez API ({llm_provider_labels[provider_choice]}): generowanie tytułów/obietnic działa z szablonów."
            )

        st.divider()

        # === YOUTUBE SYNC ===
        st.subheader("📺 YouTube Sync")

        yt_sync = get_youtube_sync()
        last_sync = yt_sync.get_last_sync_time()
        yt_api_key = config.get_youtube_api_key()
        yt_channel_id = config.get("channel_id", "")
        credentials_source = yt_sync.get_credentials_source()

        yt_api_key_input = st.text_input(
            "YouTube API Key (public)",
            value=yt_api_key,
            type="password",
            help="Klucz do publicznych zapytań YouTube Data API",
        )
        yt_channel_input = st.text_input(
            "Channel ID (public)",
            value=yt_channel_id,
            help="ID kanału do publicznego syncu (bez OAuth)",
        )
        if yt_api_key_input != yt_api_key:
            config.set_youtube_api_key(yt_api_key_input)
        if yt_channel_input != yt_channel_id:
            config.set("channel_id", yt_channel_input)
        yt_sync.set_api_key(yt_api_key_input)
        yt_sync.set_channel_id(yt_channel_input)

        if last_sync:
            st.caption(f"Ostatnia sync: {last_sync}")
        if credentials_source:
            st.caption(f"Credentials: {credentials_source}")

        if not google_api_available:
            st.warning("⚠️ Zainstaluj: `pip install google-api-python-client google-auth-oauthlib`")
        elif yt_sync.has_credentials():
            if st.button("🔄 Synchronizuj dane (OAuth)", use_container_width=True):
                with st.spinner("Loguję do YouTube..."):
                    success, msg = yt_sync.authenticate()
                    if success:
                        st.success(msg)
                        with st.spinner("Pobieram dane..."):
                            _, sync_msg = yt_sync.sync_all(include_analytics=True)
                            st.success(sync_msg)
                            st.rerun()
                    else:
                        st.error(msg)
        else:
            if yt_sync.ensure_public_client() and yt_channel_input:
                st.info("Tryb publiczny: tylko dane z YouTube Data API (bez Analytics).")
                if st.button("🔄 Synchronizuj dane (public)", use_container_width=True):
                    with st.spinner("Pobieram dane publiczne..."):
                        _, sync_msg = yt_sync.sync_all(
                            include_analytics=False, include_transcripts=False
                        )
                        st.success(sync_msg)
                        st.rerun()
            with st.expander("📖 Jak skonfigurować?"):
                st.markdown(yt_sync.setup_instructions())

        st.divider()

        # === DANE KANAŁU ===
        st.subheader("📊 Dane kanału")

        if merged_df is not None:
            st.success(f"✅ {len(merged_df)} filmów")

            required_cols = {"title", "views"}
            recommended_cols = {"retention", "label", "published_at"}
            present_cols = set(merged_df.columns)
            missing_required = sorted(required_cols - present_cols)
            missing_recommended = sorted(recommended_cols - present_cols)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Filmy", f"{len(merged_df)}")
            with col2:
                st.metric(
                    "Views",
                    "✅" if "views" in present_cols else "❌",
                    help=show_tooltip("channel_views"),
                )
            with col3:
                st.metric(
                    "Retention",
                    "✅" if "retention" in present_cols else "❌",
                    help=show_tooltip("channel_retention"),
                )
            with col4:
                st.metric(
                    "Label",
                    "✅" if "label" in present_cols else "❌",
                    help=show_tooltip("channel_label"),
                )

            st.caption("Wymagane: title, views")
            if missing_required:
                st.warning(f"Brakuje wymaganych danych: {', '.join(missing_required)}.")
            if missing_recommended:
                st.info(f"Brakuje rekomendowanych danych: {', '.join(missing_recommended)}.")
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

    return get_llm_settings()
