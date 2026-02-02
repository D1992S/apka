"""Data management tab rendering."""

from pathlib import Path
from typing import Callable

import pandas as pd
from pandas.errors import EmptyDataError
import streamlit as st


def render_data_tab(
    *,
    config,
    merged_df: pd.DataFrame | None,
    validate_channel_dataframe: Callable[[pd.DataFrame], dict],
    load_manual_scripts: Callable[[Path], list],
    get_youtube_sync: Callable,
    channel_data_dir: Path,
    merged_data_file: Path,
    script_sync_dir: Path,
    google_api_available: bool,
) -> None:
    """Render the data management tab."""
    st.header("📁 Zarządzanie danymi")

    st.subheader("📺 YouTube Sync")

    yt_sync = get_youtube_sync()
    yt_sync.set_api_key(config.get_youtube_api_key())
    yt_sync.set_channel_id(config.get("channel_id", ""))

    if google_api_available and yt_sync.has_credentials():
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 Pełna synchronizacja", use_container_width=True):
                with st.spinner("Synchronizuję..."):
                    success, msg = yt_sync.authenticate()
                    if success:
                        _, sync_msg = yt_sync.sync_all(
                            include_analytics=True, include_transcripts=False
                        )
                        st.success(sync_msg)
                        st.rerun()
                    else:
                        st.error(msg)

        with col2:
            if st.button("📝 Sync z transkryptami", use_container_width=True):
                with st.spinner("Synchronizuję (z transkryptami - może potrwać)..."):
                    success, msg = yt_sync.authenticate()
                    if success:
                        _, sync_msg = yt_sync.sync_all(
                            include_analytics=True, include_transcripts=True
                        )
                        st.success(sync_msg)
                        st.rerun()
                    else:
                        st.error(msg)
    else:
        st.info("Skonfiguruj YouTube API aby używać sync")
        if yt_sync.ensure_public_client() and config.get("channel_id"):
            if st.button("🔄 Publiczny sync (bez Analytics)", key="public_sync_data"):
                with st.spinner("Pobieram dane publiczne..."):
                    _, sync_msg = yt_sync.sync_all(
                        include_analytics=False, include_transcripts=False
                    )
                    st.success(sync_msg)
                    st.rerun()
        with st.expander("📖 Instrukcje"):
            st.markdown(yt_sync.setup_instructions())

    st.subheader("📝 Ręczny sync skryptów")
    manual_sync_help = f"Wrzucaj gotowe skrypty do folderu: {script_sync_dir.resolve()}"
    if st.button(
        "📥 Wczytaj skrypty z folderu",
        help=manual_sync_help,
        use_container_width=True,
    ):
        script_sync_dir.mkdir(exist_ok=True)
        scripts = load_manual_scripts(script_sync_dir)
        if scripts:
            st.success(f"Zsynchronizowano {len(scripts)} skryptów z folderu.")
            with st.expander("📄 Podgląd nazw plików"):
                st.write([script["name"] for script in scripts])
        else:
            st.warning("Nie znaleziono żadnych skryptów w podanym folderze.")

    st.divider()

    st.subheader("📤 Ręczny upload CSV")
    st.caption("Wymagane kolumny: `title`, `views`.")

    uploaded_files = st.file_uploader(
        "Przeciągnij pliki CSV", type=["csv"], accept_multiple_files=True
    )

    if uploaded_files:
        validations = []
        loaded_frames = []
        for file in uploaded_files:
            try:
                df = pd.read_csv(file)
            except EmptyDataError:
                st.error(f"Plik {file.name} jest pusty lub bez nagłówków.")
                validations.append(
                    {
                        "missing_required": ["title", "views"],
                        "missing_recommended": [],
                        "warnings": ["Plik CSV jest pusty lub nie zawiera nagłówków."],
                    }
                )
                continue
            except Exception as exc:
                st.error(f"Nie udało się wczytać pliku {file.name}: {exc}")
                validations.append(
                    {
                        "missing_required": ["title", "views"],
                        "missing_recommended": [],
                        "warnings": [],
                    }
                )
                continue
            loaded_frames.append(df)
            st.write(
                f"**{file.name}**: {len(df)} wierszy, kolumny: {', '.join(df.columns[:5])}"
            )
            issues = validate_channel_dataframe(df)
            validations.append(issues)
            if issues["missing_required"]:
                st.error(
                    "Brak wymaganych kolumn: "
                    f"{', '.join(issues['missing_required'])}. "
                    "CSV musi zawierać `title` i `views`."
                )
            if issues["missing_recommended"]:
                st.warning(
                    f"Brak rekomendowanych kolumn: {', '.join(issues['missing_recommended'])}"
                )
            for warning in issues["warnings"]:
                st.warning(warning)

        if st.button("💾 Zapisz i połącz dane"):
            if any(v["missing_required"] for v in validations):
                st.error(
                    "Nie można zapisać danych: brakuje wymaganych kolumn (title, views)."
                )
                st.stop()
            if not loaded_frames:
                st.error("Nie wczytano żadnych poprawnych plików CSV do zapisu.")
                st.stop()
            channel_data_dir.mkdir(exist_ok=True)

            merged = pd.concat(loaded_frames, ignore_index=True)

            if "title" in merged.columns:
                merged = merged.drop_duplicates(subset=["title"], keep="last")

            merged.to_csv(merged_data_file, index=False)
            st.success(f"✅ Zapisano {len(merged)} wierszy do {merged_data_file}")
            st.rerun()

    st.divider()

    st.subheader("📊 Aktualne dane")

    if merged_df is not None:
        issues = validate_channel_dataframe(merged_df)
        if issues["missing_required"]:
            st.error(
                "Brak wymaganych kolumn: "
                f"{', '.join(issues['missing_required'])}. "
                "CSV musi zawierać `title` i `views`."
            )
        if issues["missing_recommended"]:
            st.warning(
                f"Brak rekomendowanych kolumn: {', '.join(issues['missing_recommended'])}"
            )
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

        display_cols = [c for c in ["title", "views", "retention", "label"] if c in merged_df.columns]
        if display_cols:
            st.dataframe(merged_df[display_cols].head(20), use_container_width=True)

        if st.button("🗑️ Usuń wszystkie dane", type="secondary"):
            if merged_data_file.exists():
                merged_data_file.unlink()
            synced = channel_data_dir / "synced_channel_data.csv"
            if synced.exists():
                synced.unlink()
            st.success("Dane usunięte")
            st.rerun()
    else:
        st.info("Brak danych. Użyj YouTube Sync lub wgraj CSV.")
