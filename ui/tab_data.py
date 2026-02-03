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
        loaded_frames = []
        skipped_required_files = []
        unreadable_files = []
        for file in uploaded_files:
            try:
                df = pd.read_csv(file)
            except EmptyDataError:
                st.error(f"Plik {file.name} jest pusty lub bez nagłówków.")
                unreadable_files.append(file.name)
                continue
            except Exception as exc:
                st.error(f"Nie udało się wczytać pliku {file.name}: {exc}")
                unreadable_files.append(file.name)
                continue
            loaded_frames.append(df)
            st.write(
                f"**{file.name}**: {len(df)} wierszy, kolumny: {', '.join(df.columns[:5])}"
            )
            issues = validate_channel_dataframe(df)
            if issues["missing_required"]:
                st.error(
                    "Brak wymaganych kolumn: "
                    f"{', '.join(issues['missing_required'])}. "
                    "CSV musi zawierać `title` i `views`."
                )
                loaded_frames.pop()
                skipped_required_files.append(file.name)
                continue
            if issues["missing_recommended"]:
                st.warning(
                    f"Brak rekomendowanych kolumn: {', '.join(issues['missing_recommended'])}"
                )
            for warning in issues["warnings"]:
                st.warning(warning)

        if st.button("💾 Zapisz i połącz dane"):
            if not loaded_frames:
                st.error("Nie wczytano żadnych poprawnych plików CSV do zapisu.")
                st.stop()
            if skipped_required_files or unreadable_files:
                skipped_details = []
                if skipped_required_files:
                    skipped_details.append(
                        "pominięto pliki bez wymaganych kolumn: "
                        + ", ".join(skipped_required_files)
                    )
                if unreadable_files:
                    skipped_details.append(
                        "pominięto pliki, których nie dało się odczytać: "
                        + ", ".join(unreadable_files)
                    )
                st.warning("Zapisano tylko poprawne pliki CSV — " + "; ".join(skipped_details))
            channel_data_dir.mkdir(exist_ok=True)

            if merged_df is not None and "title" in merged_df.columns:
                merged = merged_df.copy().drop_duplicates(subset=["title"], keep="last")
            else:
                merged = pd.DataFrame()

            for df in loaded_frames:
                if "title" not in df.columns:
                    continue
                incoming = df.drop_duplicates(subset=["title"], keep="last").set_index("title")
                if merged.empty:
                    merged = incoming.reset_index()
                    continue
                merged = merged.set_index("title")
                existing_label = merged["label"] if "label" in merged.columns else None
                merged.update(incoming)
                if existing_label is not None:
                    if "label" in merged.columns:
                        merged["label"] = merged["label"].combine_first(existing_label)
                    else:
                        merged["label"] = existing_label
                merged = merged.reset_index()

            if not merged.empty and "title" in merged.columns:
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

        st.subheader("🛠️ Szybkie poprawki danych")
        fix_col1, fix_col2 = st.columns(2)
        with fix_col1:
            if "label" not in merged_df.columns:
                if st.button("➕ Dodaj kolumnę label", use_container_width=True):
                    updated = merged_df.copy()
                    updated["label"] = "BORDER"
                    updated.to_csv(merged_data_file, index=False)
                    st.success("✅ Dodano kolumnę label.")
                    st.rerun()
        with fix_col2:
            numeric_cols = [col for col in ["views", "retention"] if col in merged_df.columns]
            if numeric_cols:
                if st.button("🧹 Wyczyść wartości views/retention", use_container_width=True):
                    updated = merged_df.copy()
                    for col in numeric_cols:
                        updated[col] = pd.to_numeric(updated[col], errors="coerce")
                    updated.to_csv(merged_data_file, index=False)
                    st.success("✅ Wyczyściłem wartości liczbowe.")
                    st.rerun()

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

        if "title" in merged_df.columns:
            st.subheader("🏷️ Etykiety PASS/BORDER/FAIL")
            if "label" not in merged_df.columns:
                st.info("Brak kolumny `label`. Możesz ją dodać ręcznie poniżej.")
            editable_df = merged_df.copy()
            if "label" not in editable_df.columns:
                editable_df["label"] = "BORDER"
            editable_df["label"] = (
                editable_df["label"].fillna("BORDER").astype(str).str.upper()
            )
            label_edit_cols = [c for c in ["title", "views", "retention", "label"] if c in editable_df.columns]
            edited_labels = st.data_editor(
                editable_df[label_edit_cols],
                column_config={
                    "label": st.column_config.SelectboxColumn(
                        "label",
                        help="PASS/BORDER/FAIL",
                        options=["PASS", "BORDER", "FAIL"],
                        required=True,
                    ),
                },
                disabled=[c for c in label_edit_cols if c != "label"],
                use_container_width=True,
            )
            if st.button("💾 Zapisz etykiety", type="primary"):
                updated = merged_df.drop(columns=["label"], errors="ignore").copy()
                label_updates = edited_labels[["title", "label"]].copy()
                label_updates["label"] = (
                    label_updates["label"].fillna("BORDER").astype(str).str.upper()
                )
                updated = updated.merge(label_updates, on="title", how="left")
                updated.to_csv(merged_data_file, index=False)
                st.success("✅ Zapisano etykiety.")
                st.rerun()

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
