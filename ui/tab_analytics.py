"""Analytics tab rendering."""

from datetime import timedelta
from pathlib import Path
from typing import Callable

import pandas as pd
import plotly.express as px
import streamlit as st


def render_analytics_tab(
    *,
    merged_df: pd.DataFrame | None,
    history,
    advanced_available: bool,
    get_advanced_analytics: Callable,
    llm_provider: str,
    llm_model: str,
    api_key: str,
    channel_data_dir: Path,
    merged_data_file: Path,
) -> None:
    """Render the analytics tab."""
    st.header("📊 Dashboard kanału")

    if merged_df is None:
        st.warning("⚠️ Załaduj dane kanału aby zobaczyć dashboard")
        return

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
        df_pred["date"] = pd.to_datetime(df_pred[date_col], errors="coerce", utc=True)
        df_pred = df_pred.dropna(subset=["date"]).sort_values("date")

    if "views" not in df_pred.columns or df_pred.empty:
        st.info("Brak wystarczających danych (views/published_at) do prognozy.")
    else:
        planned_uploads = st.slider(
            "Planowane publikacje / miesiąc", 1, 12, 4, key="pred_uploads"
        )

        recent_df = df_pred.copy()
        if date_col:
            cutoff = pd.Timestamp.now(tz="UTC") - timedelta(days=30)
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

        st.caption(
            "Prognoza bazuje na średniej z ostatnich publikacji i zakładanej liczbie filmów w miesiącu."
        )

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("📈 Views over time")

        if "published_at" in merged_df.columns or "publishedAt" in merged_df.columns:
            date_col = "published_at" if "published_at" in merged_df.columns else "publishedAt"
            df_chart = merged_df.copy()
            df_chart["date"] = pd.to_datetime(df_chart[date_col], errors="coerce", utc=True)
            df_chart = df_chart.dropna(subset=["date"]).sort_values("date")

            if len(df_chart) > 0 and "views" in df_chart.columns:
                fig = px.line(df_chart, x="date", y="views", title="Views per video")
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

    with chart_col2:
        st.subheader("🎯 Hit Rate")

        if "label" in merged_df.columns:
            label_counts = merged_df["label"].value_counts()

            fig = px.pie(
                values=label_counts.values,
                names=label_counts.index,
                title="PASS / BORDER / FAIL",
                color_discrete_map={
                    "PASS": "#28a745",
                    "BORDER": "#ffc107",
                    "FAIL": "#dc3545",
                },
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    if advanced_available:
        st.subheader("🧬 DNA Twojego kanału")

        if st.button("🔍 Analizuj DNA"):
            data_path = (
                str(channel_data_dir / "synced_channel_data.csv")
                if (channel_data_dir / "synced_channel_data.csv").exists()
                else str(merged_data_file)
            )
            analytics = get_advanced_analytics(data_path, llm_provider, llm_model, api_key)

            if analytics:
                dna = analytics.get_packaging_dna()

                if "error" not in dna:
                    if dna.get("recommendations"):
                        st.markdown("### 💡 Kluczowe wnioski")
                        for rec in dna["recommendations"]:
                            st.info(rec)

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("### ✅ Twoje trigger words")
                        triggers = dna.get("word_triggers", {}).get("trigger_words", [])
                        for trig in triggers[:10]:
                            st.markdown(f"- **{trig['word']}** (lift: {trig['lift']}x)")

                    with col2:
                        st.markdown("### ❌ Unikaj")
                        avoid = dna.get("word_triggers", {}).get("avoid_words", [])
                        for trig in avoid[:10]:
                            st.markdown(f"- {trig['word']} (lift: {trig['lift']}x)")

    st.divider()

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
