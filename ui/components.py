"""Shared UI components for rendering results."""

from typing import Dict

import streamlit as st

from config_manager import generate_report
from ui.tooltips import TOOLTIPS


def copy_to_clipboard_button(text: str, button_text: str = "📋 Kopiuj") -> None:
    """Przycisk do kopiowania do schowka."""
    st.code(text, language=None)
    st.caption("↑ Zaznacz i skopiuj (Ctrl+C)")


def render_verdict_card(result: Dict) -> None:
    """Renderuje główną kartę z werdyktem."""
    score = result.get("final_score_with_bonus", result.get("final_score", 0))
    verdict = result.get("final_verdict", "BORDER")
    advanced_bonus = result.get("advanced_bonus", 0)

    verdict_class = f"verdict-{verdict.lower()}"
    emoji = {"PASS": "🟢", "BORDER": "🟡", "FAIL": "🔴"}.get(verdict, "🟡")

    data_score = result.get("data_score", 0)
    llm_score = result.get("llm_score", 0)
    risk_penalty = result.get("risk_penalty", 0)

    st.markdown(
        f"""
    <div class="{verdict_class}" style="padding: 2rem; border-radius: 12px; text-align: center; margin: 1rem 0;">
        <div style="font-size: 3rem; font-weight: bold;">
            {emoji} {verdict}
        </div>
        <div style="font-size: 2.5rem; margin-top: 0.5rem;">
            {score:.0f}/100
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Breakdown
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Data (ML)", f"{data_score:.0f}", help=TOOLTIPS["data_score"])
    with col2:
        st.metric("LLM", f"{llm_score:.0f}", help=TOOLTIPS["llm_score"])
    with col3:
        st.metric("Kara", f"-{risk_penalty:.0f}", help=TOOLTIPS["risk_penalty"])
    with col4:
        st.metric(
            "Bonus",
            f"+{advanced_bonus:.0f}" if advanced_bonus >= 0 else f"{advanced_bonus:.0f}",
            help="Suma bonusów z trendów, konkurencji i DNA match",
        )


def render_diagnosis(result: Dict) -> None:
    """Renderuje diagnozę i ryzyka."""
    why = result.get("why", "Brak diagnozy")

    st.markdown(
        f"""
    <div class="info-box">
        <strong>💬 DIAGNOZA:</strong><br>
        {why}
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Risk flags z wyjaśnieniami
    risk_flags = result.get("risk_flags", [])
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
            st.markdown(
                f"""
            <span style="background: rgba(255,107,107,0.2); padding: 4px 12px; border-radius: 15px; margin: 2px; display: inline-block;">
                ⚠️ {flag}
                {f'<span style="opacity:0.7; font-size:0.9em;"> - {explanation}</span>' if explanation else ''}
            </span>
            """,
                unsafe_allow_html=True,
            )


def render_dimensions(result: Dict) -> None:
    """Renderuje wymiary oceny jako progress bary."""
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

        st.markdown(
            f"""
        <div style="margin: 8px 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                <span>{name}</span>
                <span style="font-weight: bold;">{value}/100</span>
            </div>
            <div class="dim-bar">
                <div class="dim-bar-fill" style="width: {value}%; background: {color};"></div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )


def render_improvements(result: Dict) -> None:
    """Renderuje listę ulepszeń."""
    improvements = result.get("improvements", [])
    if not improvements:
        return

    st.subheader("✨ Co poprawić")

    for i, imp in enumerate(improvements[:5], 1):
        priority = "🔴" if i == 1 else "🟡" if i == 2 else "⚪"
        st.markdown(f"{priority} **{i}.** {imp}")


def render_variants_with_scores(result: Dict, evaluator=None, analytics=None) -> None:
    """Renderuje warianty tytułów z ocenami."""
    title_variants = result.get("title_variants", [])
    promise_variants = result.get("promise_variants", [])

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📝 Warianty tytułu:**")
        for i, var in enumerate(title_variants[:6], 1):
            score = ""
            if analytics and hasattr(analytics, "ab_tester") and analytics.ab_tester:
                try:
                    calculated = analytics.ab_tester._calculate_ctr_score(var)
                    score = f" `{calculated:.0f}`"
                except (TypeError, ValueError, AttributeError):
                    pass
            st.markdown(f"{i}. {var}{score}")

    with col2:
        st.markdown("**💬 Warianty obietnicy:**")
        for i, var in enumerate(promise_variants[:6], 1):
            st.markdown(f"{i}. {var}")


def render_advanced_insights(result: Dict) -> None:
    """Renderuje insights z zaawansowanych analiz."""
    insights = result.get("advanced_insights", {})
    if not insights:
        return

    st.subheader("🧬 Zaawansowane analizy")

    cols = st.columns(3)

    with cols[0]:
        if "trends" in insights:
            trends = insights["trends"]
            overall = trends.get("overall", {})
            bonus = overall.get("trend_bonus", 0)
            msg = overall.get("message", "")[:50]

            color = "#2d5a3d" if bonus > 0 else "#5a2a2a" if bonus < 0 else "#3a3a3a"

            st.markdown(
                f"""
            <div style="background: {color}; padding: 1rem; border-radius: 8px; text-align: center;">
                <div style="font-size: 0.9rem; opacity: 0.8;">📈 Trend</div>
                <div style="font-size: 1.8rem; font-weight: bold;">{bonus:+d}</div>
                <div style="font-size: 0.8rem; opacity: 0.7;">{msg}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            with st.expander("ℹ️ Co to znaczy?"):
                st.markdown(TOOLTIPS["trend_bonus"])

    with cols[1]:
        if "competition" in insights:
            comp = insights["competition"].get("analysis", {})
            bonus = comp.get("score_bonus", 0)
            saturation = comp.get("saturation", "UNKNOWN")
            emoji = comp.get("emoji", "❓")

            color = "#2d5a3d" if bonus > 0 else "#5a2a2a" if bonus < 0 else "#3a3a3a"

            st.markdown(
                f"""
            <div style="background: {color}; padding: 1rem; border-radius: 8px; text-align: center;">
                <div style="font-size: 0.9rem; opacity: 0.8;">🔍 Konkurencja</div>
                <div style="font-size: 1.8rem; font-weight: bold;">{emoji} {bonus:+d}</div>
                <div style="font-size: 0.8rem; opacity: 0.7;">{saturation}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            with st.expander("ℹ️ Co to znaczy?"):
                st.markdown(TOOLTIPS["competition_bonus"])

    with cols[2]:
        if "dna_match" in insights:
            dna = insights["dna_match"]
            bonus = dna.get("bonus", 0)
            matches = dna.get("matches", [])

            color = "#2d5a3d" if bonus > 0 else "#3a3a3a"

            st.markdown(
                f"""
            <div style="background: {color}; padding: 1rem; border-radius: 8px; text-align: center;">
                <div style="font-size: 0.9rem; opacity: 0.8;">🧬 DNA Match</div>
                <div style="font-size: 1.8rem; font-weight: bold;">+{bonus}</div>
                <div style="font-size: 0.8rem; opacity: 0.7;">{len(matches)} dopasowań</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            with st.expander("ℹ️ Co to znaczy?"):
                st.markdown(TOOLTIPS["dna_bonus"])
                if matches:
                    for match in matches:
                        st.markdown(f"- {match}")


def render_copy_report(result: Dict) -> None:
    """Renderuje przycisk do kopiowania raportu."""
    report = generate_report(result)

    with st.expander("📋 Kopiuj pełny raport"):
        st.text_area("Raport do skopiowania:", report, height=400)
        st.caption("Zaznacz wszystko (Ctrl+A) i skopiuj (Ctrl+C)")
