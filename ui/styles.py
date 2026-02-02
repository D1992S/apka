"""Style definitions for the Streamlit UI."""

import streamlit as st

CSS_STYLES = """
<style>
/* Dark mode friendly colors */
.stAlert > div {
    color: inherit !important;
}

/* Layout helpers */
.section-card {
    background: #161616;
    border: 1px solid #333;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
}
.section-card h4 {
    margin: 0 0 8px 0;
}

/* Cards */
.metric-card {
    background: var(--background-secondary, #1f1f1f);
    border: 1px solid var(--border-color, #3a3a3a);
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
    color: #FFFACD; /* Jaśniejszy żółty dla lepszego kontrastu */
}
.verdict-fail {
    background: linear-gradient(135deg, #4a1a1a 0%, #5a2a2a 100%);
    color: #FF6B6B;
}

/* Tooltips */
.tooltip-icon {
    cursor: help;
    opacity: 0.8;
    margin-left: 6px;
    font-size: 0.95em;
    display: inline-flex;
    align-items: center;
    transition: opacity 0.2s ease;
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
    background: rgba(100, 149, 237, 0.14);
    border-left: 4px solid #6495ED;
    padding: 1rem;
    border-radius: 0 8px 8px 0;
    margin: 1rem 0;
}

/* Warning boxes */
.warning-box {
    background: rgba(255, 193, 7, 0.14);
    border-left: 4px solid #FFC107;
    padding: 1rem;
    border-radius: 0 8px 8px 0;
    margin: 1rem 0;
}

/* Success boxes */
.success-box {
    background: rgba(40, 167, 69, 0.14);
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
"""


def inject_styles() -> None:
    """Inject CSS styles into the Streamlit app."""
    st.markdown(CSS_STYLES, unsafe_allow_html=True)
