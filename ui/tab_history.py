"""History tab rendering."""

import streamlit as st


def render_history_tab(history, vault) -> None:
    """Render the history tab."""
    st.header("📜 Historia ocen")

    all_history = history.get_all()

    if not all_history:
        st.info("Brak ocen w historii. Oceń pierwszy pomysł!")
        return

    search_query = st.text_input("🔍 Szukaj w historii", key="history_search")

    filtered = all_history
    if search_query:
        filtered = history.search(search_query)

    st.caption(f"Pokazuję {len(filtered)} z {len(all_history)} ocen")

    for entry in filtered[:50]:
        verdict = entry.get("final_verdict", "BORDER")
        score = entry.get("final_score_with_bonus", entry.get("final_score", 0))

        color = "#2d5a3d" if verdict == "PASS" else "#4a4000" if verdict == "BORDER" else "#4a1a1a"
        emoji = "🟢" if verdict == "PASS" else "🟡" if verdict == "BORDER" else "🔴"

        with st.expander(
            f"{emoji} {entry.get('title', 'Bez tytułu')[:60]}... | {score:.0f}/100 | {entry.get('timestamp', '')[:10]}"
        ):
            a1, a2, _ = st.columns([1, 1, 2])
            with a1:
                if st.button("🗑️ Usuń", key=f"hist_del_{entry.get('id','')}"):
                    history.delete(entry.get("id", ""))
                    st.rerun()
            with a2:
                if st.button("💾 Do Vault", key=f"hist_to_vault_{entry.get('id','')}"):
                    payload = entry.get("payload") or entry
                    vault.add(
                        title=entry.get("title", ""),
                        promise=entry.get("promise", ""),
                        score=int(entry.get("final_score_with_bonus", entry.get("final_score", 0)) or 0),
                        reason="Zapis z historii",
                        tags=["history_import"],
                        topic=(payload.get("topic") if isinstance(payload, dict) else ""),
                        payload=payload,
                        status="new",
                    )
                    st.success("✅ Zapisano do Vault.")

            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"**Tytuł:** {entry.get('title', '')}")
                st.markdown(f"**Obietnica:** {entry.get('promise', '') or 'Brak'}")
                if entry.get("status"):
                    st.markdown(f"**Status:** {entry.get('status')}")
                if entry.get("tags"):
                    st.markdown(f"**Tagi:** {', '.join(entry.get('tags', []))}")
                if entry.get("why"):
                    st.markdown(f"**Diagnoza:** {entry.get('why', '')}")

                with st.expander("📦 Pełny wynik (payload)", expanded=False):
                    st.json(entry.get("payload") or {})

                improvements = entry.get("improvements") or []
                if improvements:
                    st.markdown("**Ulepszenia:**")
                    for imp in improvements[:3]:
                        st.markdown(f"- {imp}")

            with col2:
                st.metric("Score", f"{score:.0f}/100")
                st.metric("Data", f"{entry.get('data_score', 0):.0f}")
                st.metric("LLM", f"{entry.get('llm_score', 0):.0f}")

                st.divider()
                if entry.get("published"):
                    st.success(f"✅ Published: {entry.get('actual_views', 0):,} views")
                    if entry.get("prediction_accuracy"):
                        st.metric("Accuracy", f"{entry['prediction_accuracy']}%")
                else:
                    st.markdown("**📊 Dodaj tracking:**")
                    actual_views = st.number_input(
                        "Rzeczywiste views",
                        min_value=0,
                        key=f"track_views_{entry['id']}",
                    )
                    if st.button("💾 Zapisz", key=f"track_btn_{entry['id']}"):
                        history.update_tracking(entry["id"], actual_views)
                        st.success("✅ Zapisano!")
                        st.rerun()

    st.divider()
    col_export1, col_export2 = st.columns(2)
    with col_export1:
        if st.button("📥 Eksportuj do CSV"):
            csv = history.export_to_csv()
            st.download_button(
                "⬇️ Pobierz CSV",
                csv,
                "evaluation_history.csv",
                "text/csv",
            )
    with col_export2:
        if st.button("📥 Eksportuj do JSON"):
            json_payload = history.export_to_json()
            st.download_button(
                "⬇️ Pobierz JSON",
                json_payload,
                "evaluation_history.json",
                "application/json",
            )
