"""Idea vault tab rendering."""

import json

import pandas as pd
import streamlit as st


def render_vault_tab(vault) -> None:
    """Render the idea vault tab."""
    st.header("💡 Idea Vault")
    st.markdown("Pomysły zapisane na później")

    reminders = vault.check_reminders()
    if reminders:
        st.warning(f"🔔 Masz {len(reminders)} przypomnienie(a)!")

        for reminder in reminders:
            st.markdown(
                f"""
            <div class="warning-box">
                <strong>{reminder['title']}</strong><br>
                {reminder['reminder_reason']}
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.divider()

    status_filter = st.selectbox(
        "Filtruj:",
        ["Wszystkie", "Nowe", "Shortlisted", "Scripted", "Użyte", "Odrzucone"],
        key="vault_filter",
    )

    status_map = {
        "Wszystkie": None,
        "Nowe": "new",
        "Shortlisted": "shortlisted",
        "Scripted": "scripted",
        "Użyte": "used",
        "Odrzucone": "discarded",
    }

    all_ideas = vault.get_all(status=status_map.get(status_filter))
    all_tags = sorted({t for idea in all_ideas for t in idea.get("tags", [])})
    selected_tags = st.multiselect(
        "Filtruj po tagach", options=all_tags, default=[], key="vault_tag_filter"
    )
    if selected_tags:
        all_ideas = [i for i in all_ideas if set(i.get("tags", [])) & set(selected_tags)]
    selected_ids: set[str] = set()

    if not all_ideas:
        st.info("Vault jest pusty. Zapisz pomysły podczas oceny!")
    else:
        st.markdown("### ⚙️ Akcje zbiorcze")
        bulk_status = st.selectbox(
            "Ustaw status dla zaznaczonych",
            ["new", "shortlisted", "scripted", "used", "discarded"],
            key="vault_bulk_status",
        )
        if st.button("✅ Zastosuj status do zaznaczonych", key="vault_bulk_apply"):
            for idea in all_ideas:
                if idea.get("id") in st.session_state.get("vault_selected_ids", []):
                    vault.update_metadata(idea["id"], status=bulk_status)
            st.success("Zaktualizowano zaznaczone wpisy.")
            st.rerun()
        if st.button("🗑️ Usuń zaznaczone", key="vault_bulk_delete"):
            for idea in all_ideas:
                if idea.get("id") in st.session_state.get("vault_selected_ids", []):
                    vault.remove(idea["id"])
            st.success("Usunięto zaznaczone wpisy.")
            st.rerun()

        for idea in all_ideas:
            status = idea.get("status", "new")
            if status == "waiting":
                status = "new"
            status_emoji = (
                "🆕"
                if status == "new"
                else "⭐"
                if status == "shortlisted"
                else "📝"
                if status == "scripted"
                else "✅"
                if status == "used"
                else "❌"
            )

            selected = st.checkbox("Zaznacz", key=f"vault_select_{idea['id']}")
            if selected:
                selected_ids.add(idea["id"])

            with st.expander(
                f"{status_emoji} {idea.get('title', '')[:50]}... | Score: {idea.get('score', 0)}"
            ):
                st.markdown(f"**Tytuł:** {idea.get('title', '')}")
                st.markdown(f"**Obietnica:** {idea.get('promise', '') or 'Brak'}")
                st.markdown(
                    f"**Powód zapisania:** {idea.get('reason', '') or 'Nie podano'}"
                )

                with st.expander("📦 Pełny wynik (payload)", expanded=False):
                    st.json(idea.get("payload") or {})
                st.markdown(f"**Tagi:** {', '.join(idea.get('tags', [])) or 'Brak'}")
                st.markdown(f"**Dodano:** {idea.get('added', '')[:10]}")

                col1, col2 = st.columns(2)

                with col1:
                    new_status = st.selectbox(
                        "Status",
                        ["new", "shortlisted", "scripted", "used", "discarded"],
                        index=[
                            "new",
                            "shortlisted",
                            "scripted",
                            "used",
                            "discarded",
                        ].index(
                            status
                            if status in ["new", "shortlisted", "scripted", "used", "discarded"]
                            else "new"
                        ),
                        key=f"vault_status_{idea['id']}",
                    )
                    new_tags = st.text_input(
                        "Tagi (oddziel przecinkami)",
                        value=", ".join(idea.get("tags", [])),
                        key=f"vault_tags_{idea['id']}",
                    )
                    new_notes = st.text_area(
                        "Notatki",
                        value=idea.get("notes", ""),
                        key=f"vault_notes_{idea['id']}",
                    )
                    if st.button("💾 Zapisz zmiany", key=f"vault_save_{idea['id']}"):
                        tags_list = [t.strip() for t in new_tags.split(",") if t.strip()]
                        vault.update_metadata(
                            idea["id"], tags=tags_list, status=new_status, notes=new_notes
                        )
                        st.success("✅ Zaktualizowano.")
                        st.rerun()

                with col2:
                    if st.button("🗑️ Usuń", key=f"vault_delete_{idea['id']}"):
                        vault.remove(idea["id"])
                        st.rerun()

    st.session_state["vault_selected_ids"] = list(selected_ids)

    st.divider()
    if all_ideas:
        vault_json = json.dumps(all_ideas, indent=2, ensure_ascii=False, default=str)
        st.download_button(
            "⬇️ Pobierz Vault (JSON)",
            vault_json,
            "idea_vault.json",
            "application/json",
        )
        try:
            vault_df = pd.DataFrame(all_ideas)
            st.download_button(
                "⬇️ Pobierz Vault (CSV)",
                vault_df.to_csv(index=False),
                "idea_vault.csv",
                "text/csv",
            )
        except Exception:
            pass
