"""
Testy dla TitleGenerator - ocenianie tytułów
"""

import pytest
import sys
from pathlib import Path

# Dodaj główny katalog do ścieżki
sys.path.insert(0, str(Path(__file__).parent.parent))

from topic_analyzer import TitleGenerator


class TestTitleScoring:
    """Testy dla funkcji _score_title()"""

    @pytest.fixture
    def generator(self):
        """Tworzy instancję TitleGenerator bez klienta AI"""
        return TitleGenerator(openai_client=None, channel_data=None)

    def test_ideal_length_gets_bonus(self, generator):
        """Tytuł o idealnej długości (~52 znaki) powinien dostać bonus"""
        # Idealny tytuł - 52 znaki
        ideal_title = "a" * 52
        score, _ = generator._score_title(ideal_title)

        # Bardzo krótki tytuł
        short_title = "abc"
        short_score, _ = generator._score_title(short_title)

        assert score > short_score, "Idealny tytuł powinien mieć wyższy wynik niż bardzo krótki"

    def test_power_words_give_bonus(self, generator):
        """Słowa mocy powinny dawać bonus"""
        title_with_power = "Tragedia która wstrząsnęła światem"
        title_without_power = "Wydarzenie które miało miejsce"

        score_with, _ = generator._score_title(title_with_power)
        score_without, _ = generator._score_title(title_without_power)

        assert score_with > score_without, "Słowa mocy powinny dawać bonus"

    def test_clickbait_szok_gives_penalty(self, generator):
        """Słowo 'szok' powinno dawać karę"""
        title_with_szok = "SZOKUJĄCA prawda o tym wydarzeniu"
        title_without = "Nieznana prawda o tym wydarzeniu"

        score_with, _ = generator._score_title(title_with_szok)
        score_without, _ = generator._score_title(title_without)

        assert score_with < score_without, "'Szok' powinno dawać karę"

    def test_school_pattern_gives_penalty(self, generator):
        """Szkolny szablon powinien dawać karę"""
        school_title = "Historia powstania świata"
        good_title = "Jak naprawdę powstał świat?"

        school_score, _ = generator._score_title(school_title)
        good_score, _ = generator._score_title(good_title)

        assert school_score < good_score, "Szkolny szablon powinien dawać karę"

    def test_number_with_context_gives_bonus(self, generator):
        """Liczba z kontekstem (np. '10 lat') powinna dawać bonus"""
        title_with_number = "10 lat ukrywania prawdy o katastrofie"
        title_without = "Ukrywanie prawdy o katastrofie"

        score_with, _ = generator._score_title(title_with_number)
        score_without, _ = generator._score_title(title_without)

        assert score_with > score_without, "Liczba z kontekstem powinna dawać bonus"

    def test_year_in_title_gives_bonus(self, generator):
        """Data (rok) w tytule powinna dawać bonus"""
        title_with_year = "Co wydarzyło się w 1986?"
        title_without = "Co wydarzyło się wtedy?"

        score_with, _ = generator._score_title(title_with_year)
        score_without, _ = generator._score_title(title_without)

        assert score_with > score_without, "Rok w tytule powinien dawać bonus"

    def test_excessive_caps_gives_penalty(self, generator):
        """Nadmiarowe CAPS powinny dawać karę"""
        caps_title = "CAŁA PRAWDA O TYM WYDARZENIU ZOSTAŁA UKRYTA"
        normal_title = "Cała prawda o tym wydarzeniu została ukryta"

        caps_score, _ = generator._score_title(caps_title)
        normal_score, _ = generator._score_title(normal_title)

        assert caps_score < normal_score, "Nadmiarowe CAPS powinny dawać karę"

    def test_short_caps_ok(self, generator):
        """Krótkie CAPS (np. UFO, CIA) są OK"""
        title = "Tajne dokumenty CIA o UFO"
        score, reasons = generator._score_title(title)

        # Sprawdź czy jest pozytywna wzmianka o CAPS
        assert any("CAPS" in r for r in reasons), "Krótkie CAPS powinny być akceptowane"


class TestTitleGeneration:
    """Testy dla generowania tytułów"""

    @pytest.fixture
    def generator(self):
        return TitleGenerator(openai_client=None, channel_data=None)

    def test_generate_from_templates(self, generator):
        """Powinno generować tytuły z szablonów"""
        titles = generator.generate("katastrofa lotnicza", n=5, use_ai=False)

        assert len(titles) > 0, "Powinno wygenerować co najmniej jeden tytuł"
        assert all("title" in t for t in titles), "Każdy wynik powinien mieć 'title'"
        assert all("score" in t for t in titles), "Każdy wynik powinien mieć 'score'"

    def test_titles_contain_topic(self, generator):
        """Wygenerowane tytuły powinny zawierać temat"""
        topic = "katastrofa"
        titles = generator.generate(topic, n=3, use_ai=False)

        # Co najmniej jeden tytuł powinien zawierać temat
        contains_topic = any(topic.lower() in t["title"].lower() for t in titles)
        assert contains_topic, "Co najmniej jeden tytuł powinien zawierać temat"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
