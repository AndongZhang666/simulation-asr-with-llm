import pytest

from asr_ec.data.normalization import ConservativeEnglishNormalizer, IdentityNormalizer


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("  I'VE  got  2 apples... ", "i've got 2 apples"),
        ("U.S.A.\tand Mr. Smith", "u s a and mr smith"),
        ("A\u2019T  word---spacing", "a't word spacing"),
        ("Numbers 10 and 42 stay numeric", "numbers 10 and 42 stay numeric"),
    ],
)
def test_conservative_english_normalization(source: str, expected: str) -> None:
    normalizer = ConservativeEnglishNormalizer()

    assert normalizer.normalize(source) == expected
    assert normalizer.normalizer_id == "conservative-english@v1"


def test_identity_normalizer_preserves_raw_text() -> None:
    source = " Raw, Text! "

    assert IdentityNormalizer().normalize(source) == source
