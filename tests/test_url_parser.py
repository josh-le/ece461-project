from pathlib import Path
import pytest
from ece461 import url_file_parser as ufp

def test_validate_url_basic() -> None:
    # Accepts http/https and (currently) ftp per validators.url
    assert ufp.validate_url("https://example.com")
    assert ufp.validate_url("http://example.com")
    assert ufp.validate_url("ftp://example.com")

    # For malformed, DO NOT use `is False` — the function returns a falsy object, not the literal False
    assert not ufp.validate_url("not_a_url")   # this will pass even if it's a ValidationError object

def test_parse_url_file_happy_path(tmp_path: Path) -> None:
    csv = (
        "https://github.com/user/repo,https://data.repo/ds,https://huggingface.co/user/model\n"
        ",,https://huggingface.co/user/model2\n"
        "https://github.com/org/another,,https://huggingface.co/org/another-model\n"
    )
    f = tmp_path / "links.csv"
    f.write_text(csv, encoding="utf-8")

    links = ufp.parse_url_file(str(f))
    assert len(links) == 3

    r0 = links[0]
    assert r0.model == "https://huggingface.co/user/model"
    assert r0.dataset == "https://data.repo/ds"
    assert r0.code == "https://github.com/user/repo"

    r1 = links[1]
    assert r1.model == "https://huggingface.co/user/model2"
    assert r1.dataset is None
    assert r1.code is None

    r2 = links[2]
    assert r2.model == "https://huggingface.co/org/another-model"
    assert r2.dataset is None
    assert r2.code == "https://github.com/org/another"

def test_parse_url_file_requires_three_columns(tmp_path: Path) -> None:
    bad = "only_two_columns,still_two\n"
    f = tmp_path / "bad.csv"
    f.write_text(bad, encoding="utf-8")
    with pytest.raises(IndexError):
        ufp.parse_url_file(str(f))