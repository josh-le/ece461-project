from __future__ import annotations

from pathlib import Path
from typing import List
import types
import pytest

import ece461.url_file_parser as ufp
# ufp contains: ModelLinks, parse_url_file, and uses validators.url internally


def test_missing_file_exits(tmp_path: Path) -> None:
    """
    parse_url_file should exit(1) when the CSV path does not exist.
    """
    missing = tmp_path / "nope.csv"
    with pytest.raises(SystemExit) as exc:
        ufp.parse_url_file(str(missing))
    assert int(exc.value.code) == 1


def test_wrong_number_of_columns_raises_indexerror(tmp_path: Path) -> None:
    """
    A line that doesn't have exactly 3 comma-separated values raises IndexError.
    """
    p = tmp_path / "urls.csv"
    p.write_text("only_two,cols\n", encoding="utf-8")
    with pytest.raises(IndexError):
        ufp.parse_url_file(str(p))


def test_missing_model_exits(tmp_path: Path) -> None:
    """
    An empty model field triggers a logged error and exit(1).
    """
    p = tmp_path / "urls.csv"
    p.write_text("https://code,https://data,\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        ufp.parse_url_file(str(p))
    assert int(exc.value.code) == 1


def test_invalid_model_url_structure_indexerror(tmp_path: Path) -> None:
    """
    A non-URL 'model' field (no '.co/' segment) ultimately raises IndexError
    when splitting the model string.
    """
    p = tmp_path / "urls.csv"
    p.write_text("https://code,https://data,not_a_url\n", encoding="utf-8")
    with pytest.raises(IndexError):
        ufp.parse_url_file(str(p))


def test_valid_hf_url_extracts_model_id_and_calls_validators(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    A proper Hugging Face model URL should parse and extract 'org/name' and
    validators.url should be called for model, code, and dataset when present.
    """
    p = tmp_path / "urls.csv"
    # code,dataset,model
    p.write_text(
        "https://code.example/repo,https://data.example/ds,https://huggingface.co/myorg/mymodel/some/path\n",
        encoding="utf-8",
    )

    calls: list[str] = []

    def fake_url(s: str) -> bool:
        calls.append(s)
        return True

    monkeypatch.setattr(ufp.validators, "url", fake_url, raising=True)

    links = ufp.parse_url_file(str(p))
    assert len(links) == 1

    ml: ufp.ModelLinks = links[0]
    assert ml.model == "https://huggingface.co/myorg/mymodel/some/path"
    assert ml.model_id == "myorg/mymodel"
    assert ml.code == "https://code.example/repo"
    assert ml.dataset == "https://data.example/ds"

    # validators.url called for model, code, dataset (order not guaranteed)
    assert "https://huggingface.co/myorg/mymodel/some/path" in calls
    assert "https://code.example/repo" in calls
    assert "https://data.example/ds" in calls
    assert len(calls) == 3


def test_optional_fields_blank_become_none(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    When code/dataset are blank, they should become None in ModelLinks.
    """
    p = tmp_path / "urls.csv"
    p.write_text(",,https://huggingface.co/org/name\n", encoding="utf-8")

    # ensure validators.url doesn't explode (only model is checked)
    monkeypatch.setattr(ufp.validators, "url", lambda s: True, raising=True)

    links = ufp.parse_url_file(str(p))
    assert len(links) == 1
    ml: ufp.ModelLinks = links[0]
    assert ml.model_id == "org/name"
    assert ml.code is None and ml.dataset is None


def test_multiple_valid_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Two valid rows should yield two ModelLinks with correct model_id extraction.
    """
    p = tmp_path / "urls.csv"
    p.write_text(
        "\n".join(
            [
                "https://c1,https://d1,https://huggingface.co/o1/n1",
                "https://c2,https://d2,https://huggingface.co/o2/n2/extra",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # make validators.url always accept
    monkeypatch.setattr(ufp.validators, "url", lambda s: True, raising=True)

    links = ufp.parse_url_file(str(p))
    assert [ml.model_id for ml in links] == ["o1/n1", "o2/n2"]
    assert len(links) == 2
