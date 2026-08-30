"""The credential loader must never leak, and must never override the shell."""
from __future__ import annotations

import os

from kernelforge import secrets


def test_env_var_wins_over_the_file(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("SOCLAAS_MODEL=from-file\nONLY_IN_FILE=yes\n", encoding="utf-8")
    monkeypatch.setenv("SOCLAAS_MODEL", "from-shell")
    secrets.load(str(env))
    assert os.environ["SOCLAAS_MODEL"] == "from-shell"
    assert os.environ["ONLY_IN_FILE"] == "yes"


def test_comments_blanks_and_quotes(tmp_path):
    env = tmp_path / ".env"
    env.write_text('# a comment\n\nA="quoted"\nB=plain\nC=\nnot_a_pair\n',
                   encoding="utf-8")
    found = secrets.load(str(env), override=True)
    assert found["A"] == "quoted" and found["B"] == "plain"
    assert "C" not in found and "not_a_pair" not in found


def test_missing_file_is_not_an_error(tmp_path):
    assert secrets.load(str(tmp_path / "nope.env")) == {}


def test_redaction_never_reveals_the_middle():
    # Assembled at runtime so this file does not itself contain a
    # secret-shaped literal for the scanner below to find.
    key = "clsk" + "_" + "QotPJEJC" + "_" + "jc3aKpkCp7XXfNHh"
    out = secrets.redact(key)
    assert "jc3aKpkCp7XXfNHh"[:8] not in out
    assert out.startswith("clsk_Qot") and out.endswith(key[-4:])
    assert secrets.redact(None) == "(unset)"
    assert secrets.redact("short") == "***"


def test_repository_has_no_committed_secret():
    """The deliverable is a public repo: nothing tracked may contain a key."""
    import pathlib
    import re
    root = pathlib.Path(__file__).resolve().parent.parent
    pattern = re.compile(r"clsk_[A-Za-z0-9]{6,}|sk-ant-[A-Za-z0-9]{6,}")
    offenders = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in {".py", ".md", ".json",
                                                     ".sbatch", ".txt", ".yml"}:
            continue
        if ".env" in path.name or "results" in path.parts:
            continue
        if path.name == "test_secrets.py":
            continue          # this file builds a fake key to test redaction
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if pattern.search(text):
            offenders.append(str(path.relative_to(root)))
    assert not offenders, f"secret-shaped strings in tracked files: {offenders}"
