"""The LLM client's contract with an endpoint that is not ours.

A judge will point this at a different gateway, or at nothing at all, and the
difference between "retry patiently" and "give up immediately" has to be the
right way round in both directions:

  * a **rate limit** is temporary and worth waiting minutes for -- we lost three
    arms of a model comparison to a client that gave up after 14 seconds;
  * a **refused connection** is a typo in the base URL, and retrying it eight
    times makes the reader wait six minutes to be told what we knew at once.

Both bugs were real. These tests exist so neither comes back.

No network is touched: `urlopen` is replaced by a stub that raises whatever the
test wants, and `time.sleep` is stubbed out so backoff costs nothing.
"""
from __future__ import annotations

import socket
import urllib.error

import pytest

from kernelforge.agent import proposers


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("SOCLAAS_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("SOCLAAS_API_KEY", "test-key")
    monkeypatch.setenv("SOCLAAS_MODEL", "test-model")
    monkeypatch.setattr(proposers.time, "sleep", lambda *_: None)
    return proposers.OpenAICompatProposer()


def _install(monkeypatch, side_effect):
    """Replace urlopen with a counter + a scripted failure."""
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        raise side_effect(calls["n"])

    # `_post` does `import urllib.request` at call time, so patching the
    # module attribute is what it will resolve against.
    import urllib.request as _ur
    monkeypatch.setattr(_ur, "urlopen", fake_urlopen)
    return calls


def _http(code):
    return lambda n: urllib.error.HTTPError(
        "https://example.invalid/v1", code, "boom", {}, None)


def test_rate_limit_is_retried_many_times(client, monkeypatch):
    """429 is temporary. Losing a measurement to it is not acceptable."""
    calls = _install(monkeypatch, _http(429))
    with pytest.raises(urllib.error.HTTPError):
        client._post({"model": "m"})
    assert calls["n"] >= 6, (
        f"gave up after {calls['n']} attempts; a quota window outlasts that")


def test_server_errors_are_retried(client, monkeypatch):
    calls = _install(monkeypatch, _http(503))
    with pytest.raises(urllib.error.HTTPError):
        client._post({"model": "m"})
    assert calls["n"] >= 6


def test_auth_failure_is_not_retried(client, monkeypatch):
    """A bad key will still be bad in two minutes. Say so at once."""
    calls = _install(monkeypatch, _http(401))
    with pytest.raises(urllib.error.HTTPError):
        client._post({"model": "m"})
    assert calls["n"] == 1, f"retried a 401 {calls['n']} times"


def test_refused_connection_is_not_retried(client, monkeypatch):
    """The `SOCLAAS_BASE_URL` typo case: fail fast, not in six minutes."""
    calls = _install(
        monkeypatch,
        lambda n: urllib.error.URLError(ConnectionRefusedError("refused")))
    with pytest.raises(urllib.error.URLError):
        client._post({"model": "m"})
    assert calls["n"] == 1, f"retried a refused connection {calls['n']} times"


def test_timeouts_are_retried(client, monkeypatch):
    """A slow model is worth waiting for, unlike a refused one."""
    calls = _install(monkeypatch,
                     lambda n: urllib.error.URLError(socket.timeout("slow")))
    with pytest.raises(urllib.error.URLError):
        client._post({"model": "m"})
    assert calls["n"] >= 6


def test_a_recovering_endpoint_succeeds(client, monkeypatch):
    """Two 429s then a result: the retry has to actually return the payload."""
    calls = {"n": 0}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"model": "served-model", "choices": []}'

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise urllib.error.HTTPError("u", 429, "slow down", {}, None)
        return Resp()

    import json as _json
    import urllib.request as _ur
    monkeypatch.setattr(_ur, "urlopen", fake_urlopen)
    monkeypatch.setattr(proposers.json, "load", lambda f: _json.loads(f.read()))
    out = client._post({"model": "m"})
    assert out["model"] == "served-model"
    assert calls["n"] == 3


def test_served_model_is_recorded(client):
    """The gateway aliases ids; a comparison must know what actually answered."""
    client._note_served({"model": "qwen3.8:27b"})
    assert "qwen3.8:27b" in client.served_models


def test_heuristic_proposer_needs_no_credentials(monkeypatch):
    """The whole loop has to run with no LLM at all -- that is the fallback."""
    for var in ("SOCLAAS_API_KEY", "SOCLAAS_BASE_URL", "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    p = proposers.HeuristicProposer()
    assert p.name == "heuristic"


def test_llm_proposer_refuses_to_start_without_credentials(monkeypatch):
    """And it says which variables to set, rather than failing at request time."""
    from kernelforge import secrets as _secrets
    # The constructor calls secrets.load(), which would read a developer's real
    # .env off disk and hand us working credentials. Neutralize it so this test
    # measures the no-credentials path rather than the machine it runs on.
    monkeypatch.setattr(_secrets, "load", lambda *a, **k: {})
    for var in ("SOCLAAS_API_KEY", "SOCLAAS_BASE_URL", "OPENAI_API_KEY",
                "OPENAI_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(RuntimeError) as e:
        proposers.OpenAICompatProposer()
    assert "SOCLAAS_BASE_URL" in str(e.value)
