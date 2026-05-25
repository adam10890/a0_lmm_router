from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
PLUGIN_ROOT = REPO_ROOT / "usr" / "plugins" / "a0_lmm_router"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_host_helper_bind_defaults_to_loopback(monkeypatch):
    from usr.plugins.a0_lmm_router.tools import lmm_host_helper as helper

    monkeypatch.delenv("A0_LMM_HOST_BIND", raising=False)
    monkeypatch.delenv("A0_LMM_HOST_BIND_PUBLIC", raising=False)

    assert helper._resolve_bind_host("") == "127.0.0.1"


def test_host_helper_public_bind_requires_explicit_opt_in(monkeypatch):
    from usr.plugins.a0_lmm_router.tools import lmm_host_helper as helper

    monkeypatch.setenv("A0_LMM_HOST_BIND_PUBLIC", "1")

    assert helper._resolve_bind_host("0.0.0.0") == "0.0.0.0"


def test_host_helper_token_compare_uses_compare_digest(monkeypatch):
    from usr.plugins.a0_lmm_router.tools import lmm_host_helper as helper

    calls: list[tuple[str, str]] = []

    def fake_compare(left: str, right: str) -> bool:
        calls.append((left, right))
        return left == right

    monkeypatch.setattr(helper.hmac, "compare_digest", fake_compare)

    assert helper._token_matches("secret", "secret") is True
    assert calls == [("secret", "secret")]


def test_compose_path_outside_allowlist_is_rejected(tmp_path):
    from usr.plugins.a0_lmm_router.tools import lmm_host_helper as helper

    project_dir = tmp_path / "project"
    docker_dir = project_dir / "usr" / "plugins" / "a0_lmm_router" / "docker"
    docker_dir.mkdir(parents=True)
    default_compose = docker_dir / "docker-compose.lmm.yml"
    default_compose.write_text("services: {}\n", encoding="utf-8")
    outside = tmp_path / "outside.yml"
    outside.write_text("services: {}\n", encoding="utf-8")

    resolved = helper._resolve_compose_path(str(default_compose), str(default_compose), str(project_dir))
    assert resolved == str(default_compose.resolve())

    try:
        helper._resolve_compose_path(str(outside), str(default_compose), str(project_dir))
    except ValueError as exc:
        assert "not allowed" in str(exc)
    else:
        raise AssertionError("outside compose path was accepted")


def test_cors_allows_only_local_agent_zero_origins():
    from usr.plugins.a0_lmm_router.tools import lmm_host_helper as helper

    assert helper._allowed_cors_origin("http://127.0.0.1:5080") == "http://127.0.0.1:5080"
    assert helper._allowed_cors_origin("http://localhost:5080") == "http://localhost:5080"
    assert helper._allowed_cors_origin("http://evil.example:5080") == "null"


def test_rate_limit_blocks_after_window_limit(monkeypatch):
    from usr.plugins.a0_lmm_router.tools import lmm_host_helper as helper

    helper._rate_limit_hits.clear()
    monkeypatch.setattr(helper, "RATE_LIMIT_MAX", 2)
    monkeypatch.setattr(helper, "RATE_LIMIT_WINDOW_SECONDS", 60)

    assert helper._rate_limit_allow("127.0.0.1", now=100.0) is True
    assert helper._rate_limit_allow("127.0.0.1", now=101.0) is True
    assert helper._rate_limit_allow("127.0.0.1", now=102.0) is False
    assert helper._rate_limit_allow("127.0.0.1", now=161.0) is True
