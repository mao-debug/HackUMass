"""Hidden credential entry tests use dummy values and never call a service."""

import getpass
import os
from pathlib import Path
import stat
from types import SimpleNamespace
from unittest.mock import Mock
import warnings

import pytest

from vidi.core import credentials
from vidi.core.credentials import CredentialError, prompted_gemini_key, save_gemini_key, saved_gemini_key


FAKE_KEY = "dummy-key-for-this-test"


@pytest.fixture
def terminal(monkeypatch):
    prompt = Mock(return_value=FAKE_KEY)
    monkeypatch.setattr(credentials.sys, "stdin", SimpleNamespace(isatty=Mock(return_value=True)))
    monkeypatch.setattr(credentials.getpass, "getpass", prompt)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    return prompt


def test_context_construction_does_not_prompt(terminal):
    prompted_gemini_key()
    terminal.assert_not_called()


def test_absent_key_is_set_only_during_context_and_context_yields_none(terminal, capsys, caplog):
    with prompted_gemini_key() as value:
        assert value is None
        assert os.environ["GEMINI_API_KEY"] == FAKE_KEY
    assert "GEMINI_API_KEY" not in os.environ
    terminal.assert_called_once_with("Gemini API key (hidden; this run only): ")
    assert capsys.readouterr() == ("", "")
    assert caplog.text == ""


def test_existing_key_is_restored_exactly(terminal, monkeypatch):
    previous = "previous-dummy-value"
    monkeypatch.setenv("GEMINI_API_KEY", previous)
    with prompted_gemini_key():
        assert os.environ["GEMINI_API_KEY"] == FAKE_KEY
    assert os.environ["GEMINI_API_KEY"] == previous


@pytest.mark.parametrize("existing", [None, "previous-dummy-value"])
@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_environment_is_restored_when_action_fails_or_is_cancelled(terminal, monkeypatch, existing, error):
    if existing is not None:
        monkeypatch.setenv("GEMINI_API_KEY", existing)
    with pytest.raises(error):
        with prompted_gemini_key():
            assert os.environ["GEMINI_API_KEY"] == FAKE_KEY
            raise error("Action stopped")
    assert os.environ.get("GEMINI_API_KEY") == existing


def test_nested_contexts_restore_outer_then_original_value(terminal, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "original-dummy-value")
    terminal.side_effect = ["outer-dummy-value", "inner-dummy-value"]
    with prompted_gemini_key():
        assert os.environ["GEMINI_API_KEY"] == "outer-dummy-value"
        with prompted_gemini_key():
            assert os.environ["GEMINI_API_KEY"] == "inner-dummy-value"
        assert os.environ["GEMINI_API_KEY"] == "outer-dummy-value"
    assert os.environ["GEMINI_API_KEY"] == "original-dummy-value"


@pytest.mark.parametrize("stdin", [None, SimpleNamespace(isatty=lambda: False)])
def test_nonterminal_input_is_refused_before_getpass(terminal, monkeypatch, stdin):
    monkeypatch.setattr(credentials.sys, "stdin", stdin)
    with pytest.raises(CredentialError, match="interactive terminal"):
        with prompted_gemini_key():
            pytest.fail("Invalid context entered")
    terminal.assert_not_called()
    assert "GEMINI_API_KEY" not in os.environ


def test_terminal_detection_error_is_safe(terminal, monkeypatch):
    monkeypatch.setattr(credentials.sys, "stdin", SimpleNamespace(isatty=Mock(side_effect=OSError("PRIVATE"))))
    with pytest.raises(CredentialError) as caught:
        with prompted_gemini_key():
            pytest.fail("Invalid context entered")
    assert "PRIVATE" not in str(caught.value)
    terminal.assert_not_called()


@pytest.mark.parametrize("failure", [EOFError, OSError, getpass.GetPassWarning])
def test_prompt_failures_have_safe_actionable_messages_and_preserve_environment(terminal, monkeypatch, failure):
    monkeypatch.setenv("GEMINI_API_KEY", "previous-dummy-value")
    terminal.side_effect = failure(f"PRIVATE {FAKE_KEY}")
    with pytest.raises(CredentialError, match="hidden input") as caught:
        with prompted_gemini_key():
            pytest.fail("Invalid context entered")
    assert FAKE_KEY not in str(caught.value) and "PRIVATE" not in str(caught.value)
    assert os.environ["GEMINI_API_KEY"] == "previous-dummy-value"


def test_warning_becomes_error_before_getpass_can_fall_back_to_echoed_input(terminal, capsys):
    echoed_input = Mock(return_value=FAKE_KEY)

    def unsafe_fallback(prompt):
        warnings.warn("Potentially echoed input", getpass.GetPassWarning)
        return echoed_input()

    terminal.side_effect = unsafe_fallback
    with pytest.raises(CredentialError, match="hidden input"):
        with prompted_gemini_key():
            pytest.fail("Echoed fallback entered")
    echoed_input.assert_not_called()
    assert "GEMINI_API_KEY" not in os.environ
    assert capsys.readouterr() == ("", "")


def test_prompt_cancellation_propagates_without_altering_previous_key(terminal, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "previous-dummy-value")
    terminal.side_effect = KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        with prompted_gemini_key():
            pytest.fail("Cancelled prompt entered")
    assert os.environ["GEMINI_API_KEY"] == "previous-dummy-value"


@pytest.mark.parametrize("value", [
    "", " ", " leading", "trailing ", "inside space", "inside\ttab", "inside\nnewline",
    "unicode\u00a0space", "unicode\u2003space", "unicode\u2028space", "\ufeffhidden-bom",
    "zero\u200bwidth", "null\x00byte", "control\x1bcode", "delete\x7fcode",
    '"quoted-dummy"', "'quoted-dummy'", "`quoted-dummy`", "\u201cquoted-dummy\u201d",
    '"unclosed-dummy', "unclosed-dummy'", "GEMINI_API_KEY=dummy-value", "KEY=dummy-value",
    "export GEMINI_API_KEY=dummy-value", None,
])
def test_invalid_pastes_are_rejected_without_disclosing_value(terminal, monkeypatch, value, capsys, caplog):
    monkeypatch.setenv("GEMINI_API_KEY", "previous-dummy-value")
    terminal.return_value = value
    with pytest.raises(CredentialError) as caught:
        with prompted_gemini_key():
            pytest.fail("Invalid credential entered")
    assert str(caught.value) == (
        "Enter only the API key, without quotes, whitespace, control characters, "
        "or an environment-variable assignment."
    )
    assert os.environ["GEMINI_API_KEY"] == "previous-dummy-value"
    assert capsys.readouterr() == ("", "") and caplog.text == ""


@pytest.mark.parametrize("value", ["a", "future-format-42", "some:future/token+value", "9alternative-key"])
def test_valid_keys_do_not_require_a_provider_prefix_or_length(terminal, value):
    terminal.return_value = value
    with prompted_gemini_key():
        assert os.environ["GEMINI_API_KEY"] == value
    assert "GEMINI_API_KEY" not in os.environ


@pytest.fixture
def saved_path(terminal, monkeypatch, tmp_path):
    path = tmp_path / "key-home" / ".config" / "vidi" / "gemini.key"
    monkeypatch.setattr(credentials, "gemini_key_path", lambda: path)
    return path


def existing_saved_key(path, value="previous-saved-dummy"):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    path.write_text(value, encoding="utf-8")
    path.chmod(0o600)


def test_save_then_reload_is_private_and_does_not_change_home_or_config_permissions(saved_path, terminal, capsys, caplog):
    home = saved_path.parent.parent.parent
    home.mkdir(mode=0o755, parents=True)
    config = home / ".config"
    config.mkdir(mode=0o755)
    before = (stat.S_IMODE(home.stat().st_mode), stat.S_IMODE(config.stat().st_mode))

    assert save_gemini_key() == saved_path
    terminal.assert_called_once_with("Gemini API key (hidden; saved on this device): ")

    assert saved_path.read_text() == FAKE_KEY
    assert stat.S_IMODE(saved_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(saved_path.parent.stat().st_mode) == 0o700
    assert (stat.S_IMODE(home.stat().st_mode), stat.S_IMODE(config.stat().st_mode)) == before
    assert "GEMINI_API_KEY" not in os.environ
    with saved_gemini_key() as value:
        assert value is None and os.environ["GEMINI_API_KEY"] == FAKE_KEY
    assert "GEMINI_API_KEY" not in os.environ
    assert capsys.readouterr() == ("", "") and caplog.text == ""


def test_save_does_not_change_an_exported_key(saved_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "exported-dummy")
    save_gemini_key()
    assert os.environ["GEMINI_API_KEY"] == "exported-dummy"
    assert saved_path.read_text() == FAKE_KEY


def test_overwrite_replaces_old_owned_key_and_repairs_private_modes(saved_path):
    existing_saved_key(saved_path)
    saved_path.chmod(0o644)
    saved_path.parent.chmod(0o755)

    assert save_gemini_key() == saved_path

    assert saved_path.read_text() == FAKE_KEY
    assert stat.S_IMODE(saved_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(saved_path.parent.stat().st_mode) == 0o700
    assert not list(saved_path.parent.glob(".gemini.key.*.tmp"))


@pytest.mark.parametrize("cancelled", [False, True])
def test_invalid_or_cancelled_save_preserves_previous_key(terminal, saved_path, cancelled):
    existing_saved_key(saved_path)
    if cancelled:
        terminal.side_effect = KeyboardInterrupt
    else:
        terminal.return_value = "GEMINI_API_KEY=invalid-paste"
    with pytest.raises(KeyboardInterrupt if cancelled else CredentialError):
        save_gemini_key()
    assert saved_path.read_text() == "previous-saved-dummy"
    assert not list(saved_path.parent.glob(".gemini.key.*.tmp"))


def test_failed_atomic_replace_preserves_previous_key_and_cleans_temporary_file(saved_path, monkeypatch):
    existing_saved_key(saved_path)
    monkeypatch.setattr(credentials.os, "replace", Mock(side_effect=OSError(f"PRIVATE {FAKE_KEY}")))

    with pytest.raises(CredentialError, match="could not be saved") as caught:
        save_gemini_key()

    assert "PRIVATE" not in str(caught.value) and FAKE_KEY not in str(caught.value)
    assert saved_path.read_text() == "previous-saved-dummy"
    assert not list(saved_path.parent.glob(".gemini.key.*.tmp"))


def test_missing_file_leaves_environment_and_filesystem_unchanged(saved_path):
    with saved_gemini_key() as value:
        assert value is None and "GEMINI_API_KEY" not in os.environ
    assert not saved_path.parent.exists()


def test_nonempty_export_takes_precedence_without_inspecting_saved_file(terminal, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "exported-dummy")
    lookup = Mock(side_effect=AssertionError("Saved credential must not be inspected"))
    monkeypatch.setattr(credentials, "gemini_key_path", lookup)
    with saved_gemini_key():
        assert os.environ["GEMINI_API_KEY"] == "exported-dummy"
    assert os.environ["GEMINI_API_KEY"] == "exported-dummy"
    lookup.assert_not_called()
    terminal.assert_not_called()


@pytest.mark.parametrize("prior", [None, ""])
@pytest.mark.parametrize("outcome", [None, RuntimeError, KeyboardInterrupt])
def test_loaded_key_restores_absent_or_empty_export_on_every_exit(saved_path, monkeypatch, prior, outcome):
    existing_saved_key(saved_path)
    if prior is not None:
        monkeypatch.setenv("GEMINI_API_KEY", prior)

    def use_key():
        with saved_gemini_key():
            assert os.environ["GEMINI_API_KEY"] == "previous-saved-dummy"
            if outcome is not None:
                raise outcome("Action stopped")

    if outcome is None:
        use_key()
    else:
        with pytest.raises(outcome):
            use_key()
    assert os.environ.get("GEMINI_API_KEY") == prior


@pytest.mark.parametrize("content", [b"", b"GEMINI_API_KEY=private-invalid-paste", b"\xff\xfe"])
def test_malformed_saved_key_is_rejected_without_disclosing_contents(saved_path, content):
    existing_saved_key(saved_path)
    saved_path.write_bytes(content)
    with pytest.raises(CredentialError) as caught:
        with saved_gemini_key():
            pytest.fail("Malformed saved key entered")
    assert "private-invalid-paste" not in str(caught.value)
    assert "GEMINI_API_KEY" not in os.environ


def test_unreadable_saved_key_has_sanitized_error(saved_path, monkeypatch):
    existing_saved_key(saved_path)
    monkeypatch.setattr(credentials.os, "open", Mock(side_effect=PermissionError(f"PRIVATE {FAKE_KEY}")))
    with pytest.raises(CredentialError, match="could not be read") as caught:
        with saved_gemini_key():
            pytest.fail("Unreadable saved key entered")
    assert FAKE_KEY not in str(caught.value) and "PRIVATE" not in str(caught.value)
    assert "GEMINI_API_KEY" not in os.environ


def test_symlink_is_never_loaded_or_overwritten(saved_path, tmp_path):
    existing_saved_key(saved_path)
    target = tmp_path / "target.key"
    target.write_text("target-dummy-key")
    saved_path.unlink()
    saved_path.symlink_to(target)
    with pytest.raises(CredentialError):
        with saved_gemini_key():
            pytest.fail("Symlink loaded")
    with pytest.raises(CredentialError):
        save_gemini_key()
    assert target.read_text() == "target-dummy-key" and saved_path.is_symlink()


def test_nonregular_saved_key_is_rejected(saved_path):
    existing_saved_key(saved_path)
    saved_path.unlink()
    saved_path.mkdir()
    with pytest.raises(CredentialError, match="regular file"):
        with saved_gemini_key():
            pytest.fail("Directory loaded as a key")


@pytest.mark.skipif(os.name != "posix", reason="POSIX ownership and mode checks")
def test_group_or_other_permissions_on_file_or_private_directory_are_refused(saved_path):
    existing_saved_key(saved_path)
    saved_path.chmod(0o644)
    with pytest.raises(CredentialError, match="only to its owner"):
        with saved_gemini_key():
            pytest.fail("Public key file loaded")
    assert stat.S_IMODE(saved_path.stat().st_mode) == 0o644
    saved_path.chmod(0o600)
    saved_path.parent.chmod(0o755)
    with pytest.raises(CredentialError, match="only to its owner"):
        with saved_gemini_key():
            pytest.fail("Public key directory loaded")
    assert stat.S_IMODE(saved_path.parent.stat().st_mode) == 0o755


@pytest.mark.skipif(os.name != "posix", reason="POSIX ownership check")
def test_saved_key_owned_by_another_user_is_refused(saved_path, monkeypatch):
    existing_saved_key(saved_path)
    original_lstat = Path.lstat

    def other_owner(path, *args, **kwargs):
        info = original_lstat(path, *args, **kwargs)
        if path == saved_path:
            fields = list(info)
            fields[4] = info.st_uid + 1
            return os.stat_result(fields)
        return info

    monkeypatch.setattr(Path, "lstat", other_owner)
    with pytest.raises(CredentialError, match="owned by the current user"):
        with saved_gemini_key():
            pytest.fail("Another user's key loaded")
    assert "GEMINI_API_KEY" not in os.environ
