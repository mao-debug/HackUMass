"""Hidden credential entry tests use dummy values and never call a service."""

import getpass
import os
from types import SimpleNamespace
from unittest.mock import Mock
import warnings

import pytest

from vidi.core import credentials
from vidi.core.credentials import CredentialError, prompted_gemini_key


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
