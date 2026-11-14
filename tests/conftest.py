"""Keep saved credentials isolated from the developer's real home directory."""

import pytest

from vidi.core import credentials


@pytest.fixture(autouse=True)
def isolated_saved_gemini_key(monkeypatch, tmp_path):
    """No test may implicitly read or overwrite a user's remembered API key."""
    monkeypatch.setattr(
        credentials, "gemini_key_path",
        lambda: tmp_path / "private-home" / ".config" / "vidi" / "gemini.key",
    )
