"""Hidden, temporary credential entry for an explicitly requested Gemini action."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import getpass
import os
import re
import sys
import unicodedata
import warnings


class CredentialError(RuntimeError):
    """A safe credential-entry error that never contains the supplied value."""


_KEY_VARIABLE = "GEMINI_API_KEY"
_QUOTE_CHARACTERS = {"'", '"', "`", "\u2018", "\u2019", "\u201c", "\u201d"}
_INVALID_KEY_MESSAGE = (
    "Enter only the API key, without quotes, whitespace, control characters, "
    "or an environment-variable assignment."
)


def _validate_key(value: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or any(character.isspace() for character in value)
        or any(unicodedata.category(character) in ("Cc", "Cf", "Cs") for character in value)
        or value[0] in _QUOTE_CHARACTERS
        or value[-1] in _QUOTE_CHARACTERS
        or re.match(r"[A-Za-z_][A-Za-z0-9_]*=", value) is not None
    ):
        raise CredentialError(_INVALID_KEY_MESSAGE)


@contextmanager
def prompted_gemini_key() -> Iterator[None]:
    """Set a hidden key for this context, restoring the prior environment.

    The context yields no credential value. Noninteractive input and getpass's
    potentially echoed fallback are refused before accepting a key. This does
    not load or save files, log a credential, or contact any service.
    """
    try:
        if sys.stdin is None or not sys.stdin.isatty():
            raise CredentialError("Hidden key entry requires an interactive terminal. Run this command in a terminal.")
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            value = getpass.getpass("Gemini API key (hidden; this run only): ")
    except (EOFError, getpass.GetPassWarning, OSError, ValueError):
        raise CredentialError(
            "The key could not be read with hidden input. Use an interactive terminal and try again."
        ) from None
    _validate_key(value)

    previous = os.environ.get(_KEY_VARIABLE)
    try:
        os.environ[_KEY_VARIABLE] = value
        yield None
    finally:
        if previous is None:
            os.environ.pop(_KEY_VARIABLE, None)
        else:
            os.environ[_KEY_VARIABLE] = previous
