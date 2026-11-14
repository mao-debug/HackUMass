"""Hidden credential entry and private local storage for explicit Gemini actions."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import getpass
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
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


def _read_hidden_key(*, persistent: bool = False) -> str:
    try:
        if sys.stdin is None or not sys.stdin.isatty():
            raise CredentialError("Hidden key entry requires an interactive terminal. Run this command in a terminal.")
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            prompt = (
                "Gemini API key (hidden; saved on this device): " if persistent
                else "Gemini API key (hidden; this run only): "
            )
            value = getpass.getpass(prompt)
    except (EOFError, getpass.GetPassWarning, OSError, ValueError):
        raise CredentialError(
            "The key could not be read with hidden input. Use an interactive terminal and try again."
        ) from None
    _validate_key(value)
    return value


@contextmanager
def _temporary_key(value: str) -> Iterator[None]:
    previous = os.environ.get(_KEY_VARIABLE)
    try:
        os.environ[_KEY_VARIABLE] = value
        yield None
    finally:
        if previous is None:
            os.environ.pop(_KEY_VARIABLE, None)
        else:
            os.environ[_KEY_VARIABLE] = previous


@contextmanager
def prompted_gemini_key() -> Iterator[None]:
    """Read a hidden key for this context and restore the prior environment.

    Noninteractive input and getpass's potentially echoed fallback are refused.
    This path never saves a credential or contacts any service.
    """
    value = _read_hidden_key()
    with _temporary_key(value):
        yield None


def gemini_key_path() -> Path:
    """Return the private local credential location without accessing it."""
    return Path.home() / ".config" / "vidi" / "gemini.key"


def _check_owner(info: os.stat_result) -> None:
    if os.name == "posix" and info.st_uid != os.getuid():
        raise CredentialError("The saved Gemini key location must be owned by the current user.")


def _check_directory(directory: Path, *, prepare: bool = False) -> None:
    info = directory.lstat()
    if not stat.S_ISDIR(info.st_mode):
        raise CredentialError("The private Gemini key directory must be a regular directory, not a symbolic link.")
    _check_owner(info)
    if prepare:
        directory.chmod(0o700)
    elif os.name == "posix" and info.st_mode & 0o077:
        raise CredentialError("The saved Gemini key directory must allow access only to its owner.")


def _check_file(info: os.stat_result, *, private: bool = True) -> None:
    if not stat.S_ISREG(info.st_mode):
        raise CredentialError("The saved Gemini key must be a regular file, not a symbolic link or device.")
    _check_owner(info)
    if private and os.name == "posix" and info.st_mode & 0o077:
        raise CredentialError("The saved Gemini key file must allow access only to its owner.")


def save_gemini_key() -> Path:
    """Prompt once and atomically save an owner-only key without changing env.

    Only the vidi directory is secured to 0700; existing home and .config
    permissions are not changed. Replacing an owned regular file creates a new
    0600 file. A failed write or cancelled prompt preserves any previous key.
    """
    value = _read_hidden_key(persistent=True)
    path = gemini_key_path()
    temporary = None
    descriptor = None
    try:
        path.parent.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.parent.mkdir(mode=0o700, exist_ok=True)
        _check_directory(path.parent, prepare=True)
        try:
            existing = path.lstat()
        except FileNotFoundError:
            existing = None
        if existing is not None:
            # Explicit saving may repair an owned file's loose permissions;
            # it never follows/replaces a symlink or another file owner's key.
            _check_file(existing, private=False)
        descriptor, name = tempfile.mkstemp(prefix=".gemini.key.", suffix=".tmp", dir=path.parent)
        temporary = Path(name)
        os.fchmod(descriptor, 0o600)
        stream = os.fdopen(descriptor, "w", encoding="utf-8")
        descriptor = None
        with stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        return path
    except (OSError, UnicodeError, ValueError):
        raise CredentialError("The Gemini key could not be saved. Check your private configuration directory and try again.") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _read_saved_key(path: Path) -> str | None:
    descriptor = None
    try:
        try:
            info = path.lstat()
        except FileNotFoundError:
            return None
        _check_directory(path.parent)
        _check_file(info)
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        descriptor = os.open(path, flags)
        # Verify the opened file too, avoiding a symlink/permission swap
        # between the pathname check and reading the credential bytes.
        _check_file(os.fstat(descriptor))
        stream = os.fdopen(descriptor, "r", encoding="utf-8")
        descriptor = None
        with stream:
            value = stream.read()
    except (OSError, UnicodeError, ValueError):
        raise CredentialError("The saved Gemini key could not be read. Run --save-key to replace it.") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)
    try:
        _validate_key(value)
    except CredentialError:
        raise CredentialError("The saved Gemini key is invalid. Run --save-key to replace it.") from None
    return value


@contextmanager
def saved_gemini_key() -> Iterator[None]:
    """Use a saved key temporarily, while giving a nonempty export precedence.

    A missing file leaves the environment untouched, preserving the detector's
    existing missing-key behavior. Unsafe or malformed saved files are refused.
    """
    if os.environ.get(_KEY_VARIABLE):
        yield None
        return
    value = _read_saved_key(gemini_key_path())
    if value is None:
        yield None
        return
    with _temporary_key(value):
        yield None
