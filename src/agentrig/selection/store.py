"""Explicit private-file storage for runtime preferences."""

from __future__ import annotations

import json
import os
import stat
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from agentrig.selection._data import loads_record, string
from agentrig.selection.models import PreferredModelSet


@dataclass(frozen=True, slots=True, kw_only=True)
class JsonPreferenceStore:
    """One application preference per explicit file; absent files mean no preference.

    The caller owns the parent directory. Writes use a private same-directory file
    and atomic replacement. Malformed, stale-schema, and cross-application records
    fail closed. This adapter neither finds global defaults nor activates models.
    """

    path: Path = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))

    def read(self, application_id: str) -> PreferredModelSet | None:
        string(application_id)
        try:
            if self.path.is_symlink() or self.path.parent.is_symlink():
                raise ValueError("preference file and parent must not be symbolic links")
            # NOFOLLOW closes the final-component check/open race on POSIX.
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
            descriptor = os.open(self.path, flags)
            with os.fdopen(descriptor, "rb") as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raise ValueError("preference path must identify a regular file")
                payload = stream.read(1_048_577)
            if len(payload) > 1_048_576:
                raise ValueError("preference file exceeds the size limit")
            raw = payload.decode("utf-8")
        except FileNotFoundError:
            return None
        except (OSError, UnicodeError):
            raise ValueError("preference file cannot be read") from None
        try:
            result = PreferredModelSet.from_data(loads_record(raw))
        except (ValueError, TypeError):
            raise ValueError("preference file is invalid") from None
        if result.application_id != application_id:
            raise ValueError("preference file belongs to a different application")
        return result

    def write(self, preference: PreferredModelSet) -> None:
        if not isinstance(preference, PreferredModelSet):
            raise TypeError("preference store requires PreferredModelSet")
        # Refuse accidental reuse of an application's existing preference file.
        self.read(preference.application_id)
        payload = json.dumps(preference.to_data(), sort_keys=True,
                             separators=(",", ":"), allow_nan=False) + "\n"
        if len(payload.encode("utf-8")) > 1_048_576:
            raise ValueError("preference file exceeds the size limit")
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent,
                                             prefix=".preference-", delete=False) as stream:
                temporary = stream.name
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            temporary = None
        except OSError:
            raise ValueError("preference file cannot be written") from None
        finally:
            if temporary is not None:
                try:
                    Path(temporary).unlink(missing_ok=True)
                except OSError:
                    pass
