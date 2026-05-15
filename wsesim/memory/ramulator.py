"""Optional Ramulator integration placeholder."""

from __future__ import annotations


class RamulatorBackendUnavailable(RuntimeError):
    pass


def create_ramulator_backend(*args, **kwargs):
    raise RamulatorBackendUnavailable(
        "Ramulator backend is not implemented in this initial version."
    )
