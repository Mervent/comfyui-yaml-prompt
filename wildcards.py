from __future__ import annotations

from pathlib import Path

__all__ = ["load_lines"]


def load_lines(wildcard_dir: Path, name: str) -> list[str]:
    file_path = wildcard_dir / f"{name}.txt"
    try:
        return [
            line.strip()
            for line in file_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except FileNotFoundError:
        return []
