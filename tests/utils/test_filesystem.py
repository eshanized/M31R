# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Tests for filesystem utilities — atomic writes, safe reads, and safe deletes.

Atomic writes are tested by verifying that the target file either has the full
new content or doesn't exist at all. There should never be a partially written
file.
"""

from pathlib import Path

import pytest

from m31r.utils.filesystem import atomic_write, atomic_write_bytes, safe_delete, safe_read


class TestAtomicWrite:
    def test_writes_content_successfully(self, tmp_path: Path) -> None:
        target = tmp_path / "output.txt"
        atomic_write(target, "hello world")

        assert target.exists()
        assert target.read_text(encoding="utf-8") == "hello world"

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "deep" / "output.txt"
        atomic_write(target, "nested content")

        assert target.exists()
        assert target.read_text(encoding="utf-8") == "nested content"

    def test_overwrites_existing_file(self, tmp_path: Path) -> None:
        target = tmp_path / "overwrite.txt"
        atomic_write(target, "first version")
        atomic_write(target, "second version")

        assert target.read_text(encoding="utf-8") == "second version"

    def test_no_leftover_temp_files_on_success(self, tmp_path: Path) -> None:
        target = tmp_path / "clean.txt"
        atomic_write(target, "clean write")

        temp_files = list(tmp_path.glob(".m31r_tmp_*"))
        assert len(temp_files) == 0


class TestAtomicWriteBytes:
    def test_writes_binary_content(self, tmp_path: Path) -> None:
        target = tmp_path / "binary.bin"
        data = b"\x00\x01\x02\xff"
        atomic_write_bytes(target, data)

        assert target.read_bytes() == data


class TestSafeRead:
    def test_reads_existing_file(self, tmp_path: Path) -> None:
        target = tmp_path / "readable.txt"
        target.write_text("read me", encoding="utf-8")

        content = safe_read(target)
        assert content == "read me"

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            safe_read(tmp_path / "missing.txt")

    def test_raises_on_directory(self, tmp_path: Path) -> None:
        with pytest.raises(IsADirectoryError):
            safe_read(tmp_path)


class TestSafeDelete:
    def test_deletes_existing_file(self, tmp_path: Path) -> None:
        target = tmp_path / "deleteme.txt"
        target.write_text("delete me", encoding="utf-8")

        result = safe_delete(target)
        assert result is True
        assert not target.exists()

    def test_returns_false_for_missing_file(self, tmp_path: Path) -> None:
        result = safe_delete(tmp_path / "nonexistent.txt")
        assert result is False
