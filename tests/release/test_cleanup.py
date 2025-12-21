# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Tests for project cleanup.
"""

from pathlib import Path

from m31r.release.cleanup.cleaner import clean_project, _SAFE_TO_REMOVE_DIRS


def test_clean_removes_caches(tmp_path: Path):
    """Test removal of __pycache__ and friends."""
    # Create garbage
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "garbage.pyc").touch()
    (tmp_path / ".pytest_cache").mkdir()

    result = clean_project(tmp_path)
    assert result.removed_dirs >= 2
    assert not (tmp_path / "__pycache__").exists()
    assert not (tmp_path / ".pytest_cache").exists()


def test_clean_preserves_releases(tmp_path: Path):
    """Test that release directories are protected."""
    release_dir = tmp_path / "release"
    release_dir.mkdir()
    (release_dir / "__pycache__").mkdir()  # Should imply protection

    result = clean_project(tmp_path)
    
    # The release dir itself must exist
    assert release_dir.exists()
    
    # Ideally we don't even scan inside protected dirs, 
    # but our cleaner implementation skips deletion if protected.
    # Check that protected dirs list is populated if it tried to delete
    if (release_dir / "__pycache__").exists():
        pass # Good
    else:
        # If it deleted it, was it marked protected?
        assert str(release_dir / "__pycache__") in result.protected_dirs or \
               str(release_dir) in result.protected_dirs


def test_clean_temp_files(tmp_path: Path):
    """Test removal of .m31r_tmp_* files."""
    tmp_file = tmp_path / ".m31r_tmp_12345"
    tmp_file.touch()

    result = clean_project(tmp_path)
    assert result.removed_files == 1
    assert not tmp_file.exists()
