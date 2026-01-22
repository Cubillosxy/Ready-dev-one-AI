"""Tests for path utilities."""
import os
from rdoai.utils.paths import repo_root, ensure_dir


def test_repo_root():
    root = repo_root()
    assert os.path.isdir(root)
    # Check that rdoai is a child of root
    assert os.path.isdir(os.path.join(root, "rdoai"))


def test_ensure_dir(tmp_path):
    new_dir = tmp_path / "subdir" / "deep"
    assert not new_dir.exists()
    
    ensure_dir(str(new_dir))
    
    assert new_dir.exists()
    assert new_dir.is_dir()


def test_ensure_dir_already_exists(tmp_path):
    existing = tmp_path / "exists"
    existing.mkdir()
    
    ensure_dir(str(existing)) # Should not raise
    assert existing.exists()
