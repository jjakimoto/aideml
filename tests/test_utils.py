"""
Tests for aide.utils module functions.
"""
import os
import tempfile
import zipfile
from pathlib import Path
import pytest

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import just the function we need to avoid circular imports
from aide.utils import extract_archives


class TestExtractArchives:
    """Test cases for the extract_archives function."""
    
    def test_extract_simple_zip(self, tmp_path):
        """Test extracting a simple zip file."""
        # Create a test zip file
        zip_path = tmp_path / "test.zip"
        test_file_content = "Hello, world!"
        
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("test.txt", test_file_content)
        
        # Extract archives
        extract_archives(tmp_path)
        
        # Check that zip was extracted
        extracted_dir = tmp_path / "test"
        assert extracted_dir.exists()
        assert (extracted_dir / "test.txt").exists()
        assert (extracted_dir / "test.txt").read_text() == test_file_content
        
        # Check that zip was removed
        assert not zip_path.exists()
    
    def test_extract_nested_zip(self, tmp_path):
        """Test extracting nested zip files (currently TODO)."""
        # Create inner zip
        inner_zip_path = tmp_path / "inner.zip"
        inner_content = "Inner content"
        
        with zipfile.ZipFile(inner_zip_path, 'w') as zf:
            zf.writestr("inner.txt", inner_content)
        
        # Create outer zip containing the inner zip
        outer_zip_path = tmp_path / "outer.zip"
        with zipfile.ZipFile(outer_zip_path, 'w') as zf:
            zf.write(inner_zip_path, "inner.zip")
        
        # Remove the original inner zip
        inner_zip_path.unlink()
        
        # Extract archives
        extract_archives(tmp_path)
        
        # Check that outer zip was extracted
        outer_dir = tmp_path / "outer"
        assert outer_dir.exists()
        
        # Check that inner zip was also extracted
        inner_dir = outer_dir / "inner"
        assert inner_dir.exists()
        assert (inner_dir / "inner.txt").exists()
        assert (inner_dir / "inner.txt").read_text() == inner_content
        
        # Check that both zips were removed
        assert not outer_zip_path.exists()
        assert not (outer_dir / "inner.zip").exists()
    
    def test_extract_multiple_nested_levels(self, tmp_path):
        """Test extracting multiple levels of nested zip files."""
        # Create deepest content
        deepest_content = "Deepest content"
        
        # Create level 3 zip
        level3_zip = tmp_path / "level3.zip"
        with zipfile.ZipFile(level3_zip, 'w') as zf:
            zf.writestr("deepest.txt", deepest_content)
        
        # Create level 2 zip containing level 3
        level2_zip = tmp_path / "level2.zip"
        with zipfile.ZipFile(level2_zip, 'w') as zf:
            zf.write(level3_zip, "level3.zip")
        level3_zip.unlink()
        
        # Create level 1 zip containing level 2
        level1_zip = tmp_path / "level1.zip"
        with zipfile.ZipFile(level1_zip, 'w') as zf:
            zf.write(level2_zip, "level2.zip")
        level2_zip.unlink()
        
        # Extract archives
        extract_archives(tmp_path)
        
        # Check that all levels were extracted
        level1_dir = tmp_path / "level1"
        level2_dir = level1_dir / "level2"
        level3_dir = level2_dir / "level3"
        
        assert level1_dir.exists()
        assert level2_dir.exists()
        assert level3_dir.exists()
        assert (level3_dir / "deepest.txt").exists()
        assert (level3_dir / "deepest.txt").read_text() == deepest_content
        
        # Check that all intermediate zips were removed
        assert not level1_zip.exists()
        assert not (level1_dir / "level2.zip").exists()
        assert not (level2_dir / "level3.zip").exists()
    
    def test_skip_existing_directory(self, tmp_path):
        """Test that existing directories are skipped."""
        # Create a directory that would conflict with extraction
        existing_dir = tmp_path / "test"
        existing_dir.mkdir()
        existing_file = existing_dir / "existing.txt"
        existing_file.write_text("Existing content")
        
        # Create a zip with the same name
        zip_path = tmp_path / "test.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("new.txt", "New content")
        
        # Extract archives
        extract_archives(tmp_path)
        
        # Check that existing directory was preserved
        assert existing_dir.exists()
        assert existing_file.exists()
        assert existing_file.read_text() == "Existing content"
        assert not (existing_dir / "new.txt").exists()
        
        # Check that zip still exists (not removed because extraction was skipped)
        assert zip_path.exists()
    
    def test_nested_zip_with_duplicate_names(self, tmp_path):
        """Test handling nested zips with duplicate directory names."""
        # Create content for different levels
        content1 = "Content from level 1"
        content2 = "Content from level 2"
        
        # Create inner zip with a file
        inner_zip = tmp_path / "data.zip"
        with zipfile.ZipFile(inner_zip, 'w') as zf:
            zf.writestr("file.txt", content2)
        
        # Create outer zip also named data.zip containing the inner zip
        outer_zip = tmp_path / "outer.zip"
        with zipfile.ZipFile(outer_zip, 'w') as zf:
            zf.write(inner_zip, "data.zip")
            zf.writestr("file.txt", content1)
        inner_zip.unlink()
        
        # Extract archives
        extract_archives(tmp_path)
        
        # Check that both levels were extracted properly
        outer_dir = tmp_path / "outer"
        inner_dir = outer_dir / "data"
        
        assert outer_dir.exists()
        assert inner_dir.exists()
        assert (outer_dir / "file.txt").read_text() == content1
        assert (inner_dir / "file.txt").read_text() == content2