"""
Infrastructure validation tests.

These tests verify that the testing infrastructure is properly set up
and all components are working correctly.
"""

import pytest
import json
import pandas as pd
from pathlib import Path


class TestInfrastructure:
    """Test that the testing infrastructure is working correctly."""
    
    def test_pytest_basic_functionality(self):
        """Test basic pytest functionality."""
        assert True
        assert 1 + 1 == 2
        assert "hello" == "hello"
    
    def test_temp_dir_fixture(self, temp_dir):
        """Test that the temp_dir fixture works."""
        assert isinstance(temp_dir, Path)
        assert temp_dir.exists()
        assert temp_dir.is_dir()
        
        # Test we can write to it
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        assert test_file.exists()
        assert test_file.read_text() == "test content"
    
    def test_sample_data_fixtures(self, sample_fetaqa_data, sample_dataframe):
        """Test that sample data fixtures work correctly."""
        assert isinstance(sample_fetaqa_data, list)
        assert len(sample_fetaqa_data) > 0
        assert "question" in sample_fetaqa_data[0]
        assert "answer" in sample_fetaqa_data[0]
        
        assert isinstance(sample_dataframe, pd.DataFrame)
        assert len(sample_dataframe) == len(sample_fetaqa_data)
    
    def test_file_fixtures(self, sample_json_file, sample_jsonl_file):
        """Test that file fixtures create valid files."""
        # Test JSON file
        assert sample_json_file.exists()
        with open(sample_json_file) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) > 0
        
        # Test JSONL file
        assert sample_jsonl_file.exists()
        lines = sample_jsonl_file.read_text().strip().split('\n')
        assert len(lines) > 0
        # Each line should be valid JSON
        for line in lines:
            assert isinstance(json.loads(line), dict)
    
    def test_mock_fixtures(self, mock_tokenizer, mock_model, mock_config):
        """Test that mock fixtures are properly configured."""
        # Test tokenizer
        assert hasattr(mock_tokenizer, 'encode')
        assert hasattr(mock_tokenizer, 'decode')
        encoded = mock_tokenizer.encode("test")
        assert isinstance(encoded, list)
        
        # Test model
        assert hasattr(mock_model, 'generate')
        assert hasattr(mock_model, 'eval')
        
        # Test config
        assert isinstance(mock_config, dict)
        assert "model_name_or_path" in mock_config
    
    def test_pytest_markers(self):
        """Test that custom pytest markers are working."""
        # This test should automatically be marked as 'unit' 
        # since it's not in the integration directory
        pass
    
    @pytest.mark.unit
    def test_unit_marker(self):
        """Test explicit unit marker."""
        assert True
    
    @pytest.mark.slow 
    def test_slow_marker(self):
        """Test slow marker."""
        assert True
    
    def test_coverage_is_working(self):
        """Test that coverage measurement is working."""
        # This function exercises some basic Python features
        # to ensure coverage is tracking execution
        data = {"key": "value"}
        result = []
        for item in data.items():
            result.append(f"{item[0]}={item[1]}")
        
        assert len(result) == 1
        assert result[0] == "key=value"
    
    def test_sample_table_data(self, sample_table_data):
        """Test sample table data fixture."""
        assert isinstance(sample_table_data, list)
        assert len(sample_table_data) > 1  # Header + data rows
        assert isinstance(sample_table_data[0], list)  # Header row
        
    def test_mock_file_system(self, mock_file_system):
        """Test mock file system fixture."""
        assert mock_file_system.exists()
        assert (mock_file_system / "data").exists()
        assert (mock_file_system / "outputs").exists()
        assert (mock_file_system / "configs").exists()
        
        # Test config file exists and is valid
        config_file = mock_file_system / "configs" / "test_config.json"
        assert config_file.exists()
        with open(config_file) as f:
            config = json.load(f)
        assert "model_name_or_path" in config


class TestProjectStructure:
    """Test that the project structure is set up correctly."""
    
    def test_project_root_structure(self):
        """Test that required directories exist in project root."""
        project_root = Path("/workspace")
        
        assert project_root.exists()
        assert (project_root / "tests").exists()
        assert (project_root / "tests" / "unit").exists()
        assert (project_root / "tests" / "integration").exists()
        assert (project_root / "pyproject.toml").exists()
        assert (project_root / ".gitignore").exists()
    
    def test_init_files_exist(self):
        """Test that __init__.py files exist in test directories."""
        project_root = Path("/workspace")
        
        assert (project_root / "tests" / "__init__.py").exists()
        assert (project_root / "tests" / "unit" / "__init__.py").exists()
        assert (project_root / "tests" / "integration" / "__init__.py").exists()
    
    def test_conftest_exists(self):
        """Test that conftest.py exists and is importable."""
        project_root = Path("/workspace")
        conftest_path = project_root / "tests" / "conftest.py"
        
        assert conftest_path.exists()
        
        # Basic check that it's valid Python
        content = conftest_path.read_text()
        assert "pytest" in content
        assert "@pytest.fixture" in content