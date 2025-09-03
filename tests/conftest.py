"""
Shared pytest fixtures for the FetaQA testing suite.

This module contains common fixtures used across unit and integration tests.
"""

import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, MagicMock

import pytest
import pandas as pd

# Optional imports for ML testing
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

try:
    from transformers import AutoTokenizer, AutoConfig
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_fetaqa_data():
    """Sample FetaQA dataset entries for testing."""
    return [
        {
            "id": "test_1",
            "question": "What is the population of Tokyo?",
            "context": "Tokyo is the capital of Japan with a population of 14 million.",
            "answer": "14 million",
            "table": [
                ["City", "Country", "Population"],
                ["Tokyo", "Japan", "14 million"],
                ["New York", "USA", "8.4 million"]
            ]
        },
        {
            "id": "test_2", 
            "question": "Which city has the highest GDP?",
            "context": "Economic data shows Tokyo leads in GDP.",
            "answer": "Tokyo",
            "table": [
                ["City", "GDP (Trillion USD)"],
                ["Tokyo", "1.6"],
                ["New York", "1.5"]
            ]
        }
    ]


@pytest.fixture
def sample_json_file(temp_dir, sample_fetaqa_data):
    """Create a temporary JSON file with sample data."""
    json_file = temp_dir / "sample_data.json"
    with open(json_file, 'w') as f:
        json.dump(sample_fetaqa_data, f, indent=2)
    return json_file


@pytest.fixture
def sample_jsonl_file(temp_dir, sample_fetaqa_data):
    """Create a temporary JSONL file with sample data."""
    jsonl_file = temp_dir / "sample_data.jsonl"
    with open(jsonl_file, 'w') as f:
        for item in sample_fetaqa_data:
            f.write(json.dumps(item) + '\n')
    return jsonl_file


@pytest.fixture
def sample_dataframe(sample_fetaqa_data):
    """Create a pandas DataFrame with sample data."""
    return pd.DataFrame(sample_fetaqa_data)


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing."""
    tokenizer = Mock()
    tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    tokenizer.decode.return_value = "sample decoded text"
    tokenizer.tokenize.return_value = ["sample", "tokens"]
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1
    tokenizer.max_length = 512
    return tokenizer


@pytest.fixture
def mock_model():
    """Mock transformer model for testing."""
    model = Mock()
    # Mock generate method
    if HAS_TORCH:
        model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        # Mock forward pass
        mock_output = Mock()
        mock_output.logits = torch.randn(1, 5, 32128)  # Batch, seq_len, vocab_size
        mock_output.loss = torch.tensor(0.5)
        model.return_value = mock_output
    else:
        model.generate.return_value = [[1, 2, 3, 4, 5]]
        mock_output = Mock()
        mock_output.logits = "mock_logits"
        mock_output.loss = 0.5
        model.return_value = mock_output
    
    model.eval = Mock(return_value=model)
    model.train = Mock(return_value=model)
    return model


@pytest.fixture
def mock_config():
    """Mock model configuration."""
    config = {
        "model_name_or_path": "t5-small",
        "max_input_length": 512,
        "max_output_length": 128,
        "num_train_epochs": 3,
        "learning_rate": 5e-5,
        "batch_size": 8,
        "gradient_accumulation_steps": 1,
        "warmup_steps": 100,
        "logging_steps": 10,
        "save_steps": 500,
        "eval_steps": 100,
        "output_dir": "./outputs",
        "overwrite_output_dir": True,
        "do_train": True,
        "do_eval": True,
        "evaluation_strategy": "steps",
        "save_strategy": "steps",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False
    }
    return config


@pytest.fixture
def mock_training_args(mock_config):
    """Mock training arguments object."""
    args = Mock()
    for key, value in mock_config.items():
        setattr(args, key, value)
    return args


@pytest.fixture
def mock_dataset():
    """Mock dataset object for testing."""
    dataset = Mock()
    dataset.__len__ = Mock(return_value=100)
    if HAS_TORCH:
        dataset.__getitem__ = Mock(return_value={
            'input_ids': torch.tensor([1, 2, 3, 4, 5]),
            'attention_mask': torch.tensor([1, 1, 1, 1, 1]),
            'labels': torch.tensor([2, 3, 4, 5, 6])
        })
    else:
        dataset.__getitem__ = Mock(return_value={
            'input_ids': [1, 2, 3, 4, 5],
            'attention_mask': [1, 1, 1, 1, 1],
            'labels': [2, 3, 4, 5, 6]
        })
    return dataset


@pytest.fixture
def sample_predictions():
    """Sample model predictions for testing."""
    return [
        "Tokyo has 14 million people",
        "Tokyo has the highest GDP",
        "The answer is Tokyo"
    ]


@pytest.fixture
def sample_references():
    """Sample reference answers for testing."""
    return [
        "14 million",
        "Tokyo", 
        "Tokyo"
    ]


@pytest.fixture
def mock_file_system(temp_dir):
    """Setup a mock file system structure for testing."""
    # Create directory structure
    (temp_dir / "data").mkdir()
    (temp_dir / "outputs").mkdir()
    (temp_dir / "configs").mkdir()
    
    # Create sample config file
    config = {
        "model_name_or_path": "t5-small",
        "max_input_length": 512,
        "max_output_length": 128
    }
    with open(temp_dir / "configs" / "test_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create sample data files
    sample_data = [{"id": 1, "question": "test?", "answer": "test"}]
    with open(temp_dir / "data" / "train.json", 'w') as f:
        json.dump(sample_data, f)
    
    return temp_dir


@pytest.fixture
def mock_logger():
    """Mock logger for testing."""
    logger = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.debug = Mock()
    return logger


@pytest.fixture
def mock_torch_device():
    """Mock PyTorch device."""
    if HAS_TORCH:
        return torch.device("cpu")
    else:
        # Return a mock device object
        device_mock = Mock()
        device_mock.type = "cpu"
        return device_mock


@pytest.fixture
def sample_table_data():
    """Sample table data for testing."""
    return [
        ["Name", "Age", "City"],
        ["Alice", "25", "New York"],
        ["Bob", "30", "San Francisco"],
        ["Charlie", "35", "Los Angeles"]
    ]


@pytest.fixture
def mock_transformer_pipeline():
    """Mock transformer pipeline for testing."""
    pipeline = Mock()
    pipeline.return_value = [{"generated_text": "Sample generated response"}]
    return pipeline


# Pytest configuration hooks
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location."""
    for item in items:
        # Mark tests in unit/ directory as unit tests
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        # Mark tests in integration/ directory as integration tests
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)