"""Shared pytest fixtures for MagicDrive test suite."""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict, Any

import pytest
import numpy as np
import torch
from PIL import Image


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Provide a mock configuration dictionary for testing."""
    return {
        "model": {
            "type": "magicdrive",
            "pretrained": None,
            "use_fp16": False,
        },
        "dataset": {
            "type": "nuscenes",
            "data_root": "/mock/data/root",
            "batch_size": 2,
            "num_workers": 0,
        },
        "optimizer": {
            "type": "adam",
            "lr": 1e-4,
            "weight_decay": 0.0,
        },
        "scheduler": {
            "type": "constant",
        },
        "training": {
            "num_epochs": 1,
            "log_interval": 10,
            "save_interval": 100,
            "val_interval": 50,
        },
    }


@pytest.fixture
def sample_image() -> np.ndarray:
    """Create a sample image array for testing."""
    return np.random.randint(0, 255, size=(256, 256, 3), dtype=np.uint8)


@pytest.fixture
def sample_pil_image(sample_image) -> Image.Image:
    """Create a sample PIL Image for testing."""
    return Image.fromarray(sample_image)


@pytest.fixture
def sample_tensor() -> torch.Tensor:
    """Create a sample tensor for testing."""
    return torch.randn(1, 3, 256, 256)


@pytest.fixture
def sample_batch() -> Dict[str, torch.Tensor]:
    """Create a sample batch dictionary for testing."""
    return {
        "images": torch.randn(2, 6, 3, 256, 256),  # 2 samples, 6 views
        "bev_map": torch.randn(2, 3, 200, 200),
        "bboxes_3d": torch.randn(2, 10, 7),  # 10 boxes per sample
        "labels": torch.randint(0, 10, (2, 10)),
        "camera_intrinsics": torch.randn(2, 6, 3, 3),
        "camera_extrinsics": torch.randn(2, 6, 4, 4),
    }


@pytest.fixture
def mock_dataset_item() -> Dict[str, Any]:
    """Create a mock dataset item for testing."""
    return {
        "img_metas": {
            "scene_token": "test_scene_001",
            "sample_token": "test_sample_001",
            "timestamp": 1234567890,
            "camera_names": ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
                           "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"],
        },
        "img": np.random.randint(0, 255, size=(6, 3, 256, 256), dtype=np.uint8),
        "gt_bboxes_3d": np.random.randn(10, 7).astype(np.float32),
        "gt_labels_3d": np.random.randint(0, 10, size=(10,)),
    }


@pytest.fixture
def mock_model_output() -> Dict[str, torch.Tensor]:
    """Create mock model output for testing."""
    return {
        "pred_images": torch.randn(2, 6, 3, 256, 256),
        "loss_total": torch.tensor(0.5),
        "loss_rgb": torch.tensor(0.3),
        "loss_depth": torch.tensor(0.1),
        "loss_bbox": torch.tensor(0.1),
    }


@pytest.fixture(autouse=True)
def setup_environment(monkeypatch):
    """Set up test environment variables."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).parent.parent))
    monkeypatch.setenv("TEST_MODE", "1")


@pytest.fixture
def mock_checkpoint(temp_dir) -> Path:
    """Create a mock checkpoint file for testing."""
    checkpoint_path = temp_dir / "checkpoint.pth"
    checkpoint = {
        "model_state_dict": {"dummy_param": torch.randn(10, 10)},
        "optimizer_state_dict": {"dummy_state": torch.zeros(1)},
        "epoch": 10,
        "best_metric": 0.95,
    }
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


@pytest.fixture
def disable_gpu():
    """Fixture to disable GPU for testing."""
    def _disable():
        torch.cuda.is_available = lambda: False
    
    original_is_available = torch.cuda.is_available
    _disable()
    yield
    torch.cuda.is_available = original_is_available


@pytest.fixture
def capture_logs(caplog):
    """Fixture to capture and return logs."""
    with caplog.at_level("DEBUG"):
        yield caplog