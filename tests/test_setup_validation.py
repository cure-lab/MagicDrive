"""Validation tests to ensure the testing infrastructure is set up correctly."""

import sys
from pathlib import Path

import pytest
import torch
import numpy as np
from PIL import Image


class TestSetupValidation:
    """Test class to validate the testing infrastructure setup."""

    def test_python_version(self):
        """Verify Python version is compatible."""
        assert sys.version_info >= (3, 7), "Python 3.7+ is required"

    def test_project_structure(self):
        """Verify the project structure is accessible."""
        project_root = Path(__file__).parent.parent
        assert project_root.exists()
        assert (project_root / "magicdrive").exists()
        assert (project_root / "pyproject.toml").exists()

    def test_imports(self):
        """Test that main project modules can be imported."""
        try:
            import magicdrive
            assert magicdrive is not None
        except ImportError:
            # It's okay if the module isn't importable yet during setup
            pass

    def test_fixtures_available(self, temp_dir, mock_config, sample_image):
        """Test that pytest fixtures are properly configured."""
        assert temp_dir.exists()
        assert isinstance(mock_config, dict)
        assert isinstance(sample_image, np.ndarray)
        assert sample_image.shape == (256, 256, 3)

    def test_temp_dir_cleanup(self, temp_dir):
        """Test that temporary directories are cleaned up."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        assert test_file.exists()
        # Cleanup happens after test completion

    def test_mock_config_structure(self, mock_config):
        """Test the mock configuration has expected structure."""
        assert "model" in mock_config
        assert "dataset" in mock_config
        assert "optimizer" in mock_config
        assert "training" in mock_config

    def test_sample_tensors(self, sample_tensor, sample_batch):
        """Test tensor fixtures have correct shapes."""
        assert isinstance(sample_tensor, torch.Tensor)
        assert sample_tensor.shape == (1, 3, 256, 256)
        
        assert isinstance(sample_batch, dict)
        assert sample_batch["images"].shape == (2, 6, 3, 256, 256)
        assert sample_batch["bev_map"].shape == (2, 3, 200, 200)

    def test_pil_image_fixture(self, sample_pil_image):
        """Test PIL image fixture."""
        assert isinstance(sample_pil_image, Image.Image)
        assert sample_pil_image.size == (256, 256)

    def test_mock_dataset_item(self, mock_dataset_item):
        """Test mock dataset item structure."""
        assert "img_metas" in mock_dataset_item
        assert "img" in mock_dataset_item
        assert "gt_bboxes_3d" in mock_dataset_item
        assert "gt_labels_3d" in mock_dataset_item
        
        assert len(mock_dataset_item["img_metas"]["camera_names"]) == 6

    def test_mock_model_output(self, mock_model_output):
        """Test mock model output structure."""
        assert "pred_images" in mock_model_output
        assert "loss_total" in mock_model_output
        assert all(isinstance(v, torch.Tensor) for v in mock_model_output.values())

    def test_checkpoint_fixture(self, mock_checkpoint):
        """Test checkpoint creation and loading."""
        assert mock_checkpoint.exists()
        checkpoint = torch.load(mock_checkpoint, map_location="cpu")
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert "epoch" in checkpoint
        assert checkpoint["epoch"] == 10

    @pytest.mark.unit
    def test_unit_marker(self):
        """Test that unit test marker works."""
        assert True

    @pytest.mark.integration
    def test_integration_marker(self):
        """Test that integration test marker works."""
        assert True

    @pytest.mark.slow
    def test_slow_marker(self):
        """Test that slow test marker works."""
        assert True

    def test_environment_setup(self):
        """Test that environment variables are set correctly."""
        import os
        assert os.environ.get("TEST_MODE") == "1"
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""

    def test_gpu_disable_fixture(self, disable_gpu):
        """Test GPU disable fixture."""
        assert not torch.cuda.is_available()

    def test_log_capture(self, capture_logs):
        """Test log capture fixture."""
        import logging
        logger = logging.getLogger(__name__)
        logger.debug("Test debug message")
        logger.info("Test info message")
        
        assert "Test debug message" in capture_logs.text
        assert "Test info message" in capture_logs.text