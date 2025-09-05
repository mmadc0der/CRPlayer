"""
Tests for data transfer objects (DTOs).
"""

import pytest
from pydantic import ValidationError

from dto import (
    ErrorResponse, FrameQuery, ImageQuery, SaveRegressionRequest,
    SaveSingleLabelRequest, SaveMultilabelRequest, UpsertDatasetSessionSettingsRequest
)


class TestErrorResponse:
    """Test ErrorResponse DTO."""

    def test_error_response_basic(self):
        """Test basic ErrorResponse creation."""
        error = ErrorResponse(code="test_error", message="Test error message")
        
        assert error.code == "test_error"
        assert error.message == "Test error message"
        assert error.details is None

    def test_error_response_with_details(self):
        """Test ErrorResponse with details."""
        details = {"field": "value", "count": 42}
        error = ErrorResponse(
            code="validation_error",
            message="Validation failed",
            details=details
        )
        
        assert error.code == "validation_error"
        assert error.message == "Validation failed"
        assert error.details == details

    def test_error_response_dict_serialization(self):
        """Test ErrorResponse dict serialization."""
        error = ErrorResponse(
            code="test_error",
            message="Test message",
            details={"key": "value"}
        )
        
        result = error.dict()
        expected = {
            "code": "test_error",
            "message": "Test message",
            "details": {"key": "value"}
        }
        
        assert result == expected


class TestFrameQuery:
    """Test FrameQuery DTO."""

    def test_frame_query_valid(self):
        """Test valid FrameQuery creation."""
        query = FrameQuery(session_id="test_session", idx=42)
        
        assert query.session_id == "test_session"
        assert query.idx == 42

    def test_frame_query_session_id_trimming(self):
        """Test session_id trimming."""
        query = FrameQuery(session_id="  test_session  ", idx=0)
        
        assert query.session_id == "test_session"

    def test_frame_query_missing_fields(self):
        """Test FrameQuery with missing required fields."""
        with pytest.raises(ValidationError):
            FrameQuery(session_id="test")  # Missing idx
        
        with pytest.raises(ValidationError):
            FrameQuery(idx=0)  # Missing session_id


class TestImageQuery:
    """Test ImageQuery DTO."""

    def test_image_query_valid(self):
        """Test valid ImageQuery creation."""
        query = ImageQuery(session_id="test_session", idx=10)
        
        assert query.session_id == "test_session"
        assert query.idx == 10

    def test_image_query_session_id_trimming(self):
        """Test session_id trimming."""
        query = ImageQuery(session_id="  spaced_session  ", idx=5)
        
        assert query.session_id == "spaced_session"


class TestSaveRegressionRequest:
    """Test SaveRegressionRequest DTO."""

    def test_save_regression_with_frame_id(self):
        """Test SaveRegressionRequest with frame_id."""
        request = SaveRegressionRequest(
            session_id="test_session",
            dataset_id=1,
            frame_id="frame_001",
            value=42.5
        )
        
        assert request.session_id == "test_session"
        assert request.dataset_id == 1
        assert request.frame_id == "frame_001"
        assert request.frame_idx is None
        assert request.value == 42.5

    def test_save_regression_with_frame_idx(self):
        """Test SaveRegressionRequest with frame_idx."""
        request = SaveRegressionRequest(
            session_id="test_session",
            dataset_id=1,
            frame_idx=10,
            value=99.9
        )
        
        assert request.session_id == "test_session"
        assert request.dataset_id == 1
        assert request.frame_id is None
        assert request.frame_idx == 10
        assert request.value == 99.9

    def test_save_regression_with_override_settings(self):
        """Test SaveRegressionRequest with override settings."""
        override_settings = {"custom_field": "value", "confidence": 0.95}
        request = SaveRegressionRequest(
            session_id="test_session",
            dataset_id=1,
            frame_id="frame_001",
            value=42.5,
            override_settings=override_settings
        )
        
        assert request.override_settings == override_settings

    def test_save_regression_missing_frame_identifier(self):
        """Test SaveRegressionRequest without frame_id or frame_idx fails."""
        with pytest.raises(ValidationError, match="Either frame_id or frame_idx must be provided"):
            SaveRegressionRequest(
                session_id="test_session",
                dataset_id=1,
                value=42.5
            )

    def test_save_regression_session_id_trimming(self):
        """Test session_id trimming in SaveRegressionRequest."""
        request = SaveRegressionRequest(
            session_id="  test_session  ",
            dataset_id=1,
            frame_id="frame_001",
            value=42.5
        )
        
        assert request.session_id == "test_session"

    def test_save_regression_both_frame_identifiers(self):
        """Test SaveRegressionRequest with both frame_id and frame_idx."""
        # Should be valid - both can be provided
        request = SaveRegressionRequest(
            session_id="test_session",
            dataset_id=1,
            frame_id="frame_001",
            frame_idx=10,
            value=42.5
        )
        
        assert request.frame_id == "frame_001"
        assert request.frame_idx == 10


class TestSaveSingleLabelRequest:
    """Test SaveSingleLabelRequest DTO."""

    def test_save_single_label_with_class_id(self):
        """Test SaveSingleLabelRequest with class_id."""
        request = SaveSingleLabelRequest(
            session_id="test_session",
            dataset_id=1,
            frame_id="frame_001",
            class_id=5
        )
        
        assert request.class_id == 5
        assert request.category_name is None

    def test_save_single_label_with_category_name(self):
        """Test SaveSingleLabelRequest with category_name."""
        request = SaveSingleLabelRequest(
            session_id="test_session",
            dataset_id=1,
            frame_id="frame_001",
            category_name="battle"
        )
        
        assert request.class_id is None
        assert request.category_name == "battle"

    def test_save_single_label_missing_class_identifier(self):
        """Test SaveSingleLabelRequest without class_id or category_name fails."""
        with pytest.raises(ValidationError, match="Either class_id or category_name must be provided"):
            SaveSingleLabelRequest(
                session_id="test_session",
                dataset_id=1,
                frame_id="frame_001"
            )

    def test_save_single_label_empty_category_name(self):
        """Test SaveSingleLabelRequest with empty category_name fails."""
        with pytest.raises(ValidationError, match="Either class_id or category_name must be provided"):
            SaveSingleLabelRequest(
                session_id="test_session",
                dataset_id=1,
                frame_id="frame_001",
                category_name=""
            )

    def test_save_single_label_whitespace_category_name(self):
        """Test SaveSingleLabelRequest with whitespace-only category_name fails."""
        with pytest.raises(ValidationError, match="Either class_id or category_name must be provided"):
            SaveSingleLabelRequest(
                session_id="test_session",
                dataset_id=1,
                frame_id="frame_001",
                category_name="   "
            )

    def test_save_single_label_both_class_identifiers(self):
        """Test SaveSingleLabelRequest with both class_id and category_name."""
        # Should be valid - both can be provided
        request = SaveSingleLabelRequest(
            session_id="test_session",
            dataset_id=1,
            frame_id="frame_001",
            class_id=5,
            category_name="battle"
        )
        
        assert request.class_id == 5
        assert request.category_name == "battle"


class TestSaveMultilabelRequest:
    """Test SaveMultilabelRequest DTO."""

    def test_save_multilabel_with_class_ids(self):
        """Test SaveMultilabelRequest with class_ids."""
        class_ids = [1, 2, 3]
        request = SaveMultilabelRequest(
            session_id="test_session",
            dataset_id=1,
            frame_id="frame_001",
            class_ids=class_ids
        )
        
        assert request.class_ids == class_ids
        assert request.category_names is None

    def test_save_multilabel_with_category_names(self):
        """Test SaveMultilabelRequest with category_names."""
        category_names = ["battle", "boss", "critical"]
        request = SaveMultilabelRequest(
            session_id="test_session",
            dataset_id=1,
            frame_id="frame_001",
            class_ids=[],  # Required field, but can be empty
            category_names=category_names
        )
        
        assert request.class_ids == []
        assert request.category_names == category_names

    def test_save_multilabel_empty_class_ids(self):
        """Test SaveMultilabelRequest with empty class_ids list."""
        request = SaveMultilabelRequest(
            session_id="test_session",
            dataset_id=1,
            frame_id="frame_001",
            class_ids=[]
        )
        
        assert request.class_ids == []

    def test_save_multilabel_missing_frame_identifier(self):
        """Test SaveMultilabelRequest without frame_id or frame_idx fails."""
        with pytest.raises(ValidationError, match="Either frame_id or frame_idx must be provided"):
            SaveMultilabelRequest(
                session_id="test_session",
                dataset_id=1,
                class_ids=[1, 2, 3]
            )

    def test_save_multilabel_with_override_settings(self):
        """Test SaveMultilabelRequest with override settings."""
        override_settings = {"confidence_threshold": 0.8}
        request = SaveMultilabelRequest(
            session_id="test_session",
            dataset_id=1,
            frame_id="frame_001",
            class_ids=[1, 2],
            override_settings=override_settings
        )
        
        assert request.override_settings == override_settings


class TestUpsertDatasetSessionSettingsRequest:
    """Test UpsertDatasetSessionSettingsRequest DTO."""

    def test_upsert_settings_valid(self):
        """Test valid UpsertDatasetSessionSettingsRequest."""
        settings = {"auto_advance": True, "hotkeys": {"1": "menu"}}
        request = UpsertDatasetSessionSettingsRequest(
            dataset_id=1,
            session_id="test_session",
            settings=settings
        )
        
        assert request.dataset_id == 1
        assert request.session_id == "test_session"
        assert request.settings == settings

    def test_upsert_settings_session_id_trimming(self):
        """Test session_id trimming in UpsertDatasetSessionSettingsRequest."""
        request = UpsertDatasetSessionSettingsRequest(
            dataset_id=1,
            session_id="  test_session  ",
            settings={}
        )
        
        assert request.session_id == "test_session"

    def test_upsert_settings_empty_settings(self):
        """Test UpsertDatasetSessionSettingsRequest with empty settings."""
        request = UpsertDatasetSessionSettingsRequest(
            dataset_id=1,
            session_id="test_session",
            settings={}
        )
        
        assert request.settings == {}

    def test_upsert_settings_complex_settings(self):
        """Test UpsertDatasetSessionSettingsRequest with complex settings."""
        settings = {
            "ui_preferences": {
                "theme": "dark",
                "layout": "grid"
            },
            "shortcuts": ["Ctrl+S", "Space", "Enter"],
            "thresholds": {
                "confidence": 0.8,
                "importance": 2
            }
        }
        
        request = UpsertDatasetSessionSettingsRequest(
            dataset_id=1,
            session_id="test_session",
            settings=settings
        )
        
        assert request.settings == settings

    def test_upsert_settings_missing_fields(self):
        """Test UpsertDatasetSessionSettingsRequest with missing required fields."""
        with pytest.raises(ValidationError):
            UpsertDatasetSessionSettingsRequest(
                dataset_id=1,
                session_id="test_session"
                # Missing settings
            )
        
        with pytest.raises(ValidationError):
            UpsertDatasetSessionSettingsRequest(
                dataset_id=1,
                settings={}
                # Missing session_id
            )


class TestDTOEdgeCases:
    """Test edge cases and error conditions for DTOs."""

    def test_pydantic_version_compatibility(self):
        """Test that DTOs work with different Pydantic versions."""
        # This test ensures our compatibility layer works
        request = SaveRegressionRequest(
            session_id="test",
            dataset_id=1,
            frame_id="frame_001",
            value=42.0
        )
        
        # Should work regardless of Pydantic version
        assert request.session_id == "test"
        assert request.value == 42.0

    def test_none_values_handling(self):
        """Test handling of None values in optional fields."""
        request = SaveRegressionRequest(
            session_id="test",
            dataset_id=1,
            frame_id="frame_001",
            value=42.0,
            override_settings=None
        )
        
        assert request.override_settings is None

    def test_type_coercion(self):
        """Test type coercion in DTOs."""
        # String dataset_id should be coerced to int
        request = SaveRegressionRequest(
            session_id="test",
            dataset_id="1",  # String instead of int
            frame_id="frame_001",
            value=42.0
        )
        
        assert request.dataset_id == 1
        assert isinstance(request.dataset_id, int)