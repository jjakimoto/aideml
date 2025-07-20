"""Tests for the specialized prompts module."""

import pytest

from aide.utils.specialized_prompts import (
    MLTaskType,
    TaskPromptEnhancement,
    TASK_ENHANCEMENTS,
    detect_ml_task_type,
    enhance_prompt_for_task,
    get_task_specific_review_hints,
    optimize_code_for_task
)


class TestMLTaskType:
    """Test the MLTaskType enumeration."""
    
    def test_task_types(self):
        """Test that all expected task types exist."""
        expected_types = [
            "classification", "regression", "clustering",
            "time_series", "nlp", "computer_vision",
            "reinforcement_learning", "general"
        ]
        
        for expected in expected_types:
            assert any(task.value == expected for task in MLTaskType)


class TestTaskDetection:
    """Test ML task type detection."""
    
    def test_detect_classification(self):
        """Test detecting classification tasks."""
        classification_descriptions = [
            "Classify images into categories",
            "Predict if email is spam or not spam",
            "Multi-class classification of documents",
            "Binary classification task for customer churn"
        ]
        
        for desc in classification_descriptions:
            assert detect_ml_task_type(desc) == MLTaskType.CLASSIFICATION
    
    def test_detect_regression(self):
        """Test detecting regression tasks."""
        regression_descriptions = [
            "Predict house prices based on features",
            "Estimate continuous value of stock prices",
            "Regression model for sales forecasting",
            "Predict numeric rating from reviews"
        ]
        
        for desc in regression_descriptions:
            assert detect_ml_task_type(desc) == MLTaskType.REGRESSION
    
    def test_detect_time_series(self):
        """Test detecting time series tasks."""
        time_series_descriptions = [
            "Time series forecasting of sales",
            "Predict future values based on historical data",
            "Seasonal trend analysis and prediction",
            "Temporal pattern recognition in stock data"
        ]
        
        for desc in time_series_descriptions:
            assert detect_ml_task_type(desc) == MLTaskType.TIME_SERIES
    
    def test_detect_nlp(self):
        """Test detecting NLP tasks."""
        nlp_descriptions = [
            "Text classification for sentiment analysis",
            "Natural language processing of documents",
            "Tokenize and analyze corpus of text",
            "Word embedding and document similarity"
        ]
        
        for desc in nlp_descriptions:
            assert detect_ml_task_type(desc) == MLTaskType.NLP
    
    def test_detect_computer_vision(self):
        """Test detecting computer vision tasks."""
        cv_descriptions = [
            "Image classification using CNN",
            "Object detection in pictures",
            "Visual recognition of handwritten digits",
            "Process pixel data for image segmentation"
        ]
        
        for desc in cv_descriptions:
            assert detect_ml_task_type(desc) == MLTaskType.COMPUTER_VISION
    
    def test_detect_with_data_preview(self):
        """Test detection using both description and data preview."""
        # Classification with data preview
        result = detect_ml_task_type(
            "Predict the outcome",
            "Data contains features X1, X2, X3 and binary label Y"
        )
        assert result == MLTaskType.CLASSIFICATION
        
        # Time series with data preview
        result = detect_ml_task_type(
            "Forecast values",
            "Data has timestamp column and historical measurements"
        )
        assert result == MLTaskType.TIME_SERIES
    
    def test_detect_general(self):
        """Test defaulting to general task type."""
        general_descriptions = [
            "Perform data analysis",
            "Build a model",
            "Machine learning task"
        ]
        
        for desc in general_descriptions:
            assert detect_ml_task_type(desc) == MLTaskType.GENERAL


class TestPromptEnhancement:
    """Test prompt enhancement functionality."""
    
    def test_enhance_classification_prompt(self):
        """Test enhancing a classification task prompt."""
        original_prompt = {
            "Task Description": "Classify customer churn",
            "Instructions": {}
        }
        
        enhanced = enhance_prompt_for_task(
            original_prompt,
            task_type=MLTaskType.CLASSIFICATION
        )
        
        # Check that original content is preserved
        assert "Classify customer churn" in enhanced["Task Description"]
        
        # Check that enhancements are added
        assert "ML Best Practices" in enhanced
        assert "classification" in enhanced["ML Best Practices"]
        assert "Common Pitfalls to Avoid" in enhanced
        assert "Evaluation Guidelines" in enhanced
        
        # Check specific classification guidance
        assert "stratified" in enhanced["ML Best Practices"].lower()
        assert "imbalance" in enhanced["Common Pitfalls to Avoid"].lower()
    
    def test_enhance_regression_prompt(self):
        """Test enhancing a regression task prompt."""
        original_prompt = {
            "Task Description": "Predict house prices",
            "Instructions": {"existing": "value"}
        }
        
        enhanced = enhance_prompt_for_task(
            original_prompt,
            task_type=MLTaskType.REGRESSION
        )
        
        # Check regression-specific content
        assert "scaling" in str(enhanced).lower()
        assert "outliers" in str(enhanced).lower()
        assert "residual" in enhanced["Evaluation Guidelines"].lower()
    
    def test_enhance_with_auto_detection(self):
        """Test enhancement with automatic task detection."""
        original_prompt = {
            "Task Description": "Build a text classifier for sentiment",
            "Instructions": {}
        }
        
        enhanced = enhance_prompt_for_task(
            original_prompt,
            task_description="Build a text classifier for sentiment"
        )
        
        # Should detect NLP task
        assert "ML Best Practices" in enhanced
        assert "nlp" in enhanced["ML Best Practices"].lower()
    
    def test_enhance_preserves_structure(self):
        """Test that enhancement preserves prompt structure."""
        original_prompt = {
            "Introduction": "You are an ML expert",
            "Task Description": "Classification task",
            "Custom Field": "Custom value",
            "Instructions": {
                "Step 1": "Do this",
                "Step 2": "Do that"
            }
        }
        
        enhanced = enhance_prompt_for_task(
            original_prompt,
            task_type=MLTaskType.CLASSIFICATION
        )
        
        # Original fields should be preserved
        assert enhanced["Introduction"] == "You are an ML expert"
        assert enhanced["Custom Field"] == "Custom value"
        assert enhanced["Instructions"]["Step 1"] == "Do this"
        assert enhanced["Instructions"]["Step 2"] == "Do that"
        
        # New fields should be added
        assert "Recommended Libraries" in enhanced["Instructions"]


class TestReviewHints:
    """Test task-specific review hints."""
    
    def test_classification_review_hints(self):
        """Test review hints for classification."""
        hints = get_task_specific_review_hints(MLTaskType.CLASSIFICATION)
        
        assert "classification task" in hints
        assert "best practices" in hints
        assert "avoids" in hints
        assert len(hints.split('\n')) >= 4
    
    def test_time_series_review_hints(self):
        """Test review hints for time series."""
        hints = get_task_specific_review_hints(MLTaskType.TIME_SERIES)
        
        assert "time_series task" in hints
        assert "shuffle" in hints.lower()  # Common time series pitfall
    
    def test_general_task_review_hints(self):
        """Test review hints for general tasks."""
        hints = get_task_specific_review_hints(MLTaskType.GENERAL)
        
        # General task has no specific enhancements
        assert hints == ""


class TestCodeOptimization:
    """Test code optimization functionality."""
    
    def test_optimize_code_placeholder(self):
        """Test that code optimization is currently a placeholder."""
        original_code = "def predict(X): return model.predict(X)"
        
        # Currently just returns the original code
        optimized = optimize_code_for_task(original_code, MLTaskType.CLASSIFICATION)
        assert optimized == original_code


class TestTaskEnhancements:
    """Test the TASK_ENHANCEMENTS configuration."""
    
    def test_all_enhancements_complete(self):
        """Test that all task enhancements have required fields."""
        required_fields = [
            'additional_context',
            'best_practices',
            'common_pitfalls',
            'recommended_libraries',
            'evaluation_hints'
        ]
        
        for task_type, enhancement in TASK_ENHANCEMENTS.items():
            assert enhancement.task_type == task_type
            
            for field in required_fields:
                value = getattr(enhancement, field)
                if isinstance(value, str):
                    assert len(value) > 0
                elif isinstance(value, list):
                    assert len(value) > 0
    
    def test_enhancement_quality(self):
        """Test that enhancements contain substantive content."""
        for task_type, enhancement in TASK_ENHANCEMENTS.items():
            # Each enhancement should have multiple best practices
            assert len(enhancement.best_practices) >= 3
            
            # Each enhancement should warn about pitfalls
            assert len(enhancement.common_pitfalls) >= 3
            
            # Should recommend concrete libraries
            assert len(enhancement.recommended_libraries) >= 2
            
            # Should provide evaluation guidance
            assert len(enhancement.evaluation_hints) >= 2


if __name__ == "__main__":
    pytest.main([__file__])