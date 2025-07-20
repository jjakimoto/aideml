"""Specialized prompts optimized for different ML task types.

This module provides task-specific prompt templates and enhancements
to improve the quality of generated code for different machine learning tasks.
"""

import re
from typing import Dict, Optional, List, Any
from dataclasses import dataclass
from enum import Enum


class MLTaskType(Enum):
    """Enumeration of machine learning task types."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    TIME_SERIES = "time_series"
    NLP = "nlp"
    COMPUTER_VISION = "computer_vision"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    GENERAL = "general"


@dataclass
class TaskPromptEnhancement:
    """Container for task-specific prompt enhancements."""
    task_type: MLTaskType
    additional_context: str
    best_practices: List[str]
    common_pitfalls: List[str]
    recommended_libraries: List[str]
    evaluation_hints: List[str]


# Task-specific prompt enhancements
TASK_ENHANCEMENTS: Dict[MLTaskType, TaskPromptEnhancement] = {
    MLTaskType.CLASSIFICATION: TaskPromptEnhancement(
        task_type=MLTaskType.CLASSIFICATION,
        additional_context="""
This is a classification task. Focus on:
- Class balance and stratification
- Appropriate metrics (accuracy, precision, recall, F1, AUC-ROC)
- Feature engineering for discriminative features
- Handling class imbalance if present
""",
        best_practices=[
            "Check class distribution and handle imbalance",
            "Use stratified train-test splits",
            "Consider ensemble methods for better performance",
            "Implement proper cross-validation",
            "Use appropriate evaluation metrics for the problem"
        ],
        common_pitfalls=[
            "Not handling class imbalance",
            "Using accuracy on imbalanced datasets",
            "Data leakage in preprocessing",
            "Not using stratification in splits"
        ],
        recommended_libraries=[
            "scikit-learn",
            "xgboost",
            "lightgbm",
            "imbalanced-learn"
        ],
        evaluation_hints=[
            "Use classification_report for detailed metrics",
            "Plot confusion matrix",
            "Consider ROC curves for binary classification",
            "Check precision-recall curves for imbalanced data"
        ]
    ),
    
    MLTaskType.REGRESSION: TaskPromptEnhancement(
        task_type=MLTaskType.REGRESSION,
        additional_context="""
This is a regression task. Focus on:
- Feature scaling and normalization
- Handling outliers and extreme values
- Appropriate metrics (MSE, RMSE, MAE, R²)
- Residual analysis
- Feature engineering for linear and non-linear relationships
""",
        best_practices=[
            "Explore data distribution and outliers",
            "Consider feature transformations (log, polynomial)",
            "Use regularization to prevent overfitting",
            "Validate assumptions for linear models",
            "Try both linear and non-linear models"
        ],
        common_pitfalls=[
            "Not scaling features",
            "Ignoring outliers impact",
            "Not checking residual patterns",
            "Overfitting with complex models"
        ],
        recommended_libraries=[
            "scikit-learn",
            "xgboost",
            "lightgbm",
            "statsmodels"
        ],
        evaluation_hints=[
            "Plot predicted vs actual values",
            "Analyze residual distributions",
            "Check for heteroscedasticity",
            "Use multiple error metrics"
        ]
    ),
    
    MLTaskType.TIME_SERIES: TaskPromptEnhancement(
        task_type=MLTaskType.TIME_SERIES,
        additional_context="""
This is a time series task. Focus on:
- Temporal dependencies and seasonality
- Proper train-test split (no shuffling!)
- Stationarity and trend analysis
- Lag features and rolling statistics
- Forecasting horizon considerations
""",
        best_practices=[
            "Never shuffle time series data",
            "Use walk-forward validation",
            "Check for stationarity",
            "Create lag features and rolling statistics",
            "Consider seasonal decomposition"
        ],
        common_pitfalls=[
            "Shuffling time series data",
            "Using future information (data leakage)",
            "Ignoring seasonality",
            "Not handling missing timestamps"
        ],
        recommended_libraries=[
            "pandas",
            "statsmodels",
            "prophet",
            "sktime",
            "tslearn"
        ],
        evaluation_hints=[
            "Use time-based validation splits",
            "Plot forecasts vs actuals over time",
            "Check multiple forecast horizons",
            "Consider business-specific metrics"
        ]
    ),
    
    MLTaskType.NLP: TaskPromptEnhancement(
        task_type=MLTaskType.NLP,
        additional_context="""
This is a Natural Language Processing task. Focus on:
- Text preprocessing (tokenization, cleaning)
- Feature extraction (TF-IDF, word embeddings)
- Handling vocabulary size
- Sequence length considerations
- Language-specific preprocessing
""",
        best_practices=[
            "Clean text appropriately for the task",
            "Consider both traditional and deep learning approaches",
            "Use pre-trained embeddings when available",
            "Handle out-of-vocabulary words",
            "Consider text augmentation techniques"
        ],
        common_pitfalls=[
            "Over-preprocessing (removing useful information)",
            "Not handling different text encodings",
            "Ignoring class imbalance in text classification",
            "Not considering computational constraints"
        ],
        recommended_libraries=[
            "scikit-learn",
            "spacy",
            "nltk",
            "transformers",
            "gensim"
        ],
        evaluation_hints=[
            "Use appropriate metrics for the task",
            "Consider both quantitative and qualitative evaluation",
            "Check model performance on different text lengths",
            "Analyze misclassified examples"
        ]
    ),
    
    MLTaskType.COMPUTER_VISION: TaskPromptEnhancement(
        task_type=MLTaskType.COMPUTER_VISION,
        additional_context="""
This is a Computer Vision task. Focus on:
- Image preprocessing and augmentation
- Input size and aspect ratio handling
- Transfer learning opportunities
- Computational efficiency
- Data loading optimization
""",
        best_practices=[
            "Use data augmentation appropriately",
            "Consider transfer learning from pre-trained models",
            "Normalize pixel values",
            "Handle different image sizes properly",
            "Optimize data loading for performance"
        ],
        common_pitfalls=[
            "Not normalizing images",
            "Inappropriate augmentation for the task",
            "Memory issues with large images",
            "Not using pre-trained models"
        ],
        recommended_libraries=[
            "opencv-python",
            "PIL/Pillow",
            "scikit-image",
            "albumentations",
            "torchvision",
            "tensorflow"
        ],
        evaluation_hints=[
            "Visualize predictions on sample images",
            "Check performance across different image types",
            "Consider computational cost",
            "Analyze failure cases visually"
        ]
    )
}


def detect_ml_task_type(task_description: str, data_preview: Optional[str] = None) -> MLTaskType:
    """Detect the ML task type from task description and data preview.
    
    Args:
        task_description: The task description text
        data_preview: Optional data preview information
        
    Returns:
        Detected MLTaskType
    """
    combined_text = task_description.lower()
    if data_preview:
        combined_text += " " + data_preview.lower()
    
    # Classification indicators
    classification_keywords = [
        "classify", "classification", "predict class", "predict category",
        "binary", "multiclass", "multi-class", "label", "categorize"
    ]
    
    # Regression indicators
    regression_keywords = [
        "regression", "predict value", "predict price", "forecast value",
        "continuous", "numeric prediction", "estimate"
    ]
    
    # Time series indicators
    time_series_keywords = [
        "time series", "forecast", "temporal", "time-based",
        "historical", "future prediction", "trend", "seasonality"
    ]
    
    # NLP indicators
    nlp_keywords = [
        "text", "nlp", "natural language", "sentiment", "document",
        "tokenize", "word", "sentence", "language", "corpus"
    ]
    
    # Computer vision indicators
    cv_keywords = [
        "image", "vision", "visual", "picture", "photo",
        "pixel", "opencv", "cnn", "convolutional"
    ]
    
    # Count keyword matches
    scores = {
        MLTaskType.CLASSIFICATION: sum(1 for kw in classification_keywords if kw in combined_text),
        MLTaskType.REGRESSION: sum(1 for kw in regression_keywords if kw in combined_text),
        MLTaskType.TIME_SERIES: sum(1 for kw in time_series_keywords if kw in combined_text),
        MLTaskType.NLP: sum(1 for kw in nlp_keywords if kw in combined_text),
        MLTaskType.COMPUTER_VISION: sum(1 for kw in cv_keywords if kw in combined_text),
    }
    
    # Return the task type with highest score, defaulting to GENERAL
    max_score = max(scores.values())
    if max_score > 0:
        return max(scores.items(), key=lambda x: x[1])[0]
    
    return MLTaskType.GENERAL


def enhance_prompt_for_task(prompt: Dict[str, Any], 
                           task_type: Optional[MLTaskType] = None,
                           task_description: Optional[str] = None,
                           data_preview: Optional[str] = None) -> Dict[str, Any]:
    """Enhance a prompt with task-specific optimizations.
    
    Args:
        prompt: Original prompt dictionary
        task_type: Optional explicit task type
        task_description: Task description for auto-detection
        data_preview: Data preview for auto-detection
        
    Returns:
        Enhanced prompt dictionary
    """
    # Auto-detect task type if not provided
    if task_type is None:
        if task_description:
            task_type = detect_ml_task_type(task_description, data_preview)
        else:
            task_type = MLTaskType.GENERAL
    
    # Get task-specific enhancements
    enhancement = TASK_ENHANCEMENTS.get(task_type)
    if not enhancement:
        return prompt  # Return original if no enhancement available
    
    # Create enhanced prompt
    enhanced_prompt = prompt.copy()
    
    # Add task-specific context
    if "Task Description" in enhanced_prompt:
        enhanced_prompt["Task Description"] = (
            enhanced_prompt["Task Description"] + 
            "\n\n" + enhancement.additional_context.strip()
        )
    
    # Add best practices section
    best_practices_text = "\n".join(f"- {bp}" for bp in enhancement.best_practices)
    enhanced_prompt["ML Best Practices"] = f"Follow these {task_type.value} best practices:\n{best_practices_text}"
    
    # Add common pitfalls warning
    pitfalls_text = "\n".join(f"- {cp}" for cp in enhancement.common_pitfalls)
    enhanced_prompt["Common Pitfalls to Avoid"] = f"Avoid these common {task_type.value} mistakes:\n{pitfalls_text}"
    
    # Add recommended libraries
    libs_text = ", ".join(enhancement.recommended_libraries)
    if "Instructions" in enhanced_prompt:
        enhanced_prompt["Instructions"] = (
            enhanced_prompt.get("Instructions", {}) | 
            {"Recommended Libraries": f"Consider using: {libs_text}"}
        )
    
    # Add evaluation hints
    eval_hints = "\n".join(f"- {eh}" for eh in enhancement.evaluation_hints)
    enhanced_prompt["Evaluation Guidelines"] = f"For evaluation:\n{eval_hints}"
    
    return enhanced_prompt


def get_task_specific_review_hints(task_type: MLTaskType) -> str:
    """Get task-specific hints for code review.
    
    Args:
        task_type: The ML task type
        
    Returns:
        Task-specific review hints as a string
    """
    enhancement = TASK_ENHANCEMENTS.get(task_type)
    if not enhancement:
        return ""
    
    hints = [
        f"This appears to be a {task_type.value} task.",
        f"Check if the code follows these best practices: {', '.join(enhancement.best_practices[:3])}",
        f"Verify it avoids: {', '.join(enhancement.common_pitfalls[:3])}",
        f"Expected evaluation approach: {', '.join(enhancement.evaluation_hints[:2])}"
    ]
    
    return "\n".join(hints)


def optimize_code_for_task(code: str, task_type: MLTaskType) -> str:
    """Apply task-specific optimizations to generated code.
    
    This is a post-processing step that can add task-specific
    improvements to already generated code.
    
    Args:
        code: Generated code
        task_type: The ML task type
        
    Returns:
        Optimized code
    """
    # This is a placeholder for future enhancements
    # Could add task-specific code templates, imports, etc.
    return code