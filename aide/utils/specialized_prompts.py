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
    if not code or task_type == MLTaskType.GENERAL:
        return code
    
    # Apply task-specific optimizations
    if task_type == MLTaskType.CLASSIFICATION:
        return _optimize_classification_code(code)
    elif task_type == MLTaskType.REGRESSION:
        return _optimize_regression_code(code)
    elif task_type == MLTaskType.TIME_SERIES:
        return _optimize_time_series_code(code)
    elif task_type == MLTaskType.NLP:
        return _optimize_nlp_code(code)
    elif task_type == MLTaskType.COMPUTER_VISION:
        return _optimize_computer_vision_code(code)
    else:
        return code


def _optimize_classification_code(code: str) -> str:
    """Optimize code for classification tasks."""
    import re
    
    # Check if stratified split is missing
    if "train_test_split" in code and "stratify=" not in code:
        # Find train_test_split calls and add stratify parameter
        pattern = r'train_test_split\((.*?)\)'
        def add_stratify(match):
            args = match.group(1)
            # Simple heuristic: if y is mentioned, add stratify=y
            if ', y' in args or ',y' in args:
                if 'test_size' in args:
                    return f'train_test_split({args}, stratify=y)'
                else:
                    return f'train_test_split({args}, test_size=0.2, stratify=y)'
            return match.group(0)
        code = re.sub(pattern, add_stratify, code, flags=re.DOTALL)
    
    # Add classification metrics if missing
    if "accuracy" in code.lower() and "classification_report" not in code:
        # Find where predictions are made
        if "model.score" in code or "accuracy_score" in code:
            # Add imports
            if "from sklearn.metrics import" not in code:
                import_line = "from sklearn.metrics import classification_report, confusion_matrix\n"
                # Add after other sklearn imports
                if "from sklearn" in code:
                    code = re.sub(r'(from sklearn.*\n)', r'\1' + import_line, code, count=1)
                else:
                    code = import_line + code
            
            # Add metrics calculation after evaluation
            metrics_code = """
# Detailed metrics
y_pred = model.predict(X_test)
print("\\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
"""
            # Add after accuracy calculation
            code = re.sub(r'(print.*[Aa]ccuracy.*\n)', r'\1' + metrics_code, code)
    
    # Check for class imbalance handling
    if "RandomForestClassifier()" in code and "class_weight" not in code:
        code = code.replace("RandomForestClassifier()", 
                          "RandomForestClassifier(class_weight='balanced')")
    
    # Add class distribution check if missing
    if "value_counts()" not in code and "y" in code:
        check_code = "\n# Check class distribution\nprint('Class distribution:')\nprint(y.value_counts())\n"
        # Add after y is defined
        code = re.sub(r'(y = .*\n)', r'\1' + check_code, code, count=1)
    
    return code


def _optimize_regression_code(code: str) -> str:
    """Optimize code for regression tasks."""
    import re
    
    # Add feature scaling if missing
    if "fit" in code and "StandardScaler" not in code and "MinMaxScaler" not in code:
        # Add scaler import
        if "from sklearn.preprocessing import" not in code:
            import_line = "from sklearn.preprocessing import StandardScaler\n"
            if "from sklearn" in code:
                code = re.sub(r'(from sklearn.*\n)', r'\1' + import_line, code, count=1)
            else:
                code = import_line + code
        
        # Add scaling before train_test_split
        if "train_test_split" in code:
            scaling_code = """
# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

"""
            # Replace X with X_scaled in train_test_split
            code = re.sub(r'(X = .*\n)(.*?)(train_test_split\(X,)', 
                         r'\1' + scaling_code + r'\2' + r'train_test_split(X_scaled,', 
                         code, flags=re.DOTALL)
    
    # Add more regression metrics
    if "score" in code and "mean_squared_error" not in code:
        # Add imports
        if "from sklearn.metrics import mean_squared_error" not in code:
            import_line = "from sklearn.metrics import mean_squared_error, mean_absolute_error\n"
            if "from sklearn" in code:
                code = re.sub(r'(from sklearn.*\n)', r'\1' + import_line, code, count=1)
            else:
                code = import_line + "\n" + code
        
        # Add metrics
        metrics_code = """
# Additional metrics
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mse ** 0.5
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

# Residual analysis
residuals = y_test - y_pred
print(f"\\nResiduals mean: {residuals.mean():.4f}")
print(f"Residuals std: {residuals.std():.4f}")
"""
        # Add after score calculation
        code = re.sub(r'(print.*[Ss]core.*\n)', r'\1' + metrics_code, code)
    
    return code


def _optimize_time_series_code(code: str) -> str:
    """Optimize code for time series tasks."""
    import re
    
    # Fix train_test_split for time series
    if "train_test_split" in code and "shuffle=False" not in code:
        # Replace train_test_split with time series appropriate split
        code = re.sub(r'train_test_split\((.*?)\)', 
                     r'train_test_split(\1, shuffle=False)', 
                     code, flags=re.DOTALL)
    
    # Add lag features if missing
    if "shift(" not in code and "lag" not in code and "df" in code:
        lag_code = """
# Create lag features
for lag in [1, 7, 30]:
    df[f'target_lag_{lag}'] = df['target'].shift(lag)

# Drop rows with NaN values from lag features
df = df.dropna()

"""
        # Add after dataframe is loaded
        code = re.sub(r'(df = pd\.read_csv.*\n)', r'\1' + lag_code, code)
    
    # Ensure temporal order is preserved
    if "date" in code.lower() and "sort" not in code:
        sort_code = "# Ensure data is sorted by date\ndf = df.sort_values('date')\n"
        code = re.sub(r'(df\[.date.\] = pd\.to_datetime.*\n)', r'\1' + sort_code, code)
    
    return code


def _optimize_nlp_code(code: str) -> str:
    """Optimize code for NLP tasks."""
    import re
    
    # Add text preprocessing if missing
    if "CountVectorizer" in code or "TfidfVectorizer" in code:
        if "lower()" not in code and ".str." not in code:
            preprocess_code = """
# Preprocess text
X = X.str.lower().str.replace('[^a-zA-Z0-9\\s]', '', regex=True)

"""
            code = re.sub(r'(X = df\[.*\]\n)', r'\1' + preprocess_code, code)
    
    # Suggest TF-IDF if using CountVectorizer
    if "CountVectorizer" in code and "TfidfVectorizer" not in code:
        code = code.replace("CountVectorizer", "TfidfVectorizer")
        code = code.replace("from sklearn.feature_extraction.text import CountVectorizer",
                          "from sklearn.feature_extraction.text import TfidfVectorizer")
    
    # Add train-test split if missing
    if "fit_transform" in code and "train_test_split" not in code:
        split_import = "from sklearn.model_selection import train_test_split\n"
        if "from sklearn" in code:
            code = re.sub(r'(from sklearn.*\n)', r'\1' + split_import, code, count=1)
        
        # Add split after vectorization
        split_code = """
# Split data
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

"""
        code = re.sub(r'(X_vec = vectorizer\.fit_transform.*\n)', r'\1' + split_code, code)
        
        # Update model fitting to use split data
        code = code.replace("model.fit(X_vec, y)", "model.fit(X_train, y_train)")
    
    return code


def _optimize_computer_vision_code(code: str) -> str:
    """Optimize code for computer vision tasks."""
    import re
    
    # Add data augmentation if missing
    if "Sequential" in code and "ImageDataGenerator" not in code and "augmentation" not in code:
        aug_import = "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n"
        code = aug_import + code
        
        aug_code = """
# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    rescale=1./255
)

"""
        # Add before model definition
        code = re.sub(r'(# Build model\n)', aug_code + r'\1', code)
    
    # Add normalization if missing
    if "/255" not in code and "Rescaling" not in code and "normalize" not in code:
        if "layers.Conv2D" in code:
            # Add Rescaling layer
            code = code.replace(
                "model = models.Sequential([",
                "model = models.Sequential([\n    layers.Rescaling(1./255, input_shape=(64, 64, 3)),"
            )
            # Update Conv2D to not have input_shape
            code = re.sub(r'layers\.Conv2D\((.*?), input_shape=\(64, 64, 3\)\)', 
                         r'layers.Conv2D(\1)', code)
    
    # Add regularization if missing
    if "Dropout" not in code and "Dense" in code:
        # Add dropout after flatten
        code = re.sub(r'(layers\.Flatten\(\),\n)', 
                     r'\1    layers.Dropout(0.5),\n', code)
    
    return code