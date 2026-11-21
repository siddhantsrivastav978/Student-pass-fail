"""Dataset presets. Pick one with --dataset when running the scripts."""

DATASETS = {
    # ~76,000 students. Kaggle Playground Series S4E6 (Academic Success).
    # Target: Graduate / Dropout / Enrolled -> we use Graduate = pass, Dropout = fail.
    # You must click "Join Competition" on Kaggle once before downloading.
    "academic_success": {
        "kaggle_type": "competition",
        "kaggle_ref": "playground-series-s4e6",
        "file": "train.csv",
        "target": "Target",
        "pass_values": ["Graduate"],
        "fail_values": ["Dropout"],   # "Enrolled" rows are dropped (result not known yet)
        "drop_columns": ["id"],
    },
    # ~6,600 students. Exam score plus study habits, attendance, etc.
    "performance_factors": {
        "kaggle_type": "dataset",
        "kaggle_ref": "lainguyn123/student-performance-factors",
        "file": "StudentPerformanceFactors.csv",
        "target": "Exam_Score",
        "pass_mark": 65,              # score >= pass_mark counts as pass. Change if needed.
        "drop_columns": [],
    },
}

DATA_DIR = "data"
MODEL_DIR = "models"
RANDOM_STATE = 42
