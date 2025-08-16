# Configuration file for Baby Cry Analyzer

# Audio processing settings
SAMPLE_RATE = 22050  # Hz
AUDIO_DURATION = 3.0  # seconds
MAX_AUDIO_LENGTH = 5.0  # maximum audio length to process

# Model settings
DEFAULT_MODEL_TYPE = 'random_forest'
MODEL_PATH = 'models/baby_cry_rf_model.pkl'
BACKUP_MODEL_PATH = 'models/baby_cry_svm_model.pkl'

# API settings
API_HOST = '0.0.0.0'
API_PORT = 5000
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16 MB
ALLOWED_EXTENSIONS = ['wav']

# Dataset settings
DATASET_PATH = 'dataset'
CATEGORIES = ['hungry', 'pain', 'need_to_change', 'tired', 'attention']

# Feature extraction settings
N_MFCC = 13
N_CHROMA = 12
N_MEL = 128

# Training settings
VALIDATION_SPLIT = 0.2
RANDOM_STATE = 42
CROSS_VALIDATION_FOLDS = 5

# File paths
UPLOAD_FOLDER = 'uploads'
MODELS_FOLDER = 'models'
REPORTS_FOLDER = 'reports'
LOGS_FOLDER = 'logs'
