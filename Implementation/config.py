class Config:
    # Data parameters
    DATA_PATH = './data/'
    SAMPLE_LENGTH = 1024  # Adjust based on your time series length
    NUM_CLASSES = 4
    
    # Scalogram parameters
    SCALES = 32
    WAVELET = 'morl'
    
    # Model parameters
    HIDDEN_DIM = 128
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    
    # Training parameters
    TRAIN_SPLIT = 0.8
    VALIDATION_SPLIT = 0.1
    RANDOM_SEED = 42
