import numpy as np

def dropout(x: np.ndarray, p: float, rng=None) -> tuple[np.ndarray, np.ndarray]:
    # Ensure input is a numpy array to access .shape
    x = np.asarray(x) 
    
    if p >= 1.0:
        return np.zeros_like(x), np.zeros_like(x)
    
    # Use rng.random() for Generator objects, or np.random.random() for the legacy API
    random_values = rng.random(x.shape) if rng else np.random.random(x.shape)
    
    keep_mask = random_values < (1 - p)
    scale = 1 / (1 - p)
    
    # Create the pattern and apply scaling
    dropout_pattern = np.where(keep_mask, scale, 0.0)
    return x * dropout_pattern, dropout_pattern
