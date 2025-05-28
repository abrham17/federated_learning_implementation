import numpy as np
def secure_aggregate(weights_list, num_clients, noise_std=0.01):
    """Secure aggregation with differential privacy"""
    if not weights_list:
        return None
    
    # Clip updates to bound sensitivity
    clipped_weights = []
    for weights in weights_list:
        clipped = [np.clip(w, -1.0, 1.0) for w in weights]
        clipped_weights.append(clipped)
    
    # Average weights
    avg_weights = [np.zeros_like(w) for w in clipped_weights[0]]
    for client_weights in clipped_weights:
        for i, w in enumerate(client_weights):
            avg_weights[i] += w / len(clipped_weights)
    
    # Add Gaussian noise for differential privacy
    noisy_weights = []
    for w in avg_weights:
        noise = np.random.normal(0, noise_std, w.shape)
        noisy_weights.append(w + noise)
    
    return noisy_weights