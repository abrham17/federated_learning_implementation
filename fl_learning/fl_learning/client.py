import tensorflow as tf
import time
import numpy as np

class Client:
    def __init__(self, client_id, dataset, is_straggler=False, is_adversarial=False):
        self.client_id = client_id
        self.dataset = dataset
        self.is_straggler = is_straggler
        self.is_adversarial = is_adversarial
        self.model = self.create_model()

    def create_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, global_weights, epochs=5):
        self.model.set_weights(global_weights)
        if self.is_straggler:
            time.sleep(5)  # Simulate slower computation
        if self.is_adversarial:
            # Label flipping: Swap labels in dataset
            dataset = self.dataset.map(lambda x, y: (x, (y + 1) % 10))  # Flip labels (e.g., 0->1, 9->0)
        else:
            dataset = self.dataset
        history = self.model.fit(dataset, epochs=epochs, verbose=0)
        weights = self.model.get_weights()
        if self.is_adversarial:
            # Add noise for model poisoning
            weights = [w + np.random.normal(0, 0.1, w.shape) for w in weights]
        return weights, history.history['loss'][-1], history.history['accuracy'][-1]

if __name__ == "__main__":
    import sys
    client_id = int(sys.argv[1])
    dataset = ...  # Load dataset (e.g., from data_partition.py)
    client = Client(client_id, dataset, is_straggler=(client_id == 0), is_adversarial=(client_id == 1))
    # Example usage: client.train(global_weights)