import numpy as np
import random
import tensorflow as tf 
import time
import logging

class Client:
    def __init__(self, client_id, dataset, is_straggler=False, is_adversarial=False):
        self.client_id = client_id
        self.dataset = dataset
        self.is_straggler = is_straggler
        self.is_adversarial = is_adversarial
        self.model = self.create_model()
        
        # Validate dataset isn't empty
        self.has_data = self.validate_dataset()
        
    def validate_dataset(self):
        """Check if dataset has at least one batch"""
        try:
            next(iter(self.dataset))
            return True
        except StopIteration:
            logging.warning(f"Client {self.client_id} has empty dataset!")
            return False

    def create_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
            tf.keras.layers.Dense(62, activation='softmax')  # 62 classes for EMNIST
        ])
        model.compile(
            optimizer='sgd',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
            run_eagerly=True  # For better debugging
        )
        return model

    def train(self, global_weights, epochs=5):
        if not self.has_data:
            return None, None, None, "no_data"
        
        self.model.set_weights(global_weights)
        
        # Straggler simulation
        if self.is_straggler:
            time.sleep(random.uniform(1.0, 5.0))
        
        # Adversarial behavior
        if self.is_adversarial:
            # Label flipping: Swap labels in dataset
            dataset = self.dataset.map(lambda x, y: (x, (y + 31) % 62))  # EMNIST-specific (62 classes)
        else:
            dataset = self.dataset
        
        try:
            history = self.model.fit(
                dataset,
                epochs=epochs,
                verbose=0,
            ) # Ensure at least one step
        except Exception as e:
            logging.error(f"Client {self.client_id} training failed: {str(e)}")
            return None, None, None, "failed"
        
        weights = self.model.get_weights()
        return weights, history.history['loss'][-1], history.history['accuracy'][-1], "completed"
