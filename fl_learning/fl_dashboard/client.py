import random
import time
import logging
import tensorflow as tf
from tensorflow.keras import Model, layers

class Client:
    def __init__(self, client_id, dataset, is_straggler=False, is_adversarial=False):
        self.client_id = client_id
        self.dataset = dataset
        self.is_straggler = is_straggler
        self.is_adversarial = is_adversarial
        self.model = self.create_model()
        self.has_data = self.validate_dataset()
        
    def validate_dataset(self):
        try:
            # Try to get one batch
            for batch in self.dataset.take(1):
                if batch[0].shape[0] > 0:  # Check batch has at least 1 example
                    return True
            return False
        except Exception as e:
            logging.warning(f"Validation failed for client {self.client_id}: {str(e)}")
            return False

    def create_model(self):
        inputs = tf.keras.Input(shape=(28, 28, 1))
        x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)
        outputs = layers.Dense(62, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
            run_eagerly=True
        )
        return model

    def train(self, global_weights, round_num, epochs=5):
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
            )
            final_loss = history.history['loss'][-1]
            final_accuracy = history.history['accuracy'][-1]
            return self.model.get_weights(), final_loss, final_accuracy, "completed"
        except Exception as e:
            logging.error(f"Client {self.client_id} training failed: {str(e)}")
            return None, None, None, "failed"
