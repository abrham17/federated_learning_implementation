import tensorflow as tf
import tensorflow_federated as tff
import numpy as np

def create_tff_model():
    keras_model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return tff.learning.models.from_keras_model(
        keras_model,
        input_spec=(tf.TensorSpec((None, 784), tf.float32), tf.TensorSpec((None,), tf.int32)),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

def get_fl_process(strategy='fedavg', mu=0.1):
    if strategy == 'fedavg':
        return tff.learning.algorithms.build_weighted_fed_avg(
            create_tff_model,
            client_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.1),
            server_optimizer_fn=lambda: tf.keras.optimizers.SGD(1.0)
        )
    elif strategy == 'fedprox':
        return tff.learning.algorithms.build_weighted_fed_avg(
            create_tff_model,
            client_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.1),
            server_optimizer_fn=lambda: tf.keras.optimizers.SGD(1.0)
        )
    else:
        raise ValueError("Unsupported strategy")

def secure_aggregate(weights_list, num_clients, noise_std=0.01):
    noisy_weights = [
        [w + np.random.normal(0, noise_std, w.shape) for w in weights]
        for weights in weights_list
    ]
    aggregated_weights = [
        np.mean([w[i] for w in noisy_weights], axis=0)
        for i in range(len(noisy_weights[0]))
    ]
    return aggregated_weights

class Server:
    def __init__(self, num_clients, strategy='fedavg'):
        self.num_clients = num_clients
        self.process = get_fl_process(strategy)
        self.state = self.process.initialize()
        self.global_weights = None

    def run_round(self, client_datasets, selected_clients):
        weights_list = []
        for client_id in selected_clients:
            try:
                client = Client(client_id, client_datasets[client_id], is_straggler=(client_id == 0), is_adversarial=(client_id == 1))
                weights, _, _ = client.train(self.global_weights)
                weights_list.append(weights)
            except Exception as e:
                print(f"Client {client_id} failed: {e}")
        if weights_list:
            self.global_weights = secure_aggregate(weights_list, len(selected_clients))
        return self.global_weights