import os
import json
import random
import collections
from django.shortcuts import render
from django.http import JsonResponse
import tensorflow as tf
import tensorflow_federated as tff

from .data_partition import client_data

# Optional: Avoid oneDNN optimization warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# To store simulation metrics
metricss = []

def start_simulation(request):
    tff.backends.native.set_sync_local_cpp_execution_context()
    num_clients = 10

    # Load EMNIST data and partition it per client
    source, _ = tff.simulation.datasets.emnist.load_data()
    train_data = [client_data(n, source) for n in range(num_clients)]

    # Define the Keras model with 62 output units for EMNIST 'byclass'
    def model_fn():
        keras_model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(62, activation='softmax', input_shape=(784,))
        ])
        return tff.learning.models.from_keras_model(
            keras_model,
            input_spec=train_data[0].element_spec,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
        )

    # Build Federated Averaging algorithm using high-level API
    trainer = tff.learning.algorithms.build_weighted_fed_avg(
        model_fn=model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.1),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0)
    )

    # Initialize the training state
    state = trainer.initialize()

    # Simulate federated training rounds
    for round_num in range(5):
        # Randomly sample 70% of clients
        selected_data = random.sample(train_data, int(0.7 * num_clients))
        result = trainer.next(state, selected_data)
        state = result.state
        metrics = result.metrics

        # Collect metrics for dashboard
        metricss.append({
            'round': round_num + 1,
            'accuracy': float(metrics['client_work']['train']['sparse_categorical_accuracy']),
            'loss': float(metrics['client_work']['train']['loss']),
        })

    # Save metrics to JSON file
    with open('metrics.json', 'w') as f:
        json.dump(metricss, f)

    return JsonResponse({'status': 'Simulation completed', 'metrics_file': 'metrics.json'})

def dashboard(request):
    with open('metrics.json', 'r') as f:
        metrics = json.load(f)
    return render(request, 'dashboard.html', {'metrics': json.dumps(metrics)})