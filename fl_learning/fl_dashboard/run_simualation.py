from .client import Client
from .server import secure_aggregate
from .data_partition import create_non_iid_data
import threading
import tensorflow_federated as tff
import tensorflow as tf
import logging
import os
import random
from .models import Experiment, RoundLog, ClientLog
# Configuration
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.basicConfig(filename='fl_simulation.log', level=logging.INFO)
NUM_CLIENTS = 10
ROUNDS = 10
PARTICIPATION_RATE = 0.8

def run_simulation():
    global client_datasets, metrics, current_experiment
    client_datasets, class_distributions = create_non_iid_data(NUM_CLIENTS)
    clients = [
        Client(
            client_id=i,
            dataset=client_datasets[i],
            is_straggler=(i == 0),  
            is_adversarial=(i == 1) 
        )
        for i in range(NUM_CLIENTS)
    ]
    
    # Initialize global model
    global_model = clients[0].create_model()
    global_weights = global_model.get_weights()
    
    # Create experiment record
    experiment = Experiment.objects.create(
        name="Federated Learning Simulation",
        status='running',
        rounds=ROUNDS
    )
    current_experiment = experiment
    
    metrics = []
    for round_num in range(ROUNDS):
        selected_indices = random.sample(range(NUM_CLIENTS), int(PARTICIPATION_RATE * NUM_CLIENTS))
        selected_clients = [clients[i] for i in selected_indices]
        
        threads = []
        results = [None] * len(selected_clients)
        for idx, client in enumerate(selected_clients):
            thread = threading.Thread(
                target=train_client,
                args=(client, global_weights, round_num, results, idx)
            )
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join(timeout=15.0)
        
        updates = []
        client_logs = []
        for i, result in enumerate(results):
            if result and result[0] is not None:  
                weights, loss, accuracy, status = result
                updates.append(weights)
                
                client_log = ClientLog(
                    experiment=experiment,
                    client_id=selected_clients[i].client_id,
                    round_number=round_num,
                    status=status,
                    loss=loss,
                    accuracy=accuracy,
                    is_adversarial=selected_clients[i].is_adversarial,
                    is_straggler=selected_clients[i].is_straggler
                )
                client_log.save()
                client_logs.append(client_log)
        
        if updates:
            new_weights = secure_aggregate(updates, len(selected_clients))
            if new_weights:
                global_weights = new_weights
        
        # Evaluate global model
        global_model.set_weights(global_weights)
        test_loss, test_acc = evaluate_global_model(global_model)
        
        # Save round metrics
        round_log = RoundLog(
            experiment=experiment,
            round_number=round_num,
            accuracy=test_acc,
            loss=test_loss,
            clients_participated=len(updates)
        )
        round_log.save()
        
        metrics.append({
            'round': round_num,
            'accuracy': test_acc,
            'loss': test_loss,
            'participants': len(updates)
        })
        
        logging.info(f"Round {round_num}: Accuracy={test_acc:.4f}, Loss={test_loss:.4f}")
    
    # Finalize experiment
    experiment.status = 'completed'
    experiment.save()
    current_experiment = None

def train_client(client, global_weights, round_num, results, idx):
    try:
        result = client.train(global_weights, round_num)
        results[idx] = result
    except Exception as e:
        logging.error(f"Client {client.client_id} failed: {str(e)}")
        results[idx] = (None, None, None, "failed")

# simulation.py (evaluate_global_model function)
def evaluate_global_model(model):
    """Evaluate global model on test data"""
    _, test_data = tff.simulation.datasets.emnist.load_data()
    test_dataset = test_data.create_tf_dataset_from_all_clients()
    
    # Preprocess images to match model input requirements
    def preprocess(element):
        # Reshape to (28, 28, 1) without adding extra dimension
        return (tf.reshape(element['pixels'], (28, 28, 1)), element['label'])
    
    test_dataset = test_dataset.map(preprocess)
    test_dataset = test_dataset.batch(32)
    
    loss, acc = model.evaluate(test_dataset, verbose=0)
    return loss, acc
