import tensorflow as tf
import tensorflow_federated as tff
import numpy as np
import random
import threading
import logging
from server import Server
from data_partition import create_non_iid_data

logging.basicConfig(filename='fl_simulation.log', level=logging.INFO)

def run_client(client_id, dataset, global_weights, result_queue):
    try:
        client = Client(client_id, dataset, is_straggler=(client_id == 0), is_adversarial=(client_id == 1))
        weights, loss, accuracy = client.train(global_weights)
        result_queue.append((weights, loss, accuracy))
        logging.info(f"Client {client_id}: Loss={loss:.4f}, Accuracy={accuracy:.4f}")
    except Exception as e:
        logging.error(f"Client {client_id} failed: {e}")

def main():
    num_clients = 10
    client_datasets, _, _ = create_non_iid_data(num_clients)
    server = Server(num_clients)
    metrics = []
    
    for round_num in range(50):
        selected_clients = random.sample(range(num_clients), int(0.7 * num_clients))
        result_queue = []
        threads = []
        
        for client_id in selected_clients:
            thread = threading.Thread(target=run_client, args=(client_id, client_datasets[client_id], server.global_weights, result_queue))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join(timeout=10)  # Timeout for stragglers
        
        if result_queue:
            server.global_weights = secure_aggregate([r[0] for r in result_queue], len(selected_clients))
            avg_loss = np.mean([r[1] for r in result_queue])
            avg_accuracy = np.mean([r[2] for r in result_queue])
            metrics.append({'round': round_num + 1, 'loss': avg_loss, 'accuracy': avg_accuracy})
            logging.info(f"Round {round_num + 1}: Loss={avg_loss:.4f}, Accuracy={avg_accuracy:.4f}")
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot([m['round'] for m in metrics], [m['loss'] for m in metrics], label='Loss')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.title('Loss Over Rounds')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot([m['round'] for m in metrics], [m['accuracy'] for m in metrics], label='Accuracy')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Rounds')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('convergence.png')

if __name__ == "__main__":
    main()