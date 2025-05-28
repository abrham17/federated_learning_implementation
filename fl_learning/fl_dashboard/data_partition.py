import tensorflow as tf
import tensorflow_federated as tff
import numpy as np
"""
def create_non_iid_data(num_clients=10, classes_per_client=3):
    source, _ = tff.simulation.datasets.emnist.load_data()
    client_datasets = []
    data_volumes = np.random.randint(500, 2000, size=num_clients)  # Vary data volume
    class_distributions = [np.random.choice(10, classes_per_client, replace=False) for _ in range(num_clients)]
    
    for client_id in range(num_clients):
        client_data = []
        for c in class_distributions[client_id]:
            client_indices = np.where(np.array([e['label'].numpy() for e in source.create_tf_dataset_for_client(source.client_ids[client_id])]) == c)[0]
            chosen_indices = np.random.choice(client_indices, size=int(data_volumes[client_id] / classes_per_client))
            client_data.extend(chosen_indices)
        dataset = source.create_tf_dataset_for_client(source.client_ids[client_id]).filter(
            lambda e: tf.reduce_any(tf.equal(e['label'], class_distributions[client_id]))
        ).take(data_volumes[client_id]).map(
            lambda e: (tf.reshape(e['pixels'], [-1]), e['label'])
        ).batch(32)
        client_datasets.append(dataset)
    
    return client_datasets, class_distributions, data_volumes

num_clients = 10
client_datasets, class_distributions, data_volumes = create_non_iid_data(num_clients)
print("Class distributions:", class_distributions)
print("Data volumes:", data_volumes)
"""
def client_data(n , source):
  return source.create_tf_dataset_for_client(source.client_ids[n]).map(
      lambda e: (tf.reshape(e['pixels'], [-1]), e['label'])
  ).repeat(10).batch(20)