import tensorflow as tf
import tensorflow_federated as tff
import numpy as np
def create_non_iid_data(num_clients=10, classes_per_client=3, min_examples=50):
    source, _ = tff.simulation.datasets.emnist.load_data()
    client_datasets = []
    class_distributions = []
    
    # Get all available classes
    all_classes = np.arange(62)
    
    for client_id in range(num_clients):
        # Ensure we select classes that actually exist in the client's data
        valid_classes = []
        attempts = 0
        
        while len(valid_classes) < classes_per_client and attempts < 5:
            candidate_classes = np.random.choice(all_classes, classes_per_client, replace=False)
            dataset = source.create_tf_dataset_for_client(source.client_ids[client_id])
            dataset = dataset.filter(lambda x: tf.reduce_any(tf.equal(x['label'], candidate_classes)))
            
            # Check if dataset has enough examples
            if len(list(dataset.as_numpy_iterator())) >= min_examples:
                valid_classes = candidate_classes
                break
            attempts += 1
        
        # Fallback: use all classes if still no valid classes found
        if not valid_classes:
            valid_classes = all_classes
        
        # Create final dataset
        dataset = source.create_tf_dataset_for_client(source.client_ids[client_id])
        dataset = dataset.filter(lambda x: tf.reduce_any(tf.equal(x['label'], valid_classes)))
        dataset = dataset.take(min_examples + np.random.randint(0, 100))  # Ensure minimum + random extra
        dataset = dataset.map(lambda x: (tf.reshape(x['pixels'], [-1]), x['label']))
        dataset = dataset.batch(32)
        
        client_datasets.append(dataset)
        class_distributions.append(valid_classes)
    
    return client_datasets, class_distributions