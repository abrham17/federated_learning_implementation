import tensorflow as tf
import tensorflow_federated as tff
import numpy as np

def create_non_iid_data(num_clients=10, min_examples=20):
    source, _ = tff.simulation.datasets.emnist.load_data()
    client_datasets = []
    class_distributions = []
    
    # Get all client IDs
    all_client_ids = source.client_ids
    
    # Use only clients with sufficient data
    valid_client_ids = []
    for client_id in all_client_ids:
        dataset = source.create_tf_dataset_for_client(client_id)
        if len(list(dataset)) >= min_examples:
            valid_client_ids.append(client_id)
        if len(valid_client_ids) >= num_clients:
            break
    
    # If we didn't find enough clients, use what we have
    if len(valid_client_ids) < num_clients:
        print(f"Warning: Only found {len(valid_client_ids)} clients with sufficient data")
        valid_client_ids = all_client_ids[:num_clients]
    
    for client_id in valid_client_ids:
        dataset = source.create_tf_dataset_for_client(client_id)
        
        # Get actual classes present in the dataset
        all_classes = set()
        for example in dataset.as_numpy_iterator():
            all_classes.add(example['label'])
        all_classes = list(all_classes)
        
        # Select 1-3 random classes from what's actually available
        num_classes = min(3, len(all_classes))
        selected_classes = np.random.choice(all_classes, num_classes, replace=False) if all_classes else []
        
        # Filter dataset to selected classes
        if selected_classes.size > 0:
            dataset = dataset.filter(
                lambda x: tf.reduce_any(tf.equal(x['label'], selected_classes))
            )
        
        # Take a random subset of examples
        dataset_size = len(list(dataset.as_numpy_iterator()))
        take_count = min(dataset_size, min_examples + np.random.randint(0, 50))
        dataset = dataset.take(take_count)
        
        # Preprocess images
        dataset = dataset.map(
            lambda x: (tf.reshape(x['pixels'], (28, 28, 1)), x['label'])
        )
        dataset = dataset.batch(32)
        
        client_datasets.append(dataset)
        class_distributions.append(selected_classes)
    
    return client_datasets, class_distributions
