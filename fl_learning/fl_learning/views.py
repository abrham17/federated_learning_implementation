import django
from django.shortcuts import render
from django.http import JsonResponse
import subprocess
import json
import random
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN

from .server import Server
from .data_partition import create_non_iid_data

def start_simulation(request):
    num_clients = 10
    client_datasets, _, _ = create_non_iid_data(num_clients)
    server = Server(num_clients)
    metrics = []
    
    for round_num in range(5):
        selected_clients = random.sample(range(num_clients), int(0.7 * num_clients))  # 70% participation
        global_weights = server.run_round(client_datasets, selected_clients)
        # Simulate evaluation (placeholder)
        metrics.append({
            'round': round_num + 1,
            'accuracy': random.uniform(0.5, 0.9),  # Replace with real metrics
            'loss': random.uniform(0.1, 0.5)
        })
    
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f)
    return JsonResponse({'status': 'Simulation completed', 'metrics_file': 'metrics.json'})

def dashboard(request):
    with open('metrics.json', 'r') as f:
        metrics = json.load(f)
    return render(request, 'dashboard.html', {'metrics': json.dumps(metrics)})