import os
import json
import random
import threading
import logging
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import Experiment, RoundLog, ClientLog
from .run_simualation import run_simulation
# Configuration
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.basicConfig(filename='fl_simulation.log', level=logging.INFO)
NUM_CLIENTS = 10
ROUNDS = 50
PARTICIPATION_RATE = 0.7

# Global state for simulation
current_experiment = None
metrics = []
client_datasets = []


@csrf_exempt
def start_simulation(request):
    global metrics
    metrics = []
    
    # Run simulation in a separate thread
    thread = threading.Thread(target=run_simulation)
    thread.start()
    
    return JsonResponse({
        'status': 'Simulation started',
        'message': 'Check dashboard for progress'
    })

def dashboard(request):
    experiment = current_experiment or Experiment.objects.last()
    round_logs = RoundLog.objects.filter(experiment=experiment).order_by('round_number')
    
    # Prepare chart data
    rounds = [log.round_number for log in round_logs]
    accuracies = [log.accuracy for log in round_logs]
    losses = [log.loss for log in round_logs]
    participants = [log.clients_participated for log in round_logs]
    
    # Prepare client activity data
    client_logs = ClientLog.objects.filter(experiment=experiment)
    adversarial_activity = [
        log for log in client_logs if log.is_adversarial and log.status == 'adversarial'
    ]
    
    return render(request, 'dashboard.html', {
        'experiment': experiment,
        'rounds': rounds,
        'accuracies': accuracies,
        'losses': losses,
        'participants': participants,
        'adversarial_count': len(adversarial_activity),
        'client_logs': client_logs[:20]  # Show recent 20 logs
    })

def get_metrics(request):
    return JsonResponse({'metrics': metrics})
