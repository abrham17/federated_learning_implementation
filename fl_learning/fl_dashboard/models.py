from django.db import models

class Experiment(models.Model):
    STATUS_CHOICES = [
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed')
    ]
    name = models.CharField(max_length=255)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='running')
    rounds = models.IntegerField(default=50)
    start_time = models.DateTimeField(auto_now_add=True)
    end_time = models.DateTimeField(null=True, blank=True)

class RoundLog(models.Model):
    experiment = models.ForeignKey(Experiment, on_delete=models.CASCADE)
    round_number = models.IntegerField()
    accuracy = models.FloatField()
    loss = models.FloatField()
    clients_participated = models.IntegerField()
    timestamp = models.DateTimeField(auto_now_add=True)

class ClientLog(models.Model):
    STATUS_CHOICES = [
        ('completed', 'Completed'),
        ('dropped', 'Dropped'),
        ('failed', 'Failed'),
        ('adversarial', 'Adversarial')
    ]
    experiment = models.ForeignKey(Experiment, on_delete=models.CASCADE)
    client_id = models.IntegerField()
    round_number = models.IntegerField()
    status = models.CharField(max_length=20, choices=STATUS_CHOICES)
    loss = models.FloatField(null=True, blank=True)
    accuracy = models.FloatField(null=True, blank=True)
    is_adversarial = models.BooleanField(default=False)
    is_straggler = models.BooleanField(default=False)
    timestamp = models.DateTimeField(auto_now_add=True)