from django.db import models

class Prediction(models.Model):
    ACTIVITY_CHOICES = [
        ('Normal', 'Normal'),
        ('Suspicious', 'Suspicious'),
    ]
    
    RESULT_CHOICES = [
        ('Normal', 'Normal Traffic'),
        ('Attack', 'DDoS / Suspicious Activity'),
    ]

    activity_type = models.CharField(max_length=100)
    action = models.CharField(max_length=100)
    login_attempts = models.FloatField()
    file_size = models.FloatField()
    anomaly_type = models.CharField(max_length=100)
    prediction_result = models.CharField(max_length=50, choices=RESULT_CHOICES)
    confidence_score = models.FloatField()
    timestamp = models.DateTimeField(auto_now_add=True)
    hour = models.IntegerField()
    day = models.IntegerField()

    class Meta:
        ordering = ['-timestamp']

    def __str__(self):
        return f"{self.prediction_result} - {self.timestamp}"
