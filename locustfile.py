from locust import HttpUser, task, between
import random


class TrafficSignUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def predict_sign(self):
        """Test prediction endpoint"""
        files = {'file': open('test_images/sample.jpg', 'rb')}
        self.client.post('/predict', files=files)

    @task(1)
    def health_check(self):
        """Test health endpoint"""
        self.client.get('/health')

    @task(1)
    def get_metrics(self):
        """Test metrics endpoint"""
        self.client.get('/metrics')