from locust import HttpUser, between, task

_PREDICT_PAYLOAD = {
    "time": 1000.0,
    "amount": 149.62,
    "v1": -1.3598,
    "v2": -0.0728,
    "v3": 2.5363,
    "v4": 1.3782,
    "v5": -0.3383,
    "v6": 0.4624,
    "v7": 0.2396,
    "v8": 0.0987,
    "v9": 0.3638,
    "v10": 0.0908,
    "v11": -0.5516,
    "v12": -0.6178,
    "v13": -0.9914,
    "v14": -0.3112,
    "v15": 1.4682,
    "v16": -0.4704,
    "v17": 0.2076,
    "v18": 0.0258,
    "v19": 0.4036,
    "v20": 0.2514,
    "v21": -0.0183,
    "v22": 0.2779,
    "v23": -0.1105,
    "v24": 0.0669,
    "v25": 0.1285,
    "v26": -0.1891,
    "v27": 0.1336,
    "v28": -0.0210,
}


class FraudAPIUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def predict(self) -> None:
        with self.client.post("/predict", json=_PREDICT_PAYLOAD, catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"POST /predict returned {resp.status_code}")
            else:
                resp.success()

    @task
    def health(self) -> None:
        with self.client.get("/health", catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"GET /health returned {resp.status_code}")
            else:
                resp.success()

    @task
    def drift(self) -> None:
        self.client.get("/drift")
