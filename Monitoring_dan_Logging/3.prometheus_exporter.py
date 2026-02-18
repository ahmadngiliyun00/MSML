from prometheus_client import start_http_server, Gauge, Counter, Histogram
import time, requests

MODEL_HEALTH_URL = "http://127.0.0.1:5001/health"

# 3+ metrics
mlflow_up = Gauge("mlflow_model_up", "Model health status (1=up, 0=down)")
mlflow_latency = Histogram("mlflow_health_latency_seconds", "Latency to /health")
mlflow_health_checks_total = Counter("mlflow_health_checks_total", "Total health checks")
mlflow_health_failures_total = Counter("mlflow_health_failures_total", "Total failed health checks")

def loop():
    while True:
        mlflow_health_checks_total.inc()
        t0 = time.time()
        try:
            r = requests.get(MODEL_HEALTH_URL, timeout=3)
            dt = time.time() - t0
            mlflow_latency.observe(dt)

            if r.status_code == 200:
                mlflow_up.set(1)
            else:
                mlflow_up.set(0)
                mlflow_health_failures_total.inc()
        except Exception:
            dt = time.time() - t0
            mlflow_latency.observe(dt)
            mlflow_up.set(0)
            mlflow_health_failures_total.inc()

        time.sleep(5)

if __name__ == "__main__":
    start_http_server(8000, addr="0.0.0.0")  # exporter metrics: http://127.0.0.1:8000/metrics
    print("health check ok")
    loop()
