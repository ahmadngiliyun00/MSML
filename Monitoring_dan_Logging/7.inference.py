"""
7.inference.py
Client inference untuk mengetes MLflow model serving + memicu metrik monitoring.

Fungsi:
- Kirim request ke MLflow serving endpoint (/invocations)
- Ukur latency (client-side) dan catat sukses/gagal
- (Opsional) memicu failure dengan payload salah / stop server
- Cocok untuk pembuktian monitoring Prometheus + dashboard Grafana + alerting
"""

import argparse
import json
import time
from pathlib import Path

import requests


def load_payload(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Payload file tidak ditemukan: {path}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def post_invocations(server_url: str, payload: dict, timeout: float):
    """
    MLflow serving biasanya menerima:
    - Header: Content-Type: application/json
    - Body JSON: {"inputs": ...} atau {"dataframe_split": ...} tergantung model
    """
    headers = {"Content-Type": "application/json"}
    t0 = time.perf_counter()
    r = requests.post(server_url, headers=headers, json=payload, timeout=timeout)
    latency_s = time.perf_counter() - t0
    return r, latency_s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", required=True, help="MLflow serving invocations URL, contoh: http://127.0.0.1:5001/invocations")
    parser.add_argument("--payload", required=True, help="Path file JSON payload yang valid untuk model")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--repeat", type=int, default=1, help="Berapa kali inference diulang")
    parser.add_argument("--sleep", type=float, default=0.0, help="Delay antar request (detik)")
    parser.add_argument("--make-fail", action="store_true", help="Kirim payload rusak untuk memicu failures_total")
    args = parser.parse_args()

    payload = load_payload(args.payload)

    # payload rusak untuk testing failure (misal structure salah)
    bad_payload = {"invalid": "payload"}

    ok = 0
    fail = 0
    latencies = []

    for i in range(args.repeat):
        use_payload = bad_payload if args.make_fail else payload

        try:
            r, latency_s = post_invocations(args.server_url, use_payload, args.timeout)
            latencies.append(latency_s)

            if r.status_code >= 200 and r.status_code < 300:
                ok += 1
                print(f"[OK] #{i+1} status={r.status_code} latency_ms={latency_s*1000:.2f}")
                # Print ringkas output (biar tidak spam)
                try:
                    print("response:", r.json())
                except Exception:
                    print("response_text:", r.text[:300])
            else:
                fail += 1
                print(f"[FAIL] #{i+1} status={r.status_code} latency_ms={latency_s*1000:.2f}")
                print("response_text:", r.text[:300])

        except requests.RequestException as e:
            fail += 1
            print(f"[ERROR] #{i+1} request error: {e}")

        if args.sleep > 0:
            time.sleep(args.sleep)

    if latencies:
        avg_ms = (sum(latencies) / len(latencies)) * 1000
        p95_ms = sorted(latencies)[int(0.95 * (len(latencies) - 1))] * 1000
    else:
        avg_ms = 0
        p95_ms = 0

    print("\n=== SUMMARY ===")
    print(f"total={args.repeat} ok={ok} fail={fail}")
    print(f"avg_latency_ms={avg_ms:.2f} p95_latency_ms={p95_ms:.2f}")


if __name__ == "__main__":
    main()
