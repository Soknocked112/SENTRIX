"""
Step 4: show the "prevention" part. Test flows arrive one by one; the model judges each flow;
an address that is judged malicious several times in a row is BLOCKED and its later traffic is dropped.

NOTE: the dataset has no real IP addresses, so we give every flow a made-up address:
attack flows come from 10 pretend attackers, normal flows from 60 pretend clients.
Run:  python 04_prevention_demo.py
"""
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--strikes", type=int, default=3, help="suspicious flows in a row before an address is blocked")
parser.add_argument("--block-length", type=int, default=3000, help="how many flows later the block is lifted")
args = parser.parse_args()

if not (os.path.exists("models/ips_model.joblib") and os.path.exists("prepared/data.joblib")):
    raise SystemExit("Run steps 2 and 3 first.")

saved = joblib.load("models/ips_model.joblib")
model = saved["model"]
data = joblib.load("prepared/data.joblib")
X_test, y_test = data["X_test"], data["y_test"].reset_index(drop=True)

print(f"Judging {len(y_test):,} test flows ...")
predicted = model.predict(X_test)

rng = np.random.default_rng(7)
attackers = [f"172.16.0.{i}" for i in range(10, 20)]
clients = [f"192.168.10.{i}" for i in range(100, 160)]
is_attack = (y_test.values != "BENIGN")
source_ip = np.where(is_attack, rng.choice(attackers, len(y_test)), rng.choice(clients, len(y_test)))

strikes = {}
blocked_until = {}
events = []
timeline = []
counts = {"attack_total": 0, "attack_dropped": 0, "attack_passed": 0,
          "normal_total": 0, "normal_dropped": 0, "normal_passed": 0}

for i in range(len(y_test)):
    ip = source_ip[i]
    attack = bool(is_attack[i])
    counts["attack_total" if attack else "normal_total"] += 1

    kind = "attacker" if ip in attackers else "normal client"

    # 1) Has the block on this address expired?
    if ip in blocked_until and i >= blocked_until[ip]:
        del blocked_until[ip]
        strikes[ip] = 0
        events.append({"flow_number": i, "ip": ip, "event": "UNBLOCKED", "real_type": kind})

    if ip in blocked_until:
        # 2) Address is blocked: the flow is dropped before it can do any harm
        counts["attack_dropped" if attack else "normal_dropped"] += 1
    else:
        # 3) Address is allowed: the flow gets through and the model judges it
        counts["attack_passed" if attack else "normal_passed"] += 1
        if predicted[i] != "BENIGN":
            strikes[ip] = strikes.get(ip, 0) + 1
            if strikes[ip] >= args.strikes:
                blocked_until[ip] = i + args.block_length
                events.append({"flow_number": i, "ip": ip, "event": "BLOCKED", "real_type": kind})
        else:
            strikes[ip] = max(0, strikes.get(ip, 0) - 1)

    if i % 500 == 0 or i == len(y_test) - 1:
        timeline.append({"flow_number": i, **counts})

events_df = pd.DataFrame(events, columns=["flow_number", "ip", "event", "real_type"])
blocked_events = events_df[events_df["event"] == "BLOCKED"]
summary = {
    **counts,
    "attackers_blocked": int(blocked_events[blocked_events["real_type"] == "attacker"]["ip"].nunique()),
    "attackers_total": len(attackers),
    "normal_clients_wrongly_blocked": int(blocked_events[blocked_events["real_type"] == "normal client"]["ip"].nunique()),
    "normal_clients_total": len(clients),
    "attack_flows_stopped_percent": 100 * counts["attack_dropped"] / max(counts["attack_total"], 1),
    "normal_flows_dropped_percent": 100 * counts["normal_dropped"] / max(counts["normal_total"], 1),
    "strikes_needed": args.strikes,
}
os.makedirs("results", exist_ok=True)
json.dump(summary, open("results/simulation_summary.json", "w"), indent=2)
events_df.to_csv("results/block_events.csv", index=False)
pd.DataFrame(timeline).to_csv("results/timeline.csv", index=False)

print("\n=========== PREVENTION SIMULATION ===========")
print(f"Attacker addresses blocked        : {summary['attackers_blocked']} of {summary['attackers_total']}")
print(f"Normal clients wrongly blocked    : {summary['normal_clients_wrongly_blocked']} of {summary['normal_clients_total']}")
print(f"Attack flows stopped (dropped)    : {counts['attack_dropped']:,} of {counts['attack_total']:,} ({summary['attack_flows_stopped_percent']:.1f}%)")
print(f"Attack flows that got through     : {counts['attack_passed']:,}")
print(f"Normal flows lost (collateral)    : {counts['normal_dropped']:,} of {counts['normal_total']:,} ({summary['normal_flows_dropped_percent']:.2f}%)")
print("\nSaved: results/simulation_summary.json, results/block_events.csv, results/timeline.csv")
print("Continue to step 5 (the dashboard).")
