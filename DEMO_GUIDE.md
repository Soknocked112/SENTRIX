# SENTRIX: 7-Day Checkpoint Demo & Practice Guide

This guide is designed for your upcoming project presentation. Everything is pre-built, tested, and ready. Follow this step-by-step walkthrough to rehearse and deliver your live demonstration.

---

## 1. Fast Demonstration Checklist (What to Open)

Before presenting, have two terminal windows and your browser ready:

* **Terminal Window 1:** Virtual environment activated, in project root.
* **Terminal Window 2:** Streamlit Web Dashboard running (`streamlit run dashboard/app.py`).
* **Browser Window:** `http://localhost:8501` open to the dashboard.

---

## 2. Five-Minute Presentation Script

### Minute 1: The Problem & Motivation
> *"Traditional network defense is fundamentally broken at high speeds. Passive IDS tools like Snort or Zeek inspect packets out-of-band and only alert us after malicious data has already entered the network. On the other hand, traditional firewalls are fast, but they rely on static rule sets and cannot detect dynamic, multi-step attacks like slow-rate DoS floods, distributed port scans, or password brute forcing.*
>
> *Copying high-speed packet streams into user space for deep software inspection wastes massive CPU cycles and causes severe packet drop. We built SENTRIX to solve this."*

### Minute 2: The Two-Tier Architecture
> *"SENTRIX decouples network defense into two collaborating tiers:*
> 1. * **In-Kernel Data Plane (Linux eBPF/XDP):** Runs directly in the network driver before socket buffer (`sk_buff`) allocation. It checks active IP block tables and token-bucket rate limiters in $O(1)$ time (~3.5 microseconds).*
> 2. * **Intelligent Control Plane (TCN + PPO Reinforcement Learning):** Aggregates flow records into sliding temporal windows ($T=10$), extracts causal temporal features using a Dilated Causal TCN, and uses a Proximal Policy Optimization (PPO) agent to choose mitigation actions under an availability-first reward function.*
>
> *When an attack is detected, the control plane inserts an active rule into the kernel BPF map, dropping all subsequent attack traffic at wire speed."*

### Minute 3: Live Terminal Demonstration

Run the live runtime pipeline in Terminal 1:
```bash
python3 scripts/04_run_ips.py --flows 20 --delay 0.05
```

**What to point out on screen:**
* Point out the **real-time ANSI log table**: show how legitimate traffic is marked with green `PASS`, while threats are actively marked with red `DROP` or yellow `RATE_LIMIT`.
* Show the **Latency column**: point out the microsecond decision times and the execution summary table.

Next, run the High-Speed Network benchmark:
```bash
python3 scripts/06_benchmark_high_speed.py --packets 5000
```

**What to point out on screen:**
* Highlight the line: **`Fast-Path Speedup: >300x faster than userspace`**.
* Explain: *"Notice how our eBPF fast path drops attack packets at **~3.5 µs**, compared to ~1.4 ms for userspace models. This is how we protect high-speed networks from buffer bloat."*

### Minute 4: Live Web Dashboard Demo

Switch to your browser (`http://localhost:8501`):

1. **Click "Live Traffic Replay":**
   * Select an attack profile (e.g., `DDoS`) and click **"Replay Traffic Stream"**.
   * Show the stream log and the drop counters incrementing in real time.
2. **Click "High-Speed Stress Test":**
   * Select `DDoS Flood Storm` and burst volume `3,000 packets`.
   * Click **"Launch Packet Burst"**.
   * Show the evaluator:
     * **Throughput Meter:** Processed thousands of packets in seconds.
     * **Speedup Gauge:** Demonstrating the kernel drop speed.
     * **Pie Chart:** Showing line-rate attack suppression while legitimate flows pass through.

### Minute 5: 5-Month Roadmap & Conclusion
> *"Today, we have demonstrated the complete end-to-end prototype: kernel packet filtering with eBPF, causal TCN sequence modeling on 2.83M records of CIC-IDS 2017, and an autonomous PPO policy agent.*
>
> *Over the next 5 months, we will take this from a prototype to a production line-rate architecture:*
> 1. * **AF_XDP (XSK) Zero-Copy Ingestion:** Direct ring-buffer memory sharing between kernel drivers and user space.*
> 2. * **Model Quantization:** Exporting the TCN and Actor models to TensorRT/ONNX INT8 to bring userspace inference under 50 µs.*
> 3. * **SmartNIC Hardware Offload:** Compiling our eBPF bytecode directly into programmable SmartNICs (Netronome/BlueField) for line-rate filtering without host CPU intervention.*
>
> *Thank you, and we welcome any questions."*

---

## 3. Evaluator Q&A Cheat Sheet

#### Q1: "Why did you use a TCN instead of an LSTM or GRU?"
**Answer:**  
*"LSTMs are strictly sequential: step $t$ must wait for step $t-1$, which creates a severe bottleneck during training and inference. Our Dilated Causal TCN uses 1D convolutions with exponentially increasing dilations ($d=1, 2, 4, 8$). This allows fully parallel training across sequences and a receptive field of 31 steps, with zero future information leakage via our custom `Chomp1d` layer."*

#### Q2: "Why reinforcement learning instead of a simple classification threshold?"
**Answer:**  
*"Intrusion prevention has asymmetric operational penalties. In high-speed production networks, dropping legitimate business traffic (a false positive) violates availability SLAs far more severely than briefly throttling suspicious flows. A static threshold classifier cannot balance availability vs. security. Our PPO agent is trained with a $-50.0$ penalty for false drops and a $-10.0$ penalty for false throttling, autonomously learning the optimal balance between security and uptime."*

#### Q3: "How does this prevent denial of service from overloading the AI model?"
**Answer:**  
*"That is the core strength of our 2-tier design. The AI model only inspects initial packets to classify the flow. Once the PPO policy outputs a `DROP` verdict, it writes the offending source IP directly into the kernel's `blocked_ips_map`. All subsequent millions of flood packets are dropped in kernel space at ~3.5 microseconds, completely shielding the userspace AI engine from traffic floods."*

#### Q4: "What dataset did you train this on?"
**Answer:**  
*"We trained on the Canadian Institute for Cybersecurity benchmark (CIC-IDS 2017), covering 2.83 million flow records across 78 numeric network features. We consolidated 15 attack types into 6 canonical categories and generated sliding temporal windows of length $T=10$ to preserve multi-step temporal dynamics."*

---

## 4. Emergency Backup Commands

If you ever need to reset or re-verify before the presentation:
```bash
# Verify all unit tests pass cleanly
pytest

# Re-run high-speed benchmark to refresh results plots
python3 scripts/06_benchmark_high_speed.py --packets 10000

# Launch web dashboard
streamlit run dashboard/app.py
```
