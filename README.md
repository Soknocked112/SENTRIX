# SENTRIX

### In-Kernel eBPF/XDP Filtering with Temporal Convolutional Networks and Reinforcement Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![eBPF / XDP](https://img.shields.io/badge/DataPlane-Linux%20eBPF%20%2F%20XDP-orange.svg)](https://ebpf.io/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/RL%20Env-Gymnasium-green.svg)](https://gymnasium.farama.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

SENTRIX is an active intrusion prevention system (IPS) designed to bridge high-throughput kernel packet filtering with sequential machine learning. It couples in-kernel packet inspection and filtering using Linux eBPF/XDP with a userspace control plane driven by a Dilated Causal Temporal Convolutional Network (TCN) and a Proximal Policy Optimization (PPO) reinforcement learning agent.

The system is trained and evaluated on the Canadian Institute for Cybersecurity (CIC-IDS 2017) dataset across 2.83 million flow records.

---

## Overview

Traditional network intrusion detection systems (such as Snort, Suricata, or Zeek) operate out-of-band: they inspect copies of network traffic and raise alerts after a packet has already traversed the network stack. While valuable for post-incident auditing, passive detection cannot stop ongoing data exfiltration or high-rate denial-of-service floods in real time.

Conversely, traditional firewalls and ACLs filter traffic at line rate, but rely on static IP lists, port rules, or regex signatures. They cannot detect dynamic, multi-step temporal attack patterns (such as slow-rate DoS, distributed port scans, or brute-force attempts) without accumulating massive rule sets that degrade forwarding performance.

SENTRIX decouples the architecture into two communicating layers:
1. **Fast-path data plane (in-kernel eBPF/XDP):** Operates directly inside the network driver before socket buffer (`sk_buff`) allocation. It checks active IP block tables and token-bucket rate limiters in O(1) time (~8 µs).
2. **Intelligent control plane (userspace TCN-DRL):** Aggregates flow records into sliding temporal windows, extracts temporal features using a causal TCN, and uses a PPO policy agent to select mitigation actions (`PASS`, `DROP`, `RATE_LIMIT`, `DEEP_INSPECT`) under an asymmetric SLA reward function. Decisions are written back into kernel BPF maps in real time.

---

## System Architecture

```mermaid
flowchart TD
    subgraph Ingress[Ingress Network Traffic]
        PKT[Incoming Packets - CIC-IDS 2017 / PCAP]
    end

    subgraph KernelSpace[Linux Kernel Space: eBPF / XDP]
        XDP[XDP Packet Hook: c_src/xdp_ips.c]
        BMAP{blocked_ips_map Lookup}
        RLMAP{rate_limit_map Check}
        FLOWMAP[flow_stats_map: 5-Tuple Tracking]
        DROP1[XDP_DROP: Fast Path ~8 µs]
        DROP2[XDP_DROP: Rate Limit Exceeded]
        PASS1[XDP_PASS]
    end

    subgraph ControlPlane[Userspace Control Plane]
        RING[Flow Event Buffer]
        BUF[Sliding Window Buffer: T = 10]
        TCN[Dilated Causal TCN: models/tcn.py]
        ENV[Gymnasium IPS Environment: Box 40, Discrete 4]
        PPO[PPO Actor-Critic Agent: models/drl_agent.py]
        SYNC[eBPF Map Synchronizer: ebpf/xdp_manager.py]
    end

    PKT --> XDP
    XDP --> BMAP
    BMAP -- Match --> DROP1
    BMAP -- No Match --> RLMAP
    RLMAP -- Exceeded --> DROP2
    RLMAP -- Under Limit --> FLOWMAP
    FLOWMAP --> PASS1
    FLOWMAP --> RING
    RING --> BUF
    BUF --> TCN
    TCN -- Latent Vector z_t: 32-dim --> ENV
    TCN -- Threat Logits: 6 classes --> ENV
    ENV --> PPO
    PPO -- Action: PASS / DROP / RATE_LIMIT / INSPECT --> SYNC
    SYNC -- Update BPF Tables --> BMAP
    SYNC -- Update Token Buckets --> RLMAP
```

---

## Core Components

### 1. In-Kernel eBPF/XDP Data Plane (`c_src/xdp_ips.c`, `ebpf/`)
The data plane runs directly in the Linux network driver before socket buffer allocation:
- **IP Block Table (`blocked_ips_map`):** BPF hash table storing blocked source IP addresses with nanosecond expiration timestamps. Arriving packets that match an unexpired entry are immediately dropped via `XDP_DROP` without traversing the host network stack.
- **Token Bucket Rate Limiter (`rate_limit_map`):** Per-IP token bucket tracking arrival timestamps via `bpf_ktime_get_ns()`. Replenishes tokens according to configured packets per second (PPS) and drops burst excess.
- **Flow Statistics Table (`flow_stats_map`):** Tracks active 5-tuple statistics (packet count, byte count, timestamps, TCP flags).
- **Userspace Fallback Simulator (`ebpf/xdp_simulator.py`):** For environments without root access or BPF kernel headers, a userspace simulator replicates the exact map structures and verdict logic, allowing local development and testing.

### 2. Dilated Causal TCN (`models/tcn.py`)
Rather than relying on recurrent architectures (LSTMs or GRUs) which suffer from sequential training bottlenecks and vanishing gradients, SENTRIX uses a Temporal Convolutional Network:
- **Causal Convolutions:** A custom `Chomp1d` layer trims right-side padding after 1D convolutions, ensuring that output at step $t$ depends only on inputs from step $t$ and earlier (no future information leakage).
- **Exponential Dilations:** Dilation factors double across 4 residual blocks ($d \in \{1, 2, 4, 8\}$) with kernel size $K=3$, giving an effective receptive field of 31 steps over 10-step flow sequences.
- **Dual Output Heads:** Simultaneously outputs 6-class threat probability logits (Benign, DoS, DDoS, PortScan, BruteForce, WebAttack) and extracts a 32-dimensional latent feature representation for the RL policy.

### 3. PPO Reinforcement Learning Policy (`models/drl_agent.py`, `environment/ips_env.py`)
Intrusion prevention has asymmetric operational costs: dropping legitimate business traffic (a false positive) violates availability SLAs far more severely than throttling suspicious flows. SENTRIX formulates action selection as a Markov Decision Process:
- **Observation Space (40 dimensions):** 32-dimensional TCN latent embedding combined with 8 dynamic flow indicators (duration, packet count, byte rate, active drop ratio, active rules, protocol flags).
- **Action Space (4 discrete actions):**
  - `0: PASS` — Forward packet through standard networking stack.
  - `1: DROP` — Add IP to kernel block table for immediate driver-level drops.
  - `2: RATE_LIMIT` — Throttle flow rate via token bucket.
  - `3: DEEP_INSPECT` — Route flow to a secondary inspection buffer.

The reward formulation enforces asymmetric penalties:

| Action | Flow Condition | Reward | Description |
|---|---|---:|---|
| `DROP` | Malicious flow | **+10.0** | Attack mitigated |
| `PASS` | Benign flow | **+2.0** | Legitimate traffic forwarded |
| `DROP` | Benign flow | **-50.0** | False positive (high SLA penalty) |
| `PASS` | Malicious flow | **-30.0** | False negative (security exposure) |
| `RATE_LIMIT` | Malicious flow | **+5.0** | Traffic throttled |
| `RATE_LIMIT` | Benign flow | **-10.0** | Unnecessary throttling penalty |
| `DEEP_INSPECT` | Any flow | **-0.5 + R(resolved)** | Inspection buffer overhead |

---

## Dataset & Preprocessing

The system is trained and evaluated on the **CIC-IDS 2017** benchmark:
- **Data Volume:** 2,830,743 records across 8 capture files covering Monday through Friday traffic.
- **Feature Cleaning:** Non-generalizable identifiers (Source IP, Destination IP, Timestamp, Flow ID) are removed. Infinite values and NaNs are imputed with column medians across 78 numeric features, followed by standard normalization (`StandardScaler`).
- **Class Grouping:** Granular attack signatures are mapped into 6 canonical categories: Benign, DoS, DDoS, PortScan, BruteForce, and WebAttack.
- **Sequence Generation:** Flow records are converted into sliding temporal windows of length $T=10$ with stride 1, partitioned into train (70%), validation (15%), and test (15%) splits.

---

## Experimental Results

Evaluation on the unseen test split (11,375 sequences):

### 1. TCN Threat Detection Performance

| Threat Class | Precision | Recall | F1-Score | Test Support |
|---|:---:|:---:|:---:|:---:|
| **BENIGN** | 0.9748 | 0.9284 | 0.9511 | 2,250 |
| **DoS** | 0.9767 | 0.9880 | 0.9823 | 2,250 |
| **DDoS** | 0.9873 | 0.9987 | 0.9929 | 2,250 |
| **PortScan** | 0.9833 | 0.9947 | 0.9890 | 2,250 |
| **BruteForce** | 0.9900 | 0.9976 | 0.9938 | 2,075 |
| **WebAttack** | 0.8722 | 0.9100 | 0.8907 | 300 |
| **Overall Macro Avg** | **0.9640** | **0.9696** | **0.9666** | **11,375** |
| **Overall Accuracy** | — | — | **97.93%** | **11,375** |

### 2. Mitigation & Latency Profile

| Metric | Measured Value | Description |
|---|:---:|---|
| **Overall Detection Accuracy** | **97.93%** | Multi-class sequence accuracy on test split |
| **False Positive Rate (Benign Drops)** | **2.25%** | Controlled via the -50.0 false-positive reward penalty |
| **eBPF Kernel Fast-Path Lookup** | **~8.0 µs** | O(1) hash map check inside XDP driver |
| **Decision Latency (Median P50)** | **1.50 ms** | Full pipeline: buffer + TCN + PPO inference |
| **Decision Latency (P95)** | **2.31 ms** | 95th percentile under continuous stream |
| **Decision Latency (P99)** | **2.59 ms** | 99th percentile under continuous stream |

Generated plots are stored in `results/`:
- `results/high_speed_benchmark.png` — Line-rate throughput and eBPF kernel drop latency profile.
- `results/confusion_matrix.png` — Per-class classification confusion matrix.
- `results/latency_distribution.png` — Decision latency distribution profile.
- `results/action_policy_distribution.png` — Enforced policy actions across threat classes.
- `results/tcn_training_curves.png` — Training and validation loss/accuracy curves.
- `results/drl_training_curves.png` — PPO episode reward convergence.

---

## Project Structure

```
├── c_src/
│   ├── xdp_ips.c                    # In-kernel eBPF/XDP C packet filter program
│   ├── common_kern.h                # Struct definitions for 5-tuple, token bucket, and BPF maps
│   └── Makefile                     # Clang/LLVM build rules for eBPF bytecode
├── configs/
│   └── default_config.yaml          # Model hyperparameters, environment, and path configs
├── data/
│   ├── preprocess.py                # Dataset ingestion, cleaning, normalization, and windowing
│   ├── dataset.py                   # PyTorch sliding-window Dataset and DataLoader
│   └── __init__.py
├── ebpf/
│   ├── bpf_maps.py                  # Map interfaces and dataclasses
│   ├── xdp_manager.py               # BCC loader and kernel attachment with fallback
│   ├── xdp_simulator.py             # Userspace simulator for non-root testing
│   └── __init__.py
├── environment/
│   ├── ips_env.py                   # Gymnasium IPS environment
│   ├── reward_functions.py          # Reward functions with false-positive penalties
│   └── __init__.py
├── models/
│   ├── tcn.py                       # Dilated Causal Temporal Convolutional Network
│   ├── drl_agent.py                 # PPO Actor-Critic agent with GAE
│   └── __init__.py
├── engine/
│   ├── ips_orchestrator.py          # Event loop connecting eBPF maps with TCN-DRL inference
│   ├── cli_monitor.py               # Live terminal status monitor
│   └── __init__.py
├── dashboard/
│   └── app.py                       # Streamlit web dashboard
├── scripts/
│   ├── 01_preprocess_data.py        # Data preprocessing script
│   ├── 02_train_tcn.py              # TCN model training
│   ├── 03_train_drl.py              # PPO policy training
│   ├── 04_run_ips.py                # Live IPS pipeline demo
│   ├── 05_evaluate.py               # Benchmark evaluation script
│   └── 06_benchmark_high_speed.py   # High-speed network throughput stress test
├── tests/                           # Pytest test suite
│   ├── test_data.py
│   ├── test_tcn.py
│   ├── test_env.py
│   ├── test_drl.py
│   ├── test_ebpf_simulator.py
│   └── test_orchestrator.py
├── pyproject.toml
├── setup.py
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Quickstart

### 1. Environment Setup

```bash
git clone https://github.com/Soknocked112/SENTRIX.git
cd SENTRIX

# Create virtual environment inheriting system site-packages (for BCC if available)
python3 -m venv --system-site-packages .venv
source .venv/bin/activate

# Install dependencies and package in editable mode
pip install -r requirements.txt
pip install -e .
```

### 2. Preprocess Data

Preprocesses raw CSV captures from `CICIDS 2017`, normalizes features, and generates temporal sequence files:

```bash
# Balanced sampling mode (15,000 samples per class)
python3 scripts/01_preprocess_data.py

# Or process full dataset
python3 scripts/01_preprocess_data.py --full
```

### 3. Train Models

```bash
# Train the TCN feature extractor (saved to checkpoints/tcn_best.pth)
python3 scripts/02_train_tcn.py --epochs 6 --batch_size 128

# Train the PPO agent (saved to checkpoints/drl_ppo_best.pth)
python3 scripts/03_train_drl.py --episodes 80 --steps_per_episode 300
```

### 4. Run Live Pipeline

```bash
# Simulator mode (runs without root privileges)
python3 scripts/04_run_ips.py --flows 50 --delay 0.01

# Kernel mode (attaches real xdp_ips.c to Linux network interface via BCC)
sudo python3 scripts/04_run_ips.py --kernel --flows 50
```

### 5. Run Evaluation Benchmarks

```bash
python3 scripts/05_evaluate.py
```

### 6. Run High-Speed Network Stress Test

```bash
python3 scripts/06_benchmark_high_speed.py --packets 10000
```

### 7. Interactive Web Dashboard

```bash
streamlit run dashboard/app.py
```

---

## Running Tests

The test suite covers data sequence generation, TCN causality, Gymnasium step/reset dynamics, PPO loss updates, and eBPF simulator logic:

```bash
pytest
```

---

## Future Improvements

- **Zero-Copy Ingestion (AF_XDP):** Integrate AF_XDP (XSK) sockets to share packet memory between kernel drivers and the userspace inference pipeline without buffer copies.
- **Inference Optimization:** Export the TCN and Actor networks to ONNX Runtime or TensorRT with INT8 quantization to reduce userspace decision latency below 100 µs.
- **SmartNIC Hardware Offload:** Compile eBPF bytecode directly to programmable SmartNICs (e.g. Netronome / BlueField) to achieve line-rate filtering without host CPU utilization.

---

## Project Team

This project was developed collaboratively by a 2-person engineering team:
- **Data Plane & Infrastructure:** Linux eBPF/XDP kernel hook implementation, BPF map synchronizer, packet simulation environment, and network throughput benchmarks.
- **Control Plane & Modeling:** Temporal Convolutional Network (TCN) architecture, causal feature extraction, Gymnasium IPS environment, and PPO reinforcement learning agent.

---

## License

This project is licensed under the [MIT License](LICENSE).
