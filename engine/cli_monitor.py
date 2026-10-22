"""
cli_monitor.py - Terminal Live Monitor for eBPF-RL-TCN IPS Engine
"""

import sys
import time
from typing import Any, Dict


class CLIMonitor:
    """Renders formatted real-time status output in the terminal."""

    def __init__(self):
        self.start_time = time.time()

    def render_header(self):
        print("=" * 78)
        print(" eBPF-RL-TCN Intrusion Prevention System | Live Control Plane Monitor")
        print("=" * 78)

    def render_step(self, step: int, event: Dict[str, Any], metrics: Dict[str, Any]):
        uptime = time.time() - self.start_time
        action_color = {
            "PASS": "\033[92m",        # Green
            "DROP": "\033[91m",        # Red
            "RATE_LIMIT": "\033[93m",  # Yellow
            "DEEP_INSPECT": "\033[96m" # Cyan
        }.get(event.get("action_name", ""), "\033[0m")
        reset_color = "\033[0m"

        line = (
            f"[{uptime:6.1f}s | Flow #{step:05d}] "
            f"IP: {event.get('src_ip', 'N/A'):<15} | "
            f"Threat: Cls {event.get('threat_pred', 0)} ({event.get('threat_prob', 0.0):.2f}) | "
            f"Action: {action_color}{event.get('action_name', 'UNKNOWN'):<12}{reset_color} | "
            f"Latency: {event.get('latency_us', 0.0):6.2f} µs | "
            f"Drops: {metrics.get('dropped_packets', 0)}"
        )
        print(line)
        sys.stdout.flush()

    def render_summary(self, total_flows: int, latencies: Dict[str, float], actions: Dict[int, int]):
        print("\n" + "=" * 78)
        print(" RUNTIME EXECUTION SUMMARY")
        print("=" * 78)
        print(f"Total Flows Processed: {total_flows:,}")
        print(f"Action Distribution:")
        print(f"  - PASS:         {actions.get(0, 0):,}")
        print(f"  - DROP:         {actions.get(1, 0):,}")
        print(f"  - RATE_LIMIT:   {actions.get(2, 0):,}")
        print(f"  - DEEP_INSPECT: {actions.get(3, 0):,}")
        print(f"Decision Latency Profile:")
        print(f"  - Mean: {latencies.get('mean_us', 0.0):.2f} µs")
        print(f"  - P50:  {latencies.get('p50_us', 0.0):.2f} µs")
        print(f"  - P95:  {latencies.get('p95_us', 0.0):.2f} µs")
        print(f"  - P99:  {latencies.get('p99_us', 0.0):.2f} µs")
        print("=" * 78)
