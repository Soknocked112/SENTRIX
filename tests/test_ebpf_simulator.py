"""
test_ebpf_simulator.py - Unit tests for eBPF Map Simulator and XDP verdicts
"""

import time
import pytest
from ebpf.xdp_simulator import XDP_DROP, XDP_PASS, XDPSimulator


def test_ebpf_simulator_blocking():
    sim = XDPSimulator()
    ip = "192.168.1.10"

    assert not sim.is_ip_blocked(ip)

    # Block IP for 10 seconds
    sim.block_ip(ip, duration_sec=10.0, reason="Test Block")
    assert sim.is_ip_blocked(ip)

    # Packet from blocked IP should yield XDP_DROP
    verdict = sim.process_packet(src_ip=ip, dst_ip="10.0.0.1", src_port=1234, dst_port=80, proto=6)
    assert verdict == XDP_DROP

    # Packet from other IP should yield XDP_PASS
    verdict_other = sim.process_packet(src_ip="192.168.1.99", dst_ip="10.0.0.1", src_port=1234, dst_port=80, proto=6)
    assert verdict_other == XDP_PASS

    # Unblock IP
    sim.unblock_ip(ip)
    assert not sim.is_ip_blocked(ip)


def test_ebpf_simulator_rate_limiting():
    sim = XDPSimulator()
    ip = "10.0.0.50"

    # Set rate limit with burst capacity 2
    sim.set_rate_limit(ip, rate_pps=0.0, burst=2.0)

    # Packet 1 (allowed)
    v1 = sim.process_packet(src_ip=ip, dst_ip="10.0.0.1", src_port=1000, dst_port=80, proto=6)
    assert v1 == XDP_PASS

    # Packet 2 (allowed)
    v2 = sim.process_packet(src_ip=ip, dst_ip="10.0.0.1", src_port=1001, dst_port=80, proto=6)
    assert v2 == XDP_PASS

    # Packet 3 (exhausted burst -> XDP_DROP)
    v3 = sim.process_packet(src_ip=ip, dst_ip="10.0.0.1", src_port=1002, dst_port=80, proto=6)
    assert v3 == XDP_DROP
