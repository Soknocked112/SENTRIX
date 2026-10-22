"""
xdp_manager.py - BCC-based eBPF/XDP Kernel Loader and Map Controller

Manages real kernel attachment via BCC (when running as root) and seamlessly
falls back to XDPSimulator when non-root privileges are detected.
"""

import ctypes
import ipaddress
import logging
import os
import socket
import time
from typing import Dict, List, Optional

from ebpf.bpf_maps import BPFMapInterface, BlockRule, FlowKey, FlowStats, IPSMetrics
from ebpf.xdp_simulator import XDPSimulator

logger = logging.getLogger("ebpf_manager")


def ip_to_u32(ip_str: str) -> int:
    """Convert IPv4 string to 32-bit unsigned integer (network byte order)."""
    return int.from_bytes(socket.inet_aton(ip_str), byteorder="little")


def u32_to_ip(ip_u32: int) -> str:
    """Convert 32-bit unsigned integer to IPv4 string."""
    return socket.inet_ntoa(ip_u32.to_bytes(4, byteorder="little"))


class XDPManager(BPFMapInterface):
    """
    Production eBPF/XDP manager supporting both real kernel execution (via BCC)
    and userspace simulation fallback.
    """

    def __init__(
        self,
        interface: str = "lo",
        c_source_path: str = "c_src/xdp_ips.c",
        force_simulator: bool = False,
    ):
        self.interface = interface
        self.c_source_path = c_source_path
        self.is_kernel_mode = False
        self.bpf = None
        self.simulator: Optional[XDPSimulator] = None

        if force_simulator or os.geteuid() != 0:
            logger.info("Non-root environment or simulator forced: initializing high-speed XDPSimulator.")
            self.simulator = XDPSimulator()
            self.is_kernel_mode = False
        else:
            try:
                from bcc import BPF
                if os.path.exists(c_source_path):
                    with open(c_source_path, "r") as f:
                        c_code = f.read()
                    self.bpf = BPF(text=c_code, cflags=["-w"])
                    fn = self.bpf.load_func("xdp_ips_filter", BPF.XDP)
                    self.bpf.attach_xdp(self.interface, fn, 0)
                    self.is_kernel_mode = True
                    logger.info(f"eBPF XDP filter successfully attached to interface: {self.interface}")
                else:
                    logger.warning(f"C source file not found at {c_source_path}, falling back to simulator.")
                    self.simulator = XDPSimulator()
            except Exception as e:
                logger.warning(f"Kernel eBPF initialization failed ({e}), falling back to XDPSimulator.")
                self.simulator = XDPSimulator()
                self.is_kernel_mode = False

    def block_ip(self, ip: str, duration_sec: float = 300.0, reason: str = "TCN-DRL Policy") -> bool:
        """Block an IP address."""
        if not self.is_kernel_mode:
            return self.simulator.block_ip(ip, duration_sec, reason)

        try:
            ip_val = ctypes.c_uint32(ip_to_u32(ip))
            expiry_ns = ctypes.c_uint64(time.time_ns() + int(duration_sec * 1e9))
            
            # Struct format corresponding to c_src/common_kern.h
            class BlockEntry(ctypes.Structure):
                _fields_ = [
                    ("expiry_ts_ns", ctypes.c_uint64),
                    ("drop_count", ctypes.c_uint64),
                    ("reason_code", ctypes.c_uint32),
                    ("action", ctypes.c_uint32),
                ]

            entry = BlockEntry(expiry_ts_ns=expiry_ns, drop_count=0, reason_code=3, action=1)
            self.bpf["blocked_ips_map"][ip_val] = entry
            return True
        except Exception as e:
            logger.error(f"Error blocking IP {ip} in kernel map: {e}")
            return False

    def unblock_ip(self, ip: str) -> bool:
        """Remove an IP address from the block list."""
        if not self.is_kernel_mode:
            return self.simulator.unblock_ip(ip)

        try:
            ip_val = ctypes.c_uint32(ip_to_u32(ip))
            del self.bpf["blocked_ips_map"][ip_val]
            return True
        except KeyError:
            return False
        except Exception as e:
            logger.error(f"Error unblocking IP {ip} in kernel map: {e}")
            return False

    def is_ip_blocked(self, ip: str) -> bool:
        """Check if an IP address is blocked."""
        if not self.is_kernel_mode:
            return self.simulator.is_ip_blocked(ip)

        try:
            ip_val = ctypes.c_uint32(ip_to_u32(ip))
            entry = self.bpf["blocked_ips_map"].get(ip_val)
            if entry:
                now_ns = time.time_ns()
                if now_ns < entry.expiry_ts_ns:
                    return True
                else:
                    del self.bpf["blocked_ips_map"][ip_val]
            return False
        except Exception:
            return False

    def set_rate_limit(self, ip: str, rate_pps: float = 100.0, burst: float = 50.0) -> bool:
        """Configure token bucket rate limiting for an IP."""
        if not self.is_kernel_mode:
            return self.simulator.set_rate_limit(ip, rate_pps, burst)

        try:
            ip_val = ctypes.c_uint32(ip_to_u32(ip))
            now_ns = time.time_ns()

            class TokenBucketEntry(ctypes.Structure):
                _fields_ = [
                    ("tokens", ctypes.c_uint64),
                    ("last_update_ns", ctypes.c_uint64),
                    ("rate_pps", ctypes.c_uint64),
                    ("capacity", ctypes.c_uint64),
                ]

            tb = TokenBucketEntry(
                tokens=int(burst),
                last_update_ns=now_ns,
                rate_pps=int(rate_pps),
                capacity=int(burst),
            )
            self.bpf["rate_limit_map"][ip_val] = tb
            return True
        except Exception as e:
            logger.error(f"Error setting rate limit for IP {ip}: {e}")
            return False

    def remove_rate_limit(self, ip: str) -> bool:
        """Remove rate limiter for an IP."""
        if not self.is_kernel_mode:
            return self.simulator.remove_rate_limit(ip)

        try:
            ip_val = ctypes.c_uint32(ip_to_u32(ip))
            del self.bpf["rate_limit_map"][ip_val]
            return True
        except KeyError:
            return False
        except Exception as e:
            logger.error(f"Error removing rate limit for {ip}: {e}")
            return False

    def get_flow_stats(self, flow_key: FlowKey) -> Optional[FlowStats]:
        """Retrieve 5-tuple statistics."""
        if not self.is_kernel_mode:
            return self.simulator.get_flow_stats(flow_key)
        return None

    def get_metrics(self) -> IPSMetrics:
        """Retrieve aggregated metrics."""
        if not self.is_kernel_mode:
            return self.simulator.get_metrics()

        try:
            counters = self.bpf["metrics_map"][ctypes.c_uint32(0)]
            total_rx = sum(c.rx_packets for c in counters)
            total_drop = sum(c.dropped_packets for c in counters)
            total_pass = sum(c.passed_packets for c in counters)
            total_rate_limited = sum(c.rate_limited_packets for c in counters)
            return IPSMetrics(
                rx_packets=total_rx,
                dropped_packets=total_drop,
                passed_packets=total_pass,
                rate_limited_packets=total_rate_limited,
            )
        except Exception:
            return IPSMetrics()

    def get_active_blocks(self) -> List[BlockRule]:
        """List active blocked IPs."""
        if not self.is_kernel_mode:
            return self.simulator.get_active_blocks()

        active = []
        try:
            now_ns = time.time_ns()
            for k, v in self.bpf["blocked_ips_map"].items():
                if now_ns < v.expiry_ts_ns:
                    active.append(BlockRule(
                        ip=u32_to_ip(k.value),
                        expiry_ts_ns=v.expiry_ts_ns,
                        reason="Kernel Rule",
                        drop_count=v.drop_count,
                    ))
        except Exception as e:
            logger.error(f"Error reading blocked IPs from kernel map: {e}")
        return active

    def clear(self) -> None:
        """Clear all active rules."""
        if not self.is_kernel_mode:
            self.simulator.clear()
        else:
            try:
                self.bpf["blocked_ips_map"].clear()
                self.bpf["rate_limit_map"].clear()
            except Exception:
                pass

    def detach(self) -> None:
        """Detach XDP program upon shutdown."""
        if self.is_kernel_mode and self.bpf:
            try:
                self.bpf.remove_xdp(self.interface, 0)
                logger.info(f"Detached XDP program from {self.interface}")
            except Exception as e:
                logger.error(f"Error detaching XDP: {e}")
