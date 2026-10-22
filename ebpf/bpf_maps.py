"""
bpf_maps.py - Common interface and data structures for eBPF/XDP map operations.
Supports both real kernel BPF maps and userspace simulation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import time
from typing import Dict, List, Optional, Tuple


@dataclass
class FlowKey:
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    proto: int

    def __hash__(self):
        return hash((self.src_ip, self.dst_ip, self.src_port, self.dst_port, self.proto))


@dataclass
class FlowStats:
    packets: int
    bytes: int
    first_seen_ns: int
    last_seen_ns: int
    tcp_flags: int = 0


@dataclass
class BlockRule:
    ip: str
    expiry_ts_ns: int
    reason: str
    drop_count: int = 0
    created_at: float = 0.0

    @property
    def is_expired(self) -> bool:
        return time.time_ns() >= self.expiry_ts_ns


@dataclass
class TokenBucket:
    tokens: float
    last_update_ns: int
    rate_pps: float
    capacity: float


@dataclass
class IPSMetrics:
    rx_packets: int = 0
    dropped_packets: int = 0
    passed_packets: int = 0
    rate_limited_packets: int = 0
    inspected_packets: int = 0

    @property
    def drop_ratio(self) -> float:
        return self.dropped_packets / max(1, self.rx_packets)

    def to_dict(self) -> Dict[str, float]:
        return {
            "rx_packets": self.rx_packets,
            "dropped_packets": self.dropped_packets,
            "passed_packets": self.passed_packets,
            "rate_limited_packets": self.rate_limited_packets,
            "inspected_packets": self.inspected_packets,
            "drop_ratio": self.drop_ratio,
        }


class BPFMapInterface(ABC):
    """Abstract interface defining required eBPF map operations."""

    @abstractmethod
    def block_ip(self, ip: str, duration_sec: float = 300.0, reason: str = "TCN-DRL Policy") -> bool:
        """Add an IP address to the blocked table with an expiration timestamp."""
        pass

    @abstractmethod
    def unblock_ip(self, ip: str) -> bool:
        """Remove an IP address from the blocked table."""
        pass

    @abstractmethod
    def is_ip_blocked(self, ip: str) -> bool:
        """Check whether an IP is currently blocked and unexpired."""
        pass

    @abstractmethod
    def set_rate_limit(self, ip: str, rate_pps: float = 100.0, burst: float = 50.0) -> bool:
        """Configure token bucket rate limiting for an IP."""
        pass

    @abstractmethod
    def remove_rate_limit(self, ip: str) -> bool:
        """Remove rate limit for an IP."""
        pass

    @abstractmethod
    def get_flow_stats(self, flow_key: FlowKey) -> Optional[FlowStats]:
        """Retrieve aggregated flow statistics for a given 5-tuple."""
        pass

    @abstractmethod
    def get_metrics(self) -> IPSMetrics:
        """Retrieve aggregate system metrics counters."""
        pass

    @abstractmethod
    def get_active_blocks(self) -> List[BlockRule]:
        """List all currently active block rules."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all active rules and counters."""
        pass
