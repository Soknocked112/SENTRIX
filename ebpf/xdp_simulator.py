from collections import defaultdict
import time
from typing import Dict, List, Optional, Tuple

from ebpf.bpf_maps import BPFMapInterface, BlockRule, FlowKey, FlowStats, IPSMetrics, TokenBucket

# XDP verdict codes
XDP_PASS = 0
XDP_DROP = 1
XDP_TX = 2
XDP_ABORTED = 3


class XDPSimulator(BPFMapInterface):
    """
    Userspace simulator for the XDP filtering pipeline.
    Replicates BPF map lookups and token-bucket throttling for testing without root.
    """

    def __init__(self, max_blocked: int = 65536, max_rate_limit: int = 65536):
        self.max_blocked = max_blocked
        self.max_rate_limit = max_rate_limit
        self.blocked_ips: Dict[str, BlockRule] = {}
        self.rate_limit_map: Dict[str, TokenBucket] = {}
        self.flow_stats_map: Dict[FlowKey, FlowStats] = {}
        self.metrics = IPSMetrics()

    def block_ip(self, ip: str, duration_sec: float = 300.0, reason: str = "TCN-DRL Policy") -> bool:
        now_ns = time.time_ns()
        expiry_ts_ns = now_ns + int(duration_sec * 1e9)
        self.blocked_ips[ip] = BlockRule(
            ip=ip,
            expiry_ts_ns=expiry_ts_ns,
            reason=reason,
            drop_count=0,
            created_at=time.time(),
        )
        return True

    def unblock_ip(self, ip: str) -> bool:
        if ip in self.blocked_ips:
            del self.blocked_ips[ip]
            return True
        return False

    def is_ip_blocked(self, ip: str) -> bool:
        rule = self.blocked_ips.get(ip)
        if not rule:
            return False
        if rule.is_expired:
            del self.blocked_ips[ip]
            return False
        return True

    def set_rate_limit(self, ip: str, rate_pps: float = 100.0, burst: float = 50.0) -> bool:
        now_ns = time.time_ns()
        self.rate_limit_map[ip] = TokenBucket(
            tokens=burst,
            last_update_ns=now_ns,
            rate_pps=rate_pps,
            capacity=burst,
        )
        return True

    def remove_rate_limit(self, ip: str) -> bool:
        if ip in self.rate_limit_map:
            del self.rate_limit_map[ip]
            return True
        return False

    def get_flow_stats(self, flow_key: FlowKey) -> Optional[FlowStats]:
        return self.flow_stats_map.get(flow_key)

    def get_metrics(self) -> IPSMetrics:
        return self.metrics

    def get_active_blocks(self) -> List[BlockRule]:
        now_ns = time.time_ns()
        active = []
        expired = []
        for ip, rule in self.blocked_ips.items():
            if now_ns < rule.expiry_ts_ns:
                active.append(rule)
            else:
                expired.append(ip)
        for ip in expired:
            del self.blocked_ips[ip]
        return active

    def clear(self) -> None:
        self.blocked_ips.clear()
        self.rate_limit_map.clear()
        self.flow_stats_map.clear()
        self.metrics = IPSMetrics()

    def process_packet(
        self,
        src_ip: str,
        dst_ip: str,
        src_port: int,
        dst_port: int,
        proto: int,
        pkt_len: int = 64,
        tcp_flags: int = 0,
    ) -> int:
        now_ns = time.time_ns()
        self.metrics.rx_packets += 1

        # Check blocklist map
        rule = self.blocked_ips.get(src_ip)
        if rule:
            if now_ns < rule.expiry_ts_ns:
                rule.drop_count += 1
                self.metrics.dropped_packets += 1
                return XDP_DROP
            else:
                del self.blocked_ips[src_ip]

        # Check token bucket rate limiter
        tb = self.rate_limit_map.get(src_ip)
        if tb:
            elapsed_ns = now_ns - tb.last_update_ns
            added_tokens = (elapsed_ns * tb.rate_pps) / 1e9
            if added_tokens > 0:
                tb.tokens = min(tb.capacity, tb.tokens + added_tokens)
                tb.last_update_ns = now_ns

            if tb.tokens >= 1.0:
                tb.tokens -= 1.0
                self.metrics.passed_packets += 1
            else:
                self.metrics.dropped_packets += 1
                self.metrics.rate_limited_packets += 1
                return XDP_DROP

        # Update 5-tuple flow stats
        key = FlowKey(src_ip, dst_ip, src_port, dst_port, proto)
        stats = self.flow_stats_map.get(key)
        if stats:
            stats.packets += 1
            stats.bytes += pkt_len
            stats.last_seen_ns = now_ns
            stats.tcp_flags |= tcp_flags
        else:
            self.flow_stats_map[key] = FlowStats(
                packets=1,
                bytes=pkt_len,
                first_seen_ns=now_ns,
                last_seen_ns=now_ns,
                tcp_flags=tcp_flags,
            )

        self.metrics.passed_packets += 1
        return XDP_PASS
