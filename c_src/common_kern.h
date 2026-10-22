#ifndef __COMMON_KERN_H
#define __COMMON_KERN_H

#include <linux/types.h>

#define ETH_P_IP 0x0800
#define IPPROTO_TCP 6
#define IPPROTO_UDP 17
#define IPPROTO_ICMP 1

/* Action Verdicts */
#define IPS_ACTION_PASS        0
#define IPS_ACTION_DROP        1
#define IPS_ACTION_RATE_LIMIT  2
#define IPS_ACTION_INSPECT     3

/* Reason Codes for Blocking */
#define REASON_MANUAL_RULE      1
#define REASON_TCN_ANOMALY      2
#define REASON_RL_POLICY_DROP   3
#define REASON_RATE_LIMIT_EXCEED 4

/* 5-Tuple Network Flow Identifier */
struct flow_key {
    __u32 src_ip;
    __u32 dst_ip;
    __u16 src_port;
    __u16 dst_port;
    __u8  proto;
    __u8  pad[3];
};

/* Flow Statistics Tracked in Kernel */
struct flow_stats {
    __u64 packets;
    __u64 bytes;
    __u64 first_seen_ns;
    __u64 last_seen_ns;
    __u8  tcp_flags;
    __u8  pad[7];
};

/* IP Block Table Entry */
struct ip_block_entry {
    __u64 expiry_ts_ns;
    __u64 drop_count;
    __u32 reason_code;
    __u32 action;
};

/* Token Bucket State for Fast Rate Limiting */
struct token_bucket {
    __u64 tokens;          /* Current token count */
    __u64 last_update_ns;  /* Timestamp of last token replenishment */
    __u64 rate_pps;        /* Replenishment rate in packets per second */
    __u64 capacity;        /* Max burst tokens */
};

/* Aggregate Performance Counters */
struct ips_counters {
    __u64 rx_packets;
    __u64 dropped_packets;
    __u64 passed_packets;
    __u64 rate_limited_packets;
    __u64 inspected_packets;
};

/* Real-Time Alert Event for Ring Buffer / Perf Event */
struct ips_alert_event {
    __u64 timestamp_ns;
    __u32 src_ip;
    __u32 dst_ip;
    __u16 src_port;
    __u16 dst_port;
    __u8  proto;
    __u8  action;
    __u16 threat_class;
};

#endif /* __COMMON_KERN_H */
