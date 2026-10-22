/*
 * xdp_ips.c - Real-Time eBPF / XDP High-Performance Packet Filter
 *
 * Provides in-kernel packet inspection and sub-millisecond verdict execution:
 * 1. O(1) IP Blocklist filtering via BPF Hash Map (XDP_DROP).
 * 2. Token Bucket Rate Limiting per IP/subnet (XDP_DROP / XDP_PASS).
 * 3. 5-Tuple Flow tracking & statistical aggregation.
 * 4. High-performance event forwarding to userspace TCN-DRL Control Plane.
 */

#define KBUILD_MODNAME "xdp_ips"
#include <uapi/linux/bpf.h>
#include <linux/in.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/tcp.h>
#include <linux/udp.h>
#include "common_kern.h"

/* Map 1: Dynamic Blocked IPs (populated by DRL Agent) */
BPF_TABLE("hash", __u32, struct ip_block_entry, blocked_ips_map, 65536);

/* Map 2: Active Rate Limiters (Token Bucket) */
BPF_TABLE("hash", __u32, struct token_bucket, rate_limit_map, 65536);

/* Map 3: 5-Tuple Active Flow Tracking Table */
BPF_TABLE("hash", struct flow_key, struct flow_stats, flow_stats_map, 65536);

/* Map 4: Per-CPU Aggregated IPS Counters */
BPF_TABLE("percpu_array", __u32, struct ips_counters, metrics_map, 1);

/* Perf Event Output for Real-Time Threat Alerts */
BPF_PERF_OUTPUT(alert_events);

static inline int parse_ipv4(void *data, void *data_end, __u64 *nh_off, struct iphdr **iph) {
    struct iphdr *ip = data + *nh_off;
    if ((void *)(ip + 1) > data_end)
        return -1;
    *iph = ip;
    *nh_off += (ip->ihl * 4);
    return ip->protocol;
}

int xdp_ips_filter(struct xdp_md *ctx) {
    void *data = (void *)(long)ctx->data;
    void *data_end = (void *)(long)ctx->data_end;
    __u64 nh_off = sizeof(struct ethhdr);
    __u64 now_ns = bpf_ktime_get_ns();
    __u32 zero = 0;

    /* Retrieve per-cpu counter */
    struct ips_counters *counters = metrics_map.lookup(&zero);
    if (counters) {
        counters->rx_packets++;
    }

    /* Bound check Ethernet Header */
    struct ethhdr *eth = data;
    if ((void *)(eth + 1) > data_end)
        return XDP_PASS;

    /* Filter only IPv4 traffic for this pipeline */
    if (eth->h_proto != __constant_htons(ETH_P_IP))
        return XDP_PASS;

    /* Parse IPv4 header */
    struct iphdr *iph = NULL;
    int proto = parse_ipv4(data, data_end, &nh_off, &iph);
    if (proto < 0 || !iph)
        return XDP_PASS;

    __u32 src_ip = iph->saddr;
    __u32 dst_ip = iph->daddr;
    __u16 pkt_len = (__u16)(data_end - data);

    /* ---------------------------------------------------------
     * STAGE 1: Check Dynamic IP Blocklist (DRL Action: DROP)
     * --------------------------------------------------------- */
    struct ip_block_entry *block = blocked_ips_map.lookup(&src_ip);
    if (block) {
        if (now_ns < block->expiry_ts_ns) {
            block->drop_count++;
            if (counters) counters->dropped_packets++;
            return XDP_DROP;
        } else {
            /* Block expired: allow removal by control plane or auto-pass */
            blocked_ips_map.delete(&src_ip);
        }
    }

    /* ---------------------------------------------------------
     * STAGE 2: Check Token Bucket Rate Limiter (DRL Action: RATE_LIMIT)
     * --------------------------------------------------------- */
    struct token_bucket *tb = rate_limit_map.lookup(&src_ip);
    if (tb) {
        __u64 elapsed_ns = now_ns - tb->last_update_ns;
        /* Replenish tokens based on elapsed time and configured PPS */
        __u64 added_tokens = (elapsed_ns * tb->rate_pps) / 1000000000ULL;
        if (added_tokens > 0) {
            tb->tokens += added_tokens;
            if (tb->tokens > tb->capacity) {
                tb->tokens = tb->capacity;
            }
            tb->last_update_ns = now_ns;
        }

        if (tb->tokens >= 1) {
            tb->tokens--;
            if (counters) counters->passed_packets++;
        } else {
            /* Token exhausted: drop packet */
            if (counters) {
                counters->dropped_packets++;
                counters->rate_limited_packets++;
            }
            return XDP_DROP;
        }
    }

    /* ---------------------------------------------------------
     * STAGE 3: Extract Flow Metadata & Update Statistics
     * --------------------------------------------------------- */
    struct flow_key key = {};
    key.src_ip = src_ip;
    key.dst_ip = dst_ip;
    key.proto = (__u8)proto;

    __u8 tcp_flags = 0;
    if (proto == IPPROTO_TCP) {
        struct tcphdr *tcph = data + nh_off;
        if ((void *)(tcph + 1) <= data_end) {
            key.src_port = tcph->source;
            key.dst_port = tcph->dest;
            tcp_flags = ((__u8 *)tcph)[13];
        }
    } else if (proto == IPPROTO_UDP) {
        struct udphdr *udph = data + nh_off;
        if ((void *)(udph + 1) <= data_end) {
            key.src_port = udph->source;
            key.dst_port = udph->dest;
        }
    }

    struct flow_stats *stats = flow_stats_map.lookup(&key);
    if (stats) {
        stats->packets++;
        stats->bytes += pkt_len;
        stats->last_seen_ns = now_ns;
        stats->tcp_flags |= tcp_flags;
    } else {
        struct flow_stats initial_stats = {
            .packets = 1,
            .bytes = pkt_len,
            .first_seen_ns = now_ns,
            .last_seen_ns = now_ns,
            .tcp_flags = tcp_flags,
        };
        flow_stats_map.update(&key, &initial_stats);
    }

    if (counters) counters->passed_packets++;
    return XDP_PASS;
}
