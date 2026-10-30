#!/usr/bin/env python3
"""
scripts/generate_paper_figures.py

Generates publication-quality figures for the IEEE research paper:
1. System Architecture Diagram (Data Plane vs. Control Plane)
2. Dilated Causal Convolutional Architecture (TCN Receptive Field)
3. Copies and prepares all empirical benchmark figures into paper/figures/
"""

import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def generate_architecture_figure(output_path: str):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")

    # Title Banner
    fig.patch.set_facecolor("#ffffff")

    # Kernel Space Container (Bottom half)
    kernel_rect = patches.FancyBboxPatch(
        (4, 4), 92, 22, boxstyle="round,pad=1,rounding_size=2",
        edgecolor="#0284c7", facecolor="#f0f9ff", linewidth=2.0
    )
    ax.add_patch(kernel_rect)
    ax.text(6, 23.5, "LINUX KERNEL SPACE (DATA PLANE - eBPF / XDP)", fontsize=11, fontweight="bold", color="#0369a1")

    # Userspace Container (Top half)
    user_rect = patches.FancyBboxPatch(
        (4, 32), 92, 24, boxstyle="round,pad=1,rounding_size=2",
        edgecolor="#4f46e5", facecolor="#eef2ff", linewidth=2.0
    )
    ax.add_patch(user_rect)
    ax.text(6, 53.5, "USERSPACE CONTROL PLANE (INTELLIGENCE & ORCHESTRATION)", fontsize=11, fontweight="bold", color="#3730a3")

    # Network Ingress Block
    nic_box = patches.FancyBboxPatch((6, 9), 16, 11, boxstyle="round,pad=0.5", edgecolor="#334155", facecolor="#e2e8f0", linewidth=1.5)
    ax.add_patch(nic_box)
    ax.text(14, 15.5, "Ingress Packets\n(10 GbE / NIC)", ha="center", va="center", fontsize=9, fontweight="bold")

    # XDP Driver Hook
    xdp_box = patches.FancyBboxPatch((28, 9), 20, 11, boxstyle="round,pad=0.5", edgecolor="#0284c7", facecolor="#bae6fd", linewidth=1.5)
    ax.add_patch(xdp_box)
    ax.text(38, 15.5, "XDP Hook\n(Kernel Driver)", ha="center", va="center", fontsize=9, fontweight="bold", color="#0369a1")

    # eBPF LRU Map (Fast Path)
    map_box = patches.FancyBboxPatch((54, 9), 20, 11, boxstyle="round,pad=0.5", edgecolor="#059669", facecolor="#d1fae5", linewidth=1.5)
    ax.add_patch(map_box)
    ax.text(64, 15.5, "BPF LRU Map\n(O(1) Hash Table)", ha="center", va="center", fontsize=9, fontweight="bold", color="#065f46")

    # Fast Drop / Forward
    drop_box = patches.FancyBboxPatch((80, 9), 13, 11, boxstyle="round,pad=0.5", edgecolor="#dc2626", facecolor="#fee2e2", linewidth=1.5)
    ax.add_patch(drop_box)
    ax.text(86.5, 15.5, "XDP_DROP\n(~3.4 µs)", ha="center", va="center", fontsize=9, fontweight="bold", color="#991b1b")

    # Userspace: Feature Ring Buffer
    ring_box = patches.FancyBboxPatch((8, 36), 18, 13, boxstyle="round,pad=0.5", edgecolor="#64748b", facecolor="#f1f5f9", linewidth=1.5)
    ax.add_patch(ring_box)
    ax.text(17, 43, "BPF Perf Ring\nFlow Extractor", ha="center", va="center", fontsize=9, fontweight="bold", color="#1e293b")

    # Userspace: Dilated Causal TCN
    tcn_box = patches.FancyBboxPatch((32, 36), 24, 13, boxstyle="round,pad=0.5", edgecolor="#4f46e5", facecolor="#c7d2fe", linewidth=1.5)
    ax.add_patch(tcn_box)
    ax.text(44, 43, "Dilated Causal TCN\n(Temporal Embedding $z_t$\nReceptive Field $R=31$)", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#312e81")

    # Userspace: PPO DRL Policy Agent
    drl_box = patches.FancyBboxPatch((62, 36), 20, 13, boxstyle="round,pad=0.5", edgecolor="#d97706", facecolor="#fef3c7", linewidth=1.5)
    ax.add_patch(drl_box)
    ax.text(72, 43, "PPO Policy Agent\nActor-Critic MDP\n(Action $\mathcal{A}$)", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#92400e")

    # Userspace: BPF Sync Manager
    sync_box = patches.FancyBboxPatch((86, 36), 8, 13, boxstyle="round,pad=0.5", edgecolor="#059669", facecolor="#a7f3d0", linewidth=1.5)
    ax.add_patch(sync_box)
    ax.text(90, 43, "Map\nSync", ha="center", va="center", fontsize=8, fontweight="bold", color="#065f46")

    # Connecting Arrows
    # Ingress -> XDP
    ax.annotate("", xy=(28, 14.5), xytext=(22, 14.5), arrowprops=dict(arrowstyle="->", lw=2, color="#0f172a"))
    # XDP -> Hash Table Lookup
    ax.annotate("", xy=(54, 14.5), xytext=(48, 14.5), arrowprops=dict(arrowstyle="->", lw=2, color="#0284c7"))
    # Hash Hit -> XDP_DROP
    ax.annotate("", xy=(80, 14.5), xytext=(74, 14.5), arrowprops=dict(arrowstyle="->", lw=2.5, color="#dc2626"))
    ax.text(77, 16.5, "HIT", ha="center", fontsize=8, fontweight="bold", color="#dc2626")

    # Hash Miss -> Perf Ring (Kernel to Userspace)
    ax.annotate("", xy=(17, 36), xytext=(38, 20),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#6366f1", connectionstyle="arc3,rad=-0.2", linestyle="--"))
    ax.text(24, 28, "MISS (Sample Flow)", fontsize=8, fontweight="bold", color="#4338ca")

    # Perf Ring -> TCN
    ax.annotate("", xy=(32, 42.5), xytext=(26, 42.5), arrowprops=dict(arrowstyle="->", lw=2, color="#4338ca"))
    # TCN -> PPO Agent
    ax.annotate("", xy=(62, 42.5), xytext=(56, 42.5), arrowprops=dict(arrowstyle="->", lw=2, color="#4338ca"))
    # PPO Agent -> Sync Manager
    ax.annotate("", xy=(86, 42.5), xytext=(82, 42.5), arrowprops=dict(arrowstyle="->", lw=2, color="#059669"))
    # Sync Manager -> BPF Map (Userspace to Kernel rule insertion)
    ax.annotate("", xy=(64, 20), xytext=(90, 36),
                arrowprops=dict(arrowstyle="->", lw=2.2, color="#059669", connectionstyle="arc3,rad=-0.2"))
    ax.text(82, 27, "bpf_map_update()", fontsize=8, fontweight="bold", color="#047857")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Generated System Architecture figure: {output_path}")


def generate_tcn_figure(output_path: str):
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    ax.set_xlim(-1, 17)
    ax.set_ylim(-0.5, 5.5)
    ax.axis("off")

    levels = ["Input $x_t$", "Layer 1 ($d=1$)", "Layer 2 ($d=2$)", "Layer 3 ($d=4$)", "Layer 4 ($d=8$)", "Output $z_t$"]
    dilations = [0, 1, 2, 4, 8, 0]

    # Draw level labels
    for y, label in enumerate(levels):
        ax.text(-0.8, y, label, ha="right", va="center", fontsize=9.5, fontweight="bold", color="#1e293b")

    # Draw nodes
    for y in range(5):
        for x in range(16):
            color = "#3b82f6" if y == 0 else "#6366f1"
            ax.scatter(x, y, s=120, color=color, zorder=3, edgecolors="#1e293b", linewidths=1.2)

    # Draw connections targeting node (15, 4) to illustrate receptive field
    target_x = 15
    ax.scatter(target_x, 5, s=180, color="#10b981", zorder=4, edgecolors="#065f46", linewidths=1.5)
    ax.plot([target_x, target_x], [4, 5], color="#10b981", lw=2, zorder=2)

    # Trace backward causal connections
    def trace_causal(x, layer):
        if layer == 0:
            return
        d = dilations[layer]
        # Kernel size K=3: offsets 0, -d, -2d
        offsets = [0, -d, -2 * d]
        for off in offsets:
            prev_x = x + off
            if 0 <= prev_x <= 15:
                ax.plot([prev_x, x], [layer - 1, layer], color="#f59e0b", alpha=0.8, lw=1.5, zorder=2)
                trace_causal(prev_x, layer - 1)

    trace_causal(15, 4)

    # Annotate receptive field
    ax.annotate(
        "", xy=(0, -0.3), xytext=(15, -0.3),
        arrowprops=dict(arrowstyle="<->", lw=1.8, color="#0f172a")
    )
    ax.text(7.5, -0.7, "Receptive Field $R = 1 + \sum_{l=0}^{L-1} (K-1) \cdot d_l = 31$ timesteps",
            ha="center", va="center", fontsize=10, fontweight="bold", color="#0f172a")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Generated TCN Dilated Architecture figure: {output_path}")


def main():
    paper_fig_dir = "paper/figures"
    os.makedirs(paper_fig_dir, exist_ok=True)

    generate_architecture_figure(os.path.join(paper_fig_dir, "fig1_architecture.png"))
    generate_tcn_figure(os.path.join(paper_fig_dir, "fig2_tcn_receptive_field.png"))

    # Copy benchmark plots from results/
    results_dir = "results"
    mapping = {
        "baseline_comparison.png": "fig3_baseline_comparison.png",
        "confusion_matrix.png": "fig4_confusion_matrix.png",
        "action_policy_distribution.png": "fig5_action_policy.png",
        "high_speed_benchmark.png": "fig6_high_speed_throughput.png",
        "latency_distribution.png": "fig7_latency_distribution.png",
    }

    for src_name, dst_name in mapping.items():
        src_file = os.path.join(results_dir, src_name)
        dst_file = os.path.join(paper_fig_dir, dst_name)
        if os.path.exists(src_file):
            shutil.copy2(src_file, dst_file)
            print(f"[+] Copied {src_file} -> {dst_file}")
        else:
            print(f"[!] Warning: {src_file} not found, skipping copy.")


if __name__ == "__main__":
    main()
