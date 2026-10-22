from setuptools import find_packages, setup

setup(
    name="ebpf-rl-tcn-ips",
    version="1.0.0",
    description="An eBPF-Driven Deep Reinforcement Learning and TCN Framework for Intrusion Prevention",
    author="IPS Research Team",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "gymnasium>=0.29.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "streamlit>=1.30.0",
    ],
)
