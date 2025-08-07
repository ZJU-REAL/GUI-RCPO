<h1 align="center">
  <img src="./assets/logo.png" width="40" style="vertical-align: middle; margin-right: 8px;" />
  Test-Time Reinforcement Learning for GUI Grounding via Region Consistency
</h1>

<div align="center">

<p><em>A Test-time reinforcement learning framework for GUI grounding</em></p>


</div>

---

<div align="center">
  <img src="./assets/framework.png" alt="GUI-RCPO Framework" width="80%" />
  <p><em>GUI-RC: identify the consensus region across sampling to enable more precise grounding.</em></p>
  <p><em>GUI-RCPO: transform the region consistency into rewards, and enables models to self-improve on unlabeled data.</em></p>
</div>

---

# 🎉 News

[2025-8-7] We release our paper: **Test-Time Reinforcement Learning for GUI Grounding via Region Consistency**

---

# Overview

* [Motivation](#motivation)
* [Highlights](#highlights)
* [Citation](#citation)

---

# Motivation

Current GUI grounding approaches rely heavily on large-scale pixel-level annotations and training-time optimization, which are expensive, inflexible, and difficult to scale to new domains. we observe that when GUI models generate multiple predictions, spatial overlaps across generations naturally reflect the model's localization confidence. This simple insight leads to a critical question: 

_**Can we leverage test-time computation to enhance GUI grounding performance without additional labeled data?**_

Motivated by this, we introduce GUI-RC and GUI-RCPO to unlock the untapped potential of region consistency, which enables models to self-improve without the need for labeled data.

---

# ✨ Highlights

* 🧭 **GUI-RC (Region Consistency Voting)**: Aggregates multiple sampled predictions via spatial voting to identify the consensus region—achieves **+2–3%** accuracy gains **without any training**.
* 🧠 **GUI-RCPO (Region Consistency Policy Optimization)**: Converts region consistency into self-supervised rewards for test-time reinforcement learning—enables models to iteratively **improve on unlabeled data**, reaching **+4–5%** further gains.
* 🔄 **Self-Bootstrapping Capability**: Applying GUI-RC after GUI-RCPO leads to even higher accuracy—demonstrating that our methods support **progressive self-improvement** without external supervision.
* 📊 **Robust Across Models and Benchmarks**: GUI-RC and GUI-RCPO generalize across multiple models and benchmarks, showing consistent performance boosts.


---
# 🙏 Acknowledgement

The RL Training code build from [VLM-R1 project](https://github.com/om-ai-lab/VLM-R1).

---
# 📄 Citation
