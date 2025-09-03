# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
M31R training infrastructure package.

Subsystems:
  - dataloader: streaming deterministic token dataloader
  - optimizer: AdamW factory
  - scheduler: cosine warmup scheduler
  - checkpoint: atomic checkpoint save/load
  - metrics: structured training metrics
  - engine: training loop
  - export: model export bundle
"""
