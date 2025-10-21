# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
M31R evaluation and benchmarking package.

This is the system that answers the only question that matters:
"Does the generated Rust code compile and work?"

Subsystems:
  - benchmarks: loading and parsing benchmark tasks
  - compiler: sandboxed rustc/cargo execution
  - runner: orchestrating task execution
  - metrics: computing compile rates, pass@k, latency
  - reporting: writing structured results
  - tasks: prompt construction
"""
