"""Agent-agnostic benchmark core for MariHA.

Provides the ``BenchmarkAgent`` ABC, agent registry, and benchmark-level
argument parsing / context construction.  All agent implementations should
depend on this package rather than on ``mariha.rl.sac``.
"""
