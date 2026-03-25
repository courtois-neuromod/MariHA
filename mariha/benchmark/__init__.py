"""Algorithm-agnostic benchmark core for MariHA.

Provides the ``BenchmarkAgent`` ABC, algorithm registry, and benchmark-level
argument parsing / context construction.  All algorithm implementations should
depend on this package rather than on ``mariha.rl.sac``.
"""
