MariHA Documentation
====================

**MariHA** is a Continual Reinforcement Learning (CRL) benchmark for Human-AI
alignment on Super Mario Bros (NES).  Agents are trained on a curriculum
derived from real human gameplay clips, and must retain past skills while
learning new scenes — mirroring the sequential nature of human learning.

MariHA extends the architecture of `COOM <https://github.com/TTomilin/COOM>`_
(a VizDoom-based CRL benchmark) to the pixel-based, human-grounded Mario
domain.  Three RL agents (SAC, PPO, DQN) share a common
:class:`~mariha.rl.base.BaseAgent` training loop, and nine CL baselines
compose onto any of them through agent-agnostic hooks.

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: Getting Started
      :link: getting_started
      :link-type: doc

      Install MariHA, pull the data, and run your first training.

   .. grid-item-card:: Architecture
      :link: architecture
      :link-type: doc

      How the modules fit together: curriculum → env → BaseAgent → CL method → eval.

   .. grid-item-card:: CL Methods
      :link: cl_methods
      :link-type: doc

      Nine continual learning baselines, composable on any RL agent.

   .. grid-item-card:: API Reference
      :link: api/index
      :link-type: doc

      Auto-generated reference for every class and function.

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: User Guide

   getting_started
   glossary
   architecture
   curriculum
   environment
   training
   cl_methods
   evaluation
   adding_an_agent
   adding_a_cl_method

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: API Reference

   api/index
