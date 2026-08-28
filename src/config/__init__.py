"""Config-class layer shared by every pipeline in this repo.

Each pipeline gets its own submodule of dataclasses (e.g. ``src.config.US``
for the US ultrasound preprocessing pipeline); stage modules import their
Config class from there instead of defining it inline.
"""
