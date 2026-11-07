"""Shared, explicit configuration handling for the rebuttal experiments."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load(path):
    path = Path(path).expanduser().resolve()
    with path.open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError(f"Unsupported configuration: {path}")
    config["_config_path"] = str(path)
    return config


def path(value):
    value = os.path.expandvars(str(value))
    if "$" in value:
        raise ValueError(f"Set the environment variable used in path: {value}")
    result = Path(value).expanduser()
    return result if result.is_absolute() else ROOT / result


def schedule(config, rows, world_size=None):
    train = config["training"]
    world = world_size or len(config["resources"]["gpus"])
    micro = int(train["per_device_batch_size"])
    effective = int(train["global_batch_size"])
    if effective % (world * micro):
        raise ValueError("Global batch must be divisible by world size * local batch")
    if rows <= 0 or rows % effective:
        raise ValueError("Prepared training rows must be a positive multiple of global batch")
    accumulation = effective // (world * micro)
    steps_per_epoch = rows // (world * micro)
    total_micro_steps = steps_per_epoch * int(train["epochs"])
    if train.get("max_optimizer_steps"):
        total_micro_steps = min(total_micro_steps, int(train["max_optimizer_steps"]) * accumulation)
    return {
        "world_size": world,
        "gradient_accumulation_steps": accumulation,
        "steps_per_epoch": steps_per_epoch,
        "total_micro_steps": total_micro_steps,
        "total_optimizer_steps": total_micro_steps // accumulation,
        # The native D3D scheduler and save counter both use micro-steps.
        "warmup_micro_steps": max(1, math.ceil(total_micro_steps * train["warmup_ratio"])),
        "warmup_optimizer_steps": max(1, math.ceil(total_micro_steps / accumulation * train["warmup_ratio"])),
    }


def data_metadata(config):
    meta_path = path(config["data"]["metadata"])
    with meta_path.open() as stream:
        meta = json.load(stream)
    return meta


def validate_training(config):
    if config.get("kind") != "training":
        raise ValueError("Expected kind=training")
    if config["method"] not in {"d3d", "ar", "pard2"}:
        raise ValueError("Unknown training method")
    gpu_ids = config["resources"]["gpus"]
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("GPU IDs must be nonempty and distinct")
    if config["method"] == "ar" and config["training"]["ar_loss_weight"] != 1:
        raise ValueError("The AR comparator uses response-only CE with unit weight")
    if config["training"]["optimizer"] not in {"adamw", "adamw8bit"}:
        raise ValueError("Unknown optimizer")
    schedule(config, 16)
