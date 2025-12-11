"""Pytest configuration for TwoLocusGPR.

Adds ``--jax-backend`` option to force JAX platform (cpu|gpu|tpu) and sets
``JAX_PLATFORM_NAME`` before tests run. Default is ``cpu`` for CI.
"""

import os
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--jax-backend",
        action="store",
        default="cpu",
        help="JAX platform name (cpu|gpu|tpu); sets JAX_PLATFORM_NAME",
    )


@pytest.fixture(autouse=True, scope="session")
def set_jax_backend(request):
    backend = request.config.getoption("--jax-backend")
    os.environ.setdefault("JAX_PLATFORM_NAME", backend)
    return backend
