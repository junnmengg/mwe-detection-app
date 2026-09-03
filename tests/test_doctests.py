"""Run the doctests embedded in the dependency-free modules.

Executing them from a real test module keeps `--doctest-modules` out of
`addopts`. That flag walks every collected file, including the test modules
themselves, which makes pytest collect each one twice and report duplicates.
"""

from __future__ import annotations

import doctest
from types import ModuleType

import pytest

import config
import text_processing


@pytest.mark.parametrize("module", [config, text_processing], ids=lambda module: module.__name__)
def test_module_doctests(module: ModuleType) -> None:
    results = doctest.testmod(module, verbose=False)
    assert results.failed == 0, f"{results.failed} doctest(s) failed in {module.__name__}"


def test_doctests_actually_exist() -> None:
    """Guard against the doctests silently disappearing from the modules."""
    finder = doctest.DocTestFinder()
    found = sum(
        len(test.examples) for module in (config, text_processing) for test in finder.find(module)
    )
    assert found > 0
