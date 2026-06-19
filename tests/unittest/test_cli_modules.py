import argparse
import importlib
from argparse import ArgumentParser, Namespace
from types import SimpleNamespace

import pytest

from pyrinst import cli
from pyrinst.cli import _common


def test_cli_modules_are_importable():
    module_names = [
        "pyrinst.cli.driver",
        "pyrinst.cli.sampling",
        "pyrinst.cli.fep_eval",
        "pyrinst.cli.geom",
        "pyrinst.cli.plot",
        "pyrinst.cli.rate",
    ]

    for module_name in module_names:
        module = importlib.import_module(module_name)
        assert callable(module.main)


def test_top_level_parser_has_expected_subcommands():
    parser = cli.build_parser()
    subcommands = next(action for action in parser._actions if action.dest == "command")

    assert set(subcommands.choices) == {
        "driver",
        "fep-eval",
        "geom",
        "plot",
        "rate",
        "sample",
    }
    public_commands = {action.dest for action in subcommands._choices_actions if action.help is not argparse.SUPPRESS}
    assert public_commands == {"driver", "fep-eval", "geom", "plot", "rate", "sample"}
    help_text = parser.format_help()
    assert "optimize" not in help_text
    assert "gen-ref" not in help_text


@pytest.mark.parametrize(
    "argv",
    [
        ["--help"],
        ["geom", "--help"],
        ["rate", "--help"],
        ["sample", "--help"],
        ["fep-eval", "--help"],
        ["plot", "--help"],
        ["driver", "--help"],
    ],
)
def test_cli_help_exits_successfully(argv):
    parser = cli.build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(argv)

    assert exc_info.value.code == 0


def test_load_plugin_runs_module(monkeypatch):
    called = {}

    def fake_run_path(path):
        called["path"] = path

    monkeypatch.setattr(_common.runpy, "run_path", fake_run_path)

    _common.load_plugin("custom_pes.py")

    assert called == {"path": "custom_pes.py"}


def test_builtin_potential_argument_mapping(monkeypatch):
    captured = {}

    class DummyPotential:
        def __init__(self, symbols, **kwargs):
            captured["symbols"] = symbols
            captured["kwargs"] = kwargs

    monkeypatch.setitem(_common.POTENTIAL_REGISTRY, "mace", DummyPotential)
    monkeypatch.setattr(_common, "BUILTIN_POTENTIALS", ("mace",))
    args = Namespace(
        potential="mace",
        plugin=None,
        input_file="model.model",
        additional_files=["A"],
        working_dir="work",
    )

    potential = _common.build_potential(["H"], args)

    assert isinstance(potential, DummyPotential)
    assert captured["symbols"] == ["H"]
    assert captured["kwargs"]["template_input"] == "model.model"
    assert captured["kwargs"]["add_files"] == ["A"]
    assert captured["kwargs"]["model_paths"] == "model.model"


def test_parallel_executor_does_not_require_potential(monkeypatch):
    captured = {}

    class DummyParallelExecutor:
        def __init__(self, symbols):
            captured["symbols"] = symbols

    monkeypatch.setattr(_common, "ParallelExecutor", DummyParallelExecutor)
    args = SimpleNamespace(parallel=True, plugin=None, potential=None)

    executor = _common.build_optimization_executor(["H"], args, ArgumentParser())

    assert isinstance(executor, DummyParallelExecutor)
    assert captured == {"symbols": ["H"]}


def test_geom_parser_accepts_single_mode():
    parser = cli.build_parser()

    args = parser.parse_args(["geom", "water.xyz", "--mode", "single", "-P", "mace", "-F", "model.model"])

    assert args.command == "geom"
    assert args.mode == "single"


def test_console_script_compat_functions_exist():
    from pyrinst.cli import geom

    assert callable(geom.gen_ref_main)
    assert callable(geom.optimize_main)
