import importlib


def test_cli_modules_are_importable():
    module_names = [
        "pyrinst.cli.gen_ref",
        "pyrinst.cli.sampling",
        "pyrinst.cli.fep_eval",
        "pyrinst.cli.optimize",
    ]

    for module_name in module_names:
        module = importlib.import_module(module_name)
        assert callable(module.main)
