import json
from argparse import Namespace
from .drivers import DRIVER_REGISTRY
from .proxies import CacheProxy


def get_pes(args: Namespace, atoms: list[str] = None):
    kwargs = vars(args)
    kwargs["template_input"] = kwargs["mainInputFile"]
    pes_type: str = kwargs.pop("PES")
    if pes_type.lower() in DRIVER_REGISTRY:
        return CacheProxy(DRIVER_REGISTRY[pes_type.lower()](atoms, **kwargs))
    if ":" in pes_type:
        module_name, class_name = pes_type.split(":")
        try:
            module = __import__(module_name, fromlist=[class_name])
            pes_class = getattr(module, class_name)
            with open(args.mainInputFile, "r") as f:
                main_input = json.load(f)
            if isinstance(main_input, dict):
                pes = pes_class(**main_input)
            elif isinstance(main_input, list):
                pes = pes_class(*main_input)
            else:
                raise ValueError(f"Unknown input file format: {args.mainInputFile}")
            return pes
        except (FileNotFoundError, AttributeError, ImportError) as e:
            raise ValueError(f"Failed to load custom PES '{pes_type}': {e}")
    else:
        raise ValueError(
            f"Invalid PES '{pes_type}'. Must be one of {DRIVER_REGISTRY.keys()} "
            f"OR a path in format 'path/to/file.py:ClassName'"
        )
