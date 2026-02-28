import json
from argparse import Namespace


def get_pes(args: Namespace, atoms: list[str] = None):
    kwargs = vars(args)
    kwargs["template_input"] = kwargs["mainInputFile"]
    pes_type: str = kwargs["PES"]
    if ":" in pes_type:
        module_name, class_name = pes_type.split(":")
        try:
            module = __import__(module_name, fromlist=[class_name])
            pes_class = getattr(module, class_name)
            if args.mainInputFile is None:
                return pes_class()
            with open(args.mainInputFile) as f:
                main_input = json.load(f)
            if isinstance(main_input, dict):
                return pes_class(**main_input)
            elif isinstance(main_input, list):
                return pes_class(*main_input)
            else:
                raise ValueError(f"Unknown input file format: {args.mainInputFile}")
        except (FileNotFoundError, AttributeError, ImportError) as e:
            raise ValueError(f"Failed to load custom PES '{pes_type}': {e}") from e
    else:
        raise ValueError()
