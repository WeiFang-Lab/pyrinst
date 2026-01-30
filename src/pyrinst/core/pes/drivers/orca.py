from .abc import OnTheFlyDriver


class Orca(OnTheFlyDriver):
    _runcmd: str = "orca"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._grad_cmd: str = "EnGrad"
        self._hess_cmd: str = "Freq"
        self._input: str = f"{self._sys_name}.inp"
        self._output: str = f"{self._sys_name}.out"
        self.args: str = f"{self._input} > {self._output}"
