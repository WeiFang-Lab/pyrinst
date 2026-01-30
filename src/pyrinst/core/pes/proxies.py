from numpy.typing import NDArray

from .abc import PES
from .drivers import OnTheFlyDriver, OnTheFlyResult
from pyrinst.utils.elements import element_data


class DriverProxy(PES):
    def __init__(self, driver: OnTheFlyDriver):
        for name in dir(driver):
            if not name.startswith("_"):
                if name not in dir(self):
                    setattr(self, name, getattr(driver, name))
        self._driver = driver
        self.atoms = driver.atoms
        self.mass = element_data.get_masses(self.atoms)

    def potential(self, x: NDArray) -> float:
        return self._driver.compute(x, calc_grad=False, calc_hess=False).energy

    def gradient(self, x: NDArray) -> NDArray:
        return self._driver.compute(x, calc_grad=True, calc_hess=False).grad

    def hessian(self, x: NDArray) -> NDArray:
        return self._driver.compute(x, calc_grad=True, calc_hess=True).hess

    def both(self, x: NDArray) -> tuple[float, NDArray]:
        res: OnTheFlyResult = self._driver.compute(x, calc_grad=True, calc_hess=False)
        return res.energy, res.grad

    def all(self, x: NDArray) -> tuple[float, NDArray, NDArray]:
        res: OnTheFlyResult = self._driver.compute(x, calc_grad=True, calc_hess=True)
        return res.energy, res.grad, res.hess


class CacheProxy(DriverProxy):
    def __init__(self, driver: OnTheFlyDriver):
        super().__init__(driver)
        self._energy = {}
        self._grad = {}
        self._hess = {}

    def potential(self, x: NDArray) -> float:
        hash_x = hash(x.tobytes())
        if hash_x in self._energy:
            return self._energy[hash_x]
        else:
            res = super().potential(x)
            self._energy[hash_x] = res
            return res

    def gradient(self, x: NDArray) -> NDArray:
        return self.both(x)[1]

    def hessian(self, x: NDArray) -> NDArray:
        return self.all(x)[2]

    def both(self, x: NDArray) -> tuple[float, NDArray]:
        hash_x = hash(x.tobytes())
        if hash_x in self._grad:
            return self._energy[hash_x], self._grad[hash_x]
        else:
            res = super().both(x)
            self._energy[hash_x], self._grad[hash_x] = res
            return res

    def all(self, x: NDArray) -> tuple[float, NDArray, NDArray]:
        hash_x = hash(x.tobytes())
        if hash_x in self._hess:
            return self._energy[hash_x], self._grad[hash_x], self._hess[hash_x]
        else:
            res = super().all(x)
            self._energy[hash_x], self._grad[hash_x], self._hess[hash_x] = res
            return res
