import functools
from typing import List, TypeVar

import numpy as np

from crm.base.state import State

class FormSpec:
    name: str = "form"

    density: float = 0

    shape_factor = 1
    volume_fraction_powers = np.array([3.])

    dimensionality = 1

    def __init__(self, name: str):
        self.name = name

    def solubility(self, t: float) -> float:
        """

        :param t: Temperature in degrees C
        :return: solubility in kg solute / kg solvent
        """
        raise NotImplementedError()

    def growth_rate(self, t: float, ss: float, n: np.ndarray = None, state: State = None) -> np.ndarray:
        """

        :param state:
        :param n:
        :param t: Temperature in degrees C
        :param ss: supersaturation in (c - c*) / c*
        :return: N x M array. N is the length of t and ss, M is the dimensionality. In unit of m/s
        """
        raise NotImplementedError()

    def dissolution_rate(self, t: float, ss: float, n: np.ndarray = None, state: State = None) -> np.ndarray:
        """
        see @growth_rate. This should return a negative value.
        :param state:
        :param n:
        :param t:
        :param ss:
        :return:
        """
        raise NotImplementedError()

    def nucleation_rate(self, t: float, ss: float, vf: float, state: State = None) -> np.ndarray:
        """

        :param state:
        :param t:
        :param ss:
        :param vf: volume fraction
        :return: primary and secondary nucleation rate. In unit of #/m3/s
        """
        raise NotImplementedError()

    def volume_fraction(self, n: np.ndarray):
        """
        Compute the volume fraction (vol solid/vol liquid)
        :param n:
        :return:
        """
        if n.shape[0] == 0:
            # when there is no rows
            return 0

        ret = self.particle_volume(n).sum(axis=0)

        return ret

    def particle_volume(self, n: np.ndarray):
        """
        particle volume
        :param total_volume: return the total particle volume of each size or the per-particle volume
        :param n:
        :return:
        """
        return np.prod(n[:, :-1] ** self.volume_fraction_powers, axis=1) * self.shape_factor * n[:, -1]


    def volume_average_size(self, n: np.ndarray):
        """
        Compute the volume average size that ensures the volume balance. Conventionally, only first dimension is
        adjusted. The remaining dimensions can be fixed with arbitrary method (e.g., mean size).
        This function is being used in row compression. The partition algorithm should ensure the n to be close to each
        other, so the aggregation of the non-first dimension is valid.

        :param n: (k x N) array. Standard n.
        :return: (1 x N) array. The count will be balanced.
        """
        if n.shape[0] == 0:
            return np.zeros((1, n.shape[1]))

        count = n[:, -1].sum()
        particle_average_volume = self.volume_fraction(n) / count / self.shape_factor

        if self.volume_fraction_powers.size == 1:
            # one dimensional
            first_dim_size = particle_average_volume ** (1 / self.volume_fraction_powers[0])
            return np.hstack([first_dim_size, count]).reshape((1, 2))
        else:
            # multi-dimensional N
            non_first_dim_mean_sizes = n[:, 1:-1].mean(axis=0)

            non_first_dim_prod = np.prod(non_first_dim_mean_sizes ** self.volume_fraction_powers[1:])

            # exclude the effect of the non first dimensions. The modified particle_average_volume can be used to
            # calculate the first dimension by reciprocal power of the first dimension
            particle_average_volume = particle_average_volume / non_first_dim_prod

            first_dim_size = particle_average_volume ** (1 / self.volume_fraction_powers[0])

            ret = np.hstack([first_dim_size, non_first_dim_mean_sizes, count])
            return ret.reshape((1, -1))


class ParametricFormSpec(FormSpec):
    def __init__(self, name: str, density: float, solubility_coefs: np.ndarray, g_coefs: np.ndarray,
                 g_powers: np.ndarray, d_coefs: np.ndarray, d_powers: np.ndarray, pn_coef: float, pn_power: float,
                 pn_ke: float, sn_coef: float, sn_power: float, sn_vol_power: float, g_betas: np.ndarray = None,
                 g_eas: np.ndarray = None, d_betas: np.ndarray = None, d_eas: np.ndarray = None, pn_ea: float = 0,
                 sn_ea: float = 0, shape_factor=None, volume_fraction_powers=None):
        """
        :param solubility_coefs: solubility in g solute/g solvent. solubility = poly[0] * T**0 + ...
        :param g_coefs: coefficients of growth. For nD growth, provide a shape of (N_dim, )
        :param g_powers: powers of growth. For nD growth, provide a shape of (N_dim, )
        :param d_coefs: coefficients of dissolution. For nD growth, provide a shape of (N_dim, )
        :param d_powers: powers of dissolution. For nD growth, provide a shape of (N_dim, )
        :param pn_coef: primary nucleation coefficient
        :param pn_power: primary nucleation power of supersaturation
        :param pn_ke: primary nucleation KE
        :param sn_coef: secondary nucleation coefficient
        :param sn_power: secondary nucleation power of supersaturation
        :param sn_vol_power: secondary nucleation power of volume fraction
        :param g_betas: optional size dependent growth coefficient
        :param g_eas: optional growth activation energy
        :param d_betas: optional size dependent dissolution beta
        :param d_eas: optional dissolution activation energy
        :param pn_ea: optional primary nucleation activation energy
        :param sn_ea: optional secondary nucleation activation energy
        :param shape_factor: optional shape factor
        :param volume_fraction_powers: optional volume fraction powers.
        """

        super().__init__(name)
        self.g_coefs = g_coefs
        self.dimensionality = g_coefs.shape[0]

        self.sn_vol_power = sn_vol_power
        self.sn_coef = sn_coef
        self.sn_power = sn_power
        self.pn_ke = pn_ke
        self.pn_power = pn_power
        self.pn_coef = pn_coef
        self.d_powers = d_powers
        self.d_coefs = d_coefs
        self.g_powers = g_powers
        self.solubility_coefs = solubility_coefs

        self.g_betas = g_betas or np.zeros_like(g_coefs)
        self.g_eas = g_eas or np.zeros_like(g_coefs)

        self.d_betas = d_betas or np.zeros_like(g_coefs)
        self.d_eas = d_eas or np.zeros_like(g_coefs)
        self.pn_ea = pn_ea
        self.sn_ea = sn_ea

        self.shape_factor = shape_factor
        if volume_fraction_powers is not None:
            self.volume_fraction_powers = volume_fraction_powers
        self.density = density

    def solubility(self, t: float) -> float:
        length = len(self.solubility_coefs)
        return (self.solubility_coefs * np.ones((length,)) * t ** np.arange(0, length)).sum()

    def growth_rate(self, t: float, ss: float, n: np.ndarray = None, state=None) -> np.ndarray:
        R = 8.3145
        if np.all(self.g_betas == 0):
            return self.g_coefs * ss ** self.g_powers * np.exp(-self.g_eas / R / (t + 273.15))
        else:
            return self.g_coefs * ss ** self.g_powers * (1 + self.g_betas) * n[:, :-1] * np.exp(
                -self.g_eas / R / (t + 273.15))

    def dissolution_rate(self, t: float, ss: float, n: np.ndarray = None, state=None) -> np.ndarray:
        R = 8.3145
        ss = - ss
        if np.all(self.d_betas == 0):
            return - self.d_coefs * ss ** self.d_powers * np.exp(-self.d_eas / R / (t + 273.15))
        else:
            return - self.d_coefs * ss ** self.d_powers * (1 + self.d_betas) * n[:, :-1] * np.exp(
                -self.d_eas / R / (t + 273.15))

    def nucleation_rate(self, t: float, ss: float, vf: float, state=None) -> np.ndarray:
        R = 8.3145
        tk = t + 273.15

        pn = self.pn_coef * ss ** self.pn_power * np.exp(-self.pn_ke / R / np.log(ss + 1) ** 2) * np.exp(
            -self.pn_ea / R / tk)
        sn = self.sn_coef * ss ** self.sn_power * np.exp(-self.sn_ea / R / tk) * vf ** self.sn_vol_power
        return np.array([pn, sn])


class SystemSpec:
    name: str
    forms: List[FormSpec]

    solubility_break_point = 0

    def __init__(self, name=None):
        self.name = name or self.__class__.__name__

    @staticmethod
    def supersaturation(solubility: float, concentration: float):
        return (concentration - solubility) / solubility

    def make_state(self, state_type=State, **kwargs) -> State:
        """
        Make a state with correct number of n
        :return:
        """

        state = state_type(system_spec=self, **kwargs)
        if not "n" in kwargs:
            state.n = [np.array([]).reshape((0, f.dimensionality + 1)) for f in self.forms]  # dim + 1 for count column
        else:
            assert isinstance(kwargs["n"], list), "n must be a list"
        return state

    def get_form_names(self):
        return [f.name for f in self.forms]

    def get_form_by_name(self, name):
        fm = self._form_mapping()
        return fm[name]

    @functools.lru_cache()
    def _form_mapping(self):
        return {f.name: f for f in self.forms}
