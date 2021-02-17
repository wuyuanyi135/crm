"""
Hypothetical crystallization kinetics for testing
"""
import numpy as np

from crm.base.system_spec import SystemSpec, FormSpec, ParametricFormSpec

__all__ = [
    "Hypothetical1D",
    "HypotheticalPolymorphic1D",
    "HypotheticalEqualGrowth2D",
    "Hypothetical2D",
    "HypotheticalPolymorphicEqualGrowth2D",
    "HypotheticalPolymorphic2D"
]


class Hypothetical1D(SystemSpec):
    def __init__(self, name=None):
        super().__init__(name)

        self.forms = [ParametricFormSpec(
            name="alpha",
            density=1.54,
            solubility_coefs=np.array([4.564e-3, 3.032e-5, 8.437e-6]),
            g_coefs=np.array([0.1e-6]),
            g_powers=np.array([1]),
            d_coefs=np.array([2.2e-6]),
            d_powers=np.array([1]),
            pn_coef=1e8,
            pn_power=2,
            pn_ke=0,
            sn_coef=1e10,
            sn_power=2,
            sn_vol_power=2 / 3,
            shape_factor=0.48,
        )]


class HypotheticalPolymorphic1D(SystemSpec):

    def __init__(self, name=None):
        super().__init__(name)

        self.forms = [
            ParametricFormSpec(
                name="alpha",
                density=1.54,
                solubility_coefs=np.array([4.564e-3, 3.032e-5, 8.437e-6]),
                g_coefs=np.array([0.1e-6]),
                g_powers=np.array([1]),
                d_coefs=np.array([2.2e-6]),
                d_powers=np.array([1]),
                pn_coef=1e8,
                pn_power=2,
                pn_ke=0,
                sn_coef=1e10,
                sn_power=2,
                sn_vol_power=2 / 3,
                shape_factor=0.48,
            ),
            ParametricFormSpec(
                name="beta",
                density=1.54,
                solubility_coefs=np.array([6.222e-3, -1.165e-4, 7.644e-6]),
                g_coefs=np.array([0.1e-6]),
                g_powers=np.array([1]),
                d_coefs=np.array([2.2e-6]),
                d_powers=np.array([1]),
                pn_coef=1e7,
                pn_power=2,
                pn_ke=0,
                sn_coef=1e10,
                sn_power=2,
                sn_vol_power=2 / 3,
                shape_factor=0.48,
            )
        ]


class HypotheticalAgg1D(Hypothetical1D):

    def __init__(self, name=None, min_count=None, compression_interval=None):
        super().__init__(name)
        self.forms[0].agg_kernel = 4.86e12
        if min_count is not None:
            self.forms[0].min_count = min_count
        if compression_interval is not None:
            self.forms[0].compression_interval = compression_interval


class HypotheticalBrk1D(Hypothetical1D):
    def __init__(self, name=None, min_count=None, compression_interval=None):
        super().__init__(name)
        self.forms[0].brk_kernel = np.array([
            (0.5, 4.86e15)
        ])
        if min_count is not None:
            self.forms[0].min_count = min_count
        if compression_interval is not None:
            self.forms[0].compression_interval = compression_interval


class HypotheticalAggBrk1D(HypotheticalAgg1D, HypotheticalBrk1D):

    def __init__(self, name=None, min_count=None, compression_interval=None):
        super().__init__(name, min_count, compression_interval)


class HypotheticalEqualGrowth2D(SystemSpec):
    """
    Equivalent to the 1D growth
    """

    def __init__(self, name=None):
        super().__init__(name)

        self.forms = [ParametricFormSpec(
            name="alpha",
            density=1.54,
            solubility_coefs=np.array([4.564e-3, 3.032e-5, 8.437e-6]),
            g_coefs=np.array([0.1e-6, 0.1e-6]),
            g_powers=np.array([1, 1]),
            d_coefs=np.array([2.2e-6, 2.2e-6]),
            d_powers=np.array([1, 1]),
            pn_coef=1e8,
            pn_power=2,
            pn_ke=0,
            sn_coef=1e10,
            sn_power=2,
            sn_vol_power=2 / 3,
            shape_factor=0.48,
            volume_fraction_powers=np.array([2, 1])
        )]


class Hypothetical2D(SystemSpec):

    def __init__(self, name=None):
        super().__init__(name)

        self.forms = [ParametricFormSpec(
            name="alpha",
            density=1.54,
            solubility_coefs=np.array([4.564e-3, 3.032e-5, 8.437e-6]),
            g_coefs=np.array([0.1e-6, 0.15e-6]),
            g_powers=np.array([1, 1.2]),
            d_coefs=np.array([2.2e-6, 2.1e-6]),
            d_powers=np.array([1, 1.1]),
            pn_coef=1e8,
            pn_power=2,
            pn_ke=0,
            sn_coef=1e10,
            sn_power=2,
            sn_vol_power=2 / 3,
            shape_factor=0.48,
            volume_fraction_powers=np.array([2, 1])
        )]


class HypotheticalPolymorphicEqualGrowth2D(SystemSpec):

    def __init__(self, name=None):
        super().__init__(name)

        self.forms = [
            ParametricFormSpec(
                name="alpha",
                density=1.54,
                solubility_coefs=np.array([4.564e-3, 3.032e-5, 8.437e-6]),
                g_coefs=np.array([0.1e-6, 0.1e-6]),
                g_powers=np.array([1, 1]),
                d_coefs=np.array([2.2e-6, 2.2e-6]),
                d_powers=np.array([1, 1]),
                pn_coef=1e8,
                pn_power=2,
                pn_ke=0,
                sn_coef=1e10,
                sn_power=2,
                sn_vol_power=2 / 3,
                shape_factor=0.48,
                volume_fraction_powers=np.array([2, 1])
            ),
            ParametricFormSpec(
                name="beta",
                density=1.54,
                solubility_coefs=np.array([6.222e-3, -1.165e-4, 7.644e-6]),
                g_coefs=np.array([0.1e-6, 0.1e-6]),
                g_powers=np.array([1, 1]),
                d_coefs=np.array([2.2e-6, 2.2e-6]),
                d_powers=np.array([1, 1]),
                pn_coef=1e7,
                pn_power=2,
                pn_ke=0,
                sn_coef=1e10,
                sn_power=2,
                sn_vol_power=2 / 3,
                shape_factor=0.48,
                volume_fraction_powers=np.array([2, 1])
            )
        ]


class HypotheticalPolymorphic2D(SystemSpec):

    def __init__(self, name=None):
        super().__init__(name)

        self.forms = [
            ParametricFormSpec(
                name="alpha",
                density=1.54,
                solubility_coefs=np.array([4.564e-3, 3.032e-5, 8.437e-6]),
                g_coefs=np.array([0.1e-6, 0.15e-6]),
                g_powers=np.array([1, 1.2]),
                d_coefs=np.array([2.2e-6, 1.9e-6]),
                d_powers=np.array([1, 1]),
                pn_coef=1e8,
                pn_power=2,
                pn_ke=0,
                sn_coef=1e10,
                sn_power=2,
                sn_vol_power=2 / 3,
                shape_factor=0.48,
                volume_fraction_powers=np.array([2, 1])
            ),
            ParametricFormSpec(
                name="beta",
                density=1.54,
                solubility_coefs=np.array([6.222e-3, -1.165e-4, 7.644e-6]),
                g_coefs=np.array([0.1e-6, 0.18e-6]),
                g_powers=np.array([1, 1.1]),
                d_coefs=np.array([2.2e-6, 2.4e-6]),
                d_powers=np.array([1, 1]),
                pn_coef=1e7,
                pn_power=2,
                pn_ke=0,
                sn_coef=1e10,
                sn_power=2,
                sn_vol_power=2 / 3,
                shape_factor=0.48,
                volume_fraction_powers=np.array([2, 1])
            )
        ]
