"""Definitions for the `BNSConstraints` class."""
import astropy.constants as c
import numpy as np
import logging

from mosfit.constants import C_CGS, M_SUN_CGS, KM_CGS, G_CGS
from mosfit.modules.constraints.constraint import Constraint

# G_CGS = c.G.cgs.value


class BNSConstraints(Constraint):
    """BNS constraints.

    1. M1 < Mtov
    2. M2 > 0.8 Msun
    3. 9 < R_ns < 16
    4. Causality relation relating Mtov and radius

    Free parameters are Mchirp and q. Thus for large q, M1 can be larger or M2
    smaller than allowed by NS EoS. Constraint penalises masses outside range

    Realistic EoS prevents extreme values of radius
    """

    def __init__(self, **kwargs):
        """Initialize module."""
        super(BNSConstraints, self).__init__(**kwargs)

    def process(self, **kwargs):
        """Process module. Add constraints below."""
        self._score_modifier = 0.0
        # Mass of heavier NS
        self._m1 = kwargs[self.key('M1')]
        # Mass of lighter NS
        self._m2 = kwargs[self.key('M2')]
        if 'eos' in kwargs:
            #eosfile = "/home/daniel.finstad/projects/gw170817-eft-eos/eos_data/"
            #eosfile += "2nsat/{}.dat".format(int(kwargs[self.key('eos')]))
            eosfile = "/home/daniel.finstad/projects/multimessenger_inference/"
            eosfile += "NMMA/EOS/chiralEFT_MTOV/{}.dat".format(int(kwargs[self.key('eos')]))
            _, mdat, _ = np.loadtxt(eosfile, unpack=True)
            self._m_tov = np.max(mdat)
        else:
            self._m_tov = kwargs[self.key('Mtov')]
        self._r1 = kwargs[self.key('R1')]
        self._r2 = kwargs[self.key('R2')]
        ckm = C_CGS / KM_CGS
        self._vejecta_blue = kwargs[self.key('vejecta_blue')] / ckm
        self._vejecta_red = kwargs[self.key('vejecta_red')] / ckm

        # Soft max/min, proportional to diff^2 and scaled to -100 for 0.1 Msun
        # 1
        if self._m1 > self._m_tov:
            self._score_modifier -= (100. * (self._m1-self._m_tov))**2

        # 2
        if self._m2 < 0.8:
            self._score_modifier -= (100. * (0.8-self._m2))**2

        # 3
        if self._r1 > 16:
            self._score_modifier -= (20. * (self._r1-16))**2

        if self._r1 < 9:
            self._score_modifier -= (20. * (9-self._r1))**2

        if self._r2 > 16:
            self._score_modifier -= (20. * (self._r2-16))**2

        if self._r2 < 9:
            self._score_modifier -= (20. * (9-self._r2))**2


        ## 4 enforce causality (should not be necessary for chiEFT runs)
        #Mcaus = 1/2.82 * C_CGS**2 * self._r1 * KM_CGS / G_CGS / M_SUN_CGS
        #if self._m_tov > Mcaus:
        #    self._score_modifier -= (100. * (self._m_tov-Mcaus))**2

        # 5 ensure sane ejecta velocities
        if self._vejecta_blue > 1.0:
            self._score_modifier -= 1e10
            #logging.warn("v_ejecta_blue is %s > 1! Modifying score by %s",
            #             self._vejecta_blue, self._score_modifier)
        elif self._vejecta_blue > 0.6:
            self._score_modifier -= (1e3 * (self._vejecta_blue - 0.6)) ** 2.
            #logging.warn("v_ejecta_blue is %s > 0.6! Modifying score by %s",
            #             self._vejecta_blue, self._score_modifier)
        if self._vejecta_red > 0.6:
            self._score_modifier -= (1e3 * (self._vejecta_red - 0.6)) ** 2.
        if self._vejecta_blue < 0.0:
            self._score_modifier -= 1e10
            #logging.warn("v_ejecta_blue is %s < 0! Modifying score by %s",
            #             self._vejecta_blue, self._score_modifier)
        elif self._vejecta_blue < 0.2:
            self._score_modifier -= (1e3 * (0.2 - self._vejecta_blue)) ** 2.
            #logging.warn("v_ejecta_blue is %s < 0.2! Modifying score by %s",
            #             self._vejecta_blue, self._score_modifier)

        return {self.key('score_modifier'): self._score_modifier}
