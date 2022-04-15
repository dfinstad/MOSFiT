"""Definitions for the `BNSEjecta` class."""

import numpy as np
from astrocats.catalog.source import SOURCE
# import astropy.constants as c

from mosfit.constants import FOE, KM_CGS, M_SUN_CGS, C_CGS, G_CGS
from mosfit.modules.energetics.energetic import Energetic


# G_CGS = c.G.cgs.value


class BNSEjectaGW(Energetic):
    """
    Generate `mejecta`, `vejecta` and `kappa` from neutron star binary
    parameters.

    Includes tidal and shocked dynamical and disk wind ejecta following
    Dietrich+ 2017 and Coughlin+ 2019, with opacities from Sekiguchi+ 2016,
    Tanaka+ 2019, Metzger and Fernandez 2014, Lippuner+ 2017

    Also includes an ignorance parameter `alpha` for NS-driven winds to
    increase the fraction of blue ejecta: Mdyn_blue /= alpha
     - therefore NS surface winds turned off by setting alpha = 1
    """

    _REFERENCES = [
        {SOURCE.BIBCODE: '2017CQGra..34j5014D'},
        {SOURCE.BIBCODE: '2019MNRAS.489L..91C'},
        {SOURCE.BIBCODE: '2013PhRvD..88b3007M'},
        {SOURCE.BIBCODE: '2016PhRvD..93l4046S'},
        {SOURCE.BIBCODE: '2014MNRAS.441.3444M'},
        {SOURCE.BIBCODE: '2017MNRAS.472..904L'},
        {SOURCE.BIBCODE: '2019LRR....23....1M'},
        {SOURCE.BIBCODE: '2020MNRAS.496.1369T'},
        {SOURCE.BIBCODE: '2018PhRvL.121i1102D'}
    ]

    def process(self, mass1=None, mass2=None, disk_frac=None, Mtov=None,
                Lambda1=None, Lambda2=None, alpha=None, cos_theta_open=None,
                kappa_red=None, kappa_blue=None, kappa_purple=None,
                errMdyn=None, errMdisk=None, errvdyn=None, errvdisk=None,
                radius1=None, radius2=None, radius16=None, v_ejecta_red=None,
                **kwargs):
        """Process module."""
        ckm = C_CGS / KM_CGS

        self._m1 = mass1
        self._m2 = mass2
        self._m_total = mass1 + mass2
        self._q = mass1 / mass2
        self._mchirp = ((mass1 * mass2) ** 3 / (mass1 + mass2)) ** (1. / 5)
        self._cos_theta_open = cos_theta_open
        theta_open = np.arccos(self._cos_theta_open)
        self._disk_frac = disk_frac
        self._m_tov = Mtov
        self._Lambda1 = Lambda1
        self._Lambda2 = Lambda2
        self._alpha = alpha
        self._kappa_red = kappa_red
        self._kappa_blue = kappa_blue

        self._errMdyn = errMdyn
        self._errMdisk = errMdisk

        L1 = self._Lambda1
        L2 = self._Lambda2

        self._Lambda = 16./13 * ((self._m1 + 12*self._m2) * self._m1**4 * L1 +
                (self._m2 + 12*self._m1) * self._m2**4 * L2) / self._m_total**5

        # NS radii
        self._radius_ns = radius16  # should be R_1.6 in km!
        self._R1 = radius1  # km
        self._R2 = radius2  # km

        # Compactness of each component (Maselli et al. 2013; Yagi & Yunes 2017)
        # C = (GM/Rc^2)
        C1 = G_CGS * mass1 * M_SUN_CGS / (self._R1 * 1e5 * C_CGS ** 2.)
        C2 = G_CGS * mass2 * M_SUN_CGS / (self._R2 * 1e5 * C_CGS ** 2.)

        # set up 2-param polynomial from Nedora et al. 2022
        poly2par = lambda b: b[0] + b[1] * self._q + b[2] * self._Lambda \
                   + b[3] * self._q ** 2 + b[4] * self._q * self._Lambda \
                   + b[5] * self._Lambda ** 2

        ## M0Ref + M0/M1 calibrtion
        logMejdyn = poly2par([-1.32, -0.382, -4.47e-3, -0.339, 3.21e-3, 4.31e-7])
        ## M0Ref + M0/M1 + Leak calibration
        #logMejdyn = poly2par([-6.96, 5.26, 7.84e-4, -1.71, 5.69e-4, -9.09e-7])
        Mejdyn = 10 ** logMejdyn

        Mejdyn *= self._errMdyn

        if Mejdyn < 0:
            Mejdyn = 0.0

        # Calculate fraction of ejecta with Ye<0.25 from fits to Sekiguchi 2016
        # Also consistent with Dietrich: mostly blue at M1/M2=1, all red by M1/M2=1.2.
        # And see Bauswein 2013, shocked (blue) component decreases with M1/M2
        a_4 = 14.8609
        b_4 = -28.6148
        c_4 = 13.9597
        f_ejecta_red = min([a_4*self._q**2+b_4*self._q+c_4, 1]) # fraction can't exceed 100%

        # component ejecta quantities
        mejecta_red = Mejdyn * f_ejecta_red
        mejecta_blue = Mejdyn - mejecta_red

        # mass-averaged dynamical ejecta velocity from Nedora+ 2022
        ## M0Ref + M0/M1 calibration
        vejecta = poly2par([0.594, -0.148, -8.62e-4, -5.02e-2, 3.25e-4, 3.16e-7])
        ## M0Ref + M0/M1 + Leak calibration
        #vejecta = poly2par([0.255, 0.188, -4.44e-4, -0.146, 1.87e-4, 1.38e-7])
        vejecta *= errvdyn

        # distribute velocity according to mass, neglecting any magnetic wind
        # enhancement as this likely wasn't present in simulations
        v_ejecta_blue = (vejecta * Mejdyn - mejecta_red * v_ejecta_red) / mejecta_blue
        # convert to km/s
        vejecta_red = v_ejecta_red * ckm
        vejecta_blue = v_ejecta_blue * ckm

        # Bauswein 2013, cut-off for prompt collapse to BH
        C16 = G_CGS * self._m_tov * M_SUN_CGS / (self._radius_ns * 1e5 * C_CGS ** 2.)
        Mthr = (2.38 - 3.606 * C16) * self._m_tov

        # add magnetic wind enhancement of blue ejecta
        if self._m_total < Mthr:
            mejecta_blue /= self._alpha

        # calculate disk mass using Nedora 2022
        ## M0Ref + M0/M1 calibration
        Mdisk = poly2par([-1.85, 2.59, 7.07e-4, -0.733, -8.08e-4, 2.75e-7])
        ## M0Ref + M0/M1 + Leak calibration
        #Mdisk = poly2par([-1.26, 1.76, 3.51e-4, -0.482, -5.20e-4, 3.68e-7])

        Mdisk *= self._errMdisk

        Mejdisk = Mdisk * self._disk_frac

        mejecta_purple = Mejdisk

        mejecta_tot = mejecta_blue + mejecta_red + mejecta_purple

        # Fit for disk velocity using Metzger and Fernandez
        vdisk_max = 0.1
        vdisk_min = 0.03
        vfit = np.polyfit([self._m_tov,Mthr],[vdisk_max,vdisk_min],deg=1)

        # mass-averaged Ye from Nedora 2022
        #Ye = poly2par([0.255, 3.83e-2, 2.36e-4, -6.66e-2, -1.92e-4, -1.86e-8])

        # Get average opacity of 'purple' (disk) component
        # Mass-averaged Ye as a function of remnant lifetime from Lippuner 2017
        # Lifetime related to Mtot using Metzger handbook table 3
        if self._m_total < self._m_tov:
            # stable NS
            Ye = 0.38
            vdisk = vdisk_max
        elif self._m_total < 1.2*self._m_tov:
            # long-lived (>>100 ms) NS remnant Ye = 0.34-0.38,
            # smooth interpolation
            Yfit = np.polyfit([self._m_tov,1.2*self._m_tov],[0.38,0.34],deg=1)
            Ye = Yfit[0]*self._m_total + Yfit[1]
            vdisk = vfit[0]*self._m_total + vfit[1]
        elif self._m_total < Mthr:
            # short-lived (hypermassive) NS, Ye = 0.25-0.34, smooth interpolation
            Yfit = np.polyfit([1.2*self._m_tov,Mthr],[0.34,0.25],deg=1)
            Ye = Yfit[0]*self._m_total + Yfit[1]
            vdisk = vfit[0]*self._m_total + vfit[1]
        else:
            # prompt collapse to BH, disk is red
            Ye = 0.25
            vdisk = vdisk_min
        vdisk *= errvdisk

        ## Convert Ye to opacity using Tanaka et al 2019 for Ye >= 0.25:
        #a_6 = 2112.0
        #b_6 = -2238.9
        #c_6 = 742.35
        #d_6 = -73.14

        #kappa_purple = a_6*Ye**3 + b_6*Ye**2 + c_6*Ye + d_6
        #kappa_purple = 3.0

        vejecta_purple = vdisk * ckm

        vejecta_mean = (mejecta_purple*vejecta_purple + vejecta_red*mejecta_red +
                vejecta_blue*mejecta_blue) / (mejecta_purple + mejecta_red + mejecta_blue)

        kappa_mean = (mejecta_purple*kappa_purple + self._kappa_red*mejecta_red +
                self._kappa_blue*mejecta_blue) / (mejecta_purple + mejecta_red + mejecta_blue)

        return {self.key('mejecta_blue'): mejecta_blue,
                self.key('mejecta_red'): mejecta_red,
                self.key('mejecta_purple'): mejecta_purple,
                self.key('mejecta_dyn'): Mejdyn,
                self.key('mejecta_tot'): mejecta_tot,
                self.key('vejecta_blue'): vejecta_blue,
                self.key('vejecta_red'): vejecta_red,
                self.key('vejecta_purple'): vejecta_purple,
                self.key('vejecta_mean'): vejecta_mean,
                self.key('kappa_purple'): kappa_purple,
                self.key('kappa_mean'): kappa_mean,
                self.key('M1'): self._m1,
                self.key('M2'): self._m2,
                self.key('R1'): self._R1,
                self.key('R2'): self._R2,
                self.key('radius_ns'): self._radius_ns,
                self.key('Lambda'): self._Lambda
                }

