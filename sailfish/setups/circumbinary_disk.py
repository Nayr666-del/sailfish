"""
2D disk setups for binary problems.
"""

from math import sqrt, exp, pi, floor
import pickle as pk
from sailfish.mesh import LogSphericalMesh, PlanarCartesian2DMesh
from sailfish.physics.circumbinary import (
    EquationOfState,
    PointMass,
    SinkModel,
    ViscosityModel,
)
from sailfish.physics.kepler import OrbitalElements, OrbitalState
import cooling
from sailfish.physics.Peters_Inspiral import Orbital_Inspiral
from sailfish.setup_base import SetupBase, SetupError, param
import os
import numpy as np


pi = 3.14159265359
cgs = dict(
		G = 6.6725985e-8,
		c = 2.99792458e10,
		kb = 1.38065812e-16,
		sigmab = 5.6705119e-5,
		mp = 1.6726e-24,
		kappa = 0.4,            # electron scattering
		pc = 3.085678e18,
		msun = 1.989e33,
		h = 6.62607015e-27,
		blackbodyconst1 = 1.4745e-47,
		blackbodyconst2 = 4.79921e-11,
		c2h3 = 2.61463e-58,
		h_over_kb = 4.79921e-11,
		year=31556952
	)


class CoolInspiral(SetupBase):
    r"""
    A circumbinary disk setup for GW-inspiralling binaries.
    """

    eos                   = param("gamma-law", "EOS type: either isothermal or gamma-law")
    domain_radius         = param(10.0, "half side length of the square computational domain")
    mass_ratio            = param(1.0, "component mass ratio m2 / m1 <= 1", mutable=True)
    sink_rate             = param(1.0, "component sink rate", mutable=True)
    sink_radius           = param(0.04, "component sink radius", mutable=True)
    softening_length      = param(0.04, "gravitational softening length", mutable=True)
    buffer_is_enabled     = param(True, "whether the buffer zone is enabled", mutable=True)
    sink_model            = param("torque_free", "sink [acceleration_free|force_free|torque_free]", mutable=True)
    alpha                 = param(0.1, "alpha-viscosity parameter (gamma-law)")
    nu                    = param(0.001, "kinematic viscosity parameter (isothermal)")
    gamma_law_index_gas   = param(5.0 / 3.0, "adiabatic index (gamma-law)")
    constant_softening    = param(True, "whether to use constant softening (gamma-law)")
    retrograde            = param(False, "is disk retrograde?")
    which_diagnostics     = param("none", "diagnostics set to get from solver [none|mdots]")
    vkick                 = param(530.0, "velocity of the kick given to the merger remnant in km/s",mutable=True)
    epsilon               = param(0.05, "fraction of merger remnant lost through gw emission",mutable=True)
    direction             = param(1, "Direction of kick labelled by integers 1234.")
    # Cooling specific parameters
    central_mass_msun     = param(1e6, "Mass of the central object in solar masses")
    semimajoraxis_pc      = param(None, "Length scale in parsecs")
    mach_number_a         = param(10, "Disk Mach number") 
    target_accretion_rate = param(1., "Fraction of Eddington the disk we remap to in post-processing", mutable=True) 
    beta                  = param(1., "Gas pressure fraction P_gas/P_tot where P_tot = P_gas+P_rad") 
    # Inspiral specific parameters
    init_separation_rg    = param(50, "initial semi-major axis in grav-radii")
    init_eccentricity     = param(0.0, "orbital eccentricity of the binary")
    inspiral_start_time   = param(1000., "how many orbits before inspiral starts")
    integration_timestep  = param(0.001, "timestep for integrating the inspiral")
    semi_major_axis_list  = param([]," List of all semi-major axes over the inspiral")
    eccentricity_list     = param([]," List of all eccentricities axes over the inspiral")
    inspiral_time_list    = param([]," List of all eccentricities axes over the inspiral")
    gw_inspiral_time      = param(0.," The circular inspiral time for a0 = 1 ")
    Eccentric_Anomalies   = param([]," Find the true anomaly given the mean anomaly")
    OpticalDepthFloor     = param(0., "Minimum optical depth to measure lightcurves", mutable=True) 
    # Buffer parameters
    buffer_timestep       = param(0.001, "time step for integrating the shocked profile")
    buffer_sigma_list     = param([], "List of perturbed density profile at the outer edge")
    buffer_pressure_list  = param([], "List of perturbed pressure profile at the outer edge")

    # Code units
    a0 = 1.0
    GM = 1.0

    @property
    def Gravitational_Radius_pc(self):
        from cooling import cgs
        return cgs['G'] * self.central_mass_msun * cgs['msun'] / cgs['c'] / cgs['c'] / cgs['pc']

    @property
    def length_scale_pc(self):
        r_g = self.Gravitational_Radius_pc
        if self.semimajoraxis_pc is None:
            return r_g * self.init_separation_rg  
        else:
            return self.length_scale_pc

    @property
    def is_isothermal(self):
        return self.eos == "isothermal"

    @property
    def is_gamma_law(self):
        return self.eos == "gamma-law"

    @property
    def gamma_law_index(self):
        #return self.beta + (4-3*self.beta)**2 * (self.gamma_law_index_gas-1) / ( self.beta + 12 * (self.gamma_law_index_gas-1) * (1-self.beta) )
        return cooling.gamma_law_index(self.beta, self.gamma_law_index_gas)

    @property
    def SS73(self):
        SS73_Setup = cooling.ShakuraSunyaevDisk(
            central_mass_msun = self.central_mass_msun, 
            length_scale_pc   = self.length_scale_pc,
            mach_number_a     = self.mach_number_a,
            alpha             = self.alpha,
            gamma             = self.gamma_law_index
            )
        return SS73_Setup

#Ryan: Returns density and pressure at a

    @property
    def initial_density(self):
        sigma = self.SS73.surface_density_coefficient
        return sigma
    # Functions to return profiles at r = a
    @property
    def initial_pressure(self):
        pressure = self.SS73.surface_pressure_coefficient
        return pressure

    @property
    def AccretionRateRescaling(self):
        return self.target_accretion_rate / self.SS73._eddington_fraction

    @property
    def cooling_coefficient(self):
        CoolingCoefficient = self.SS73.cooling_coefficient
        return CoolingCoefficient


    def primitive(self, t, coords, primitive):
        x, y       = coords
        r          = sqrt(x * x + y * y)
        r_softened = sqrt(x * x + y * y + self.softening_length * self.softening_length)
        phi_hat_x  = -y / max(r, 1e-12)
        phi_hat_y  = +x / max(r, 1e-12)

        sign = 1.0
        if self.retrograde == True:
                sign = -1.

        if self.is_isothermal:
            primitive[0] = 1.0
            primitive[1] = sqrt(self.GM / r_softened) * phi_hat_x * sign
            primitive[2] = sqrt(self.GM / r_softened) * phi_hat_y * sign

        elif self.is_gamma_law:
            sigma    = self.SS73.surface_density_profile(r_softened)
            pressure = self.SS73.surface_pressure_profile(r_softened)

            primitive[0] = sigma * (0.0001 + 0.9999 * exp(-((1.0 / r_softened) ** 30)))
            primitive[1] = sign  * sqrt(self.GM / r_softened) * phi_hat_x
            primitive[2] = sign  * sqrt(self.GM / r_softened) * phi_hat_y
            primitive[3] = pressure * (0.0001 + 0.9999 * exp(-((1.0 / r_softened) ** 30)))
            

    def mesh(self, resolution):
        return PlanarCartesian2DMesh.centered_square(self.domain_radius, resolution)

    @property
    def default_resolution(self):
        return 3000

    @property
    def physics(self):
        if self.is_isothermal:
            return dict(
                eos_type=EquationOfState.LOCALLY_ISOTHERMAL,
                mach_number=self.mach_number_3a,
                point_mass_function=self.point_masses,
                buffer_is_enabled=self.buffer_is_enabled,
                buffer_driving_rate=100.0,
                buffer_onset_width=1.0,
                cooling_coefficient=0.0,
                constant_softening=self.constant_softening,
                viscosity_model=ViscosityModel.CONSTANT_NU if self.nu > 0.0 else ViscosityModel.NONE,
                viscosity_coefficient=self.nu,
                alpha=0.0,
                diagnostics=self.diagnostics,
                retrograde=self.retrograde,
            )
# Ryan: added initial pressure and density
        elif self.is_gamma_law:
            return dict(
                eos_type=EquationOfState.GAMMA_LAW,
                gamma_law_index=self.gamma_law_index,
                point_mass_function=self.point_masses,
                buffer_is_enabled=self.buffer_is_enabled,
                buffer_driving_rate=1000.0,  # default value in circumbinary.py
                buffer_onset_width=0.1,  # default value in circumbinary.py
                cooling_coefficient=self.cooling_coefficient,
                constant_softening=self.constant_softening,
                viscosity_model=ViscosityModel.CONSTANT_ALPHA if self.alpha > 0.0 else ViscosityModel.NONE,
                viscosity_coefficient=0.0,
                alpha=self.alpha,
                diagnostics=self.diagnostics,
                retrograde=self.retrograde,
                # INITIAL_SIGMA=self.initial_density,
                # INITIAL_PRESSURE=self.initial_pressure,
                # tmerge=self.InsStart,
                # vxkick=self.kick_speed_direction_x,
                # vykick=self.kick_speed_direction_y,
            )
# Added for ease of computing inspiral start time

    @property
    def diagnostics(self):
        if self.which_diagnostics == "david":
            return [
                dict(quantity="time"),
                dict(quantity="semimajor-axis"),
                dict(quantity="eccentricity"),
                dict(quantity="density_floor"),
                dict(quantity="pressure_floor"),
                dict(quantity="Accreted_energy"),
                dict(quantity="optical"),
                dict(quantity="infared"),
                dict(quantity="bolometric"),
                dict(quantity="uncounted_cells_in_lc"),
                dict(quantity="mdot", which_mass=1, accretion=True),
                dict(quantity="mdot", which_mass=2, accretion=True),
                dict(quantity="torque", which_mass='both', gravity=True),
                dict(quantity="torque", which_mass='both', accretion=True),
                dict(quantity="power" ,which_mass=1,gravity=True),
                dict(quantity="power" ,which_mass=2,gravity=True),
                dict(quantity="power" ,which_mass=1,accretion=True),
                dict(quantity="power" ,which_mass=2,accretion=True),
                dict(quantity="angular_momentum"),
                dict(quantity="uv"),
                dict(quantity="xray"),
                dict(quantity="max_temperature"),
                
                #dict(quantity="eccentricity_vector", radial_cut=(1.0, 6.0)),
                #dict(quantity="torque",which_mass='both',gravity=True, radial_cut=(0.0, 1.0)),
                #dict(quantity="torque",which_mass='both',gravity=True, radial_cut=(1.0, 10.0)),
                #dict(quantity="power",which_mass=2,gravity=True, radial_cut=(0.0, 10.0)),
                #dict(quantity="power",which_mass=2,accretion=True, radial_cut=(0.0, 10.0)),
                #dict(quantity="power",which_mass=1,gravity=True, radial_cut=(0.0, 1.0)),
                #dict(quantity="power",which_mass=1,gravity=True, radial_cut=(1.0, 10.0)),
                #dict(quantity="power",which_mass=2,gravity=True, radial_cut=(0.0, 1.0)),
                #dict(quantity="power",which_mass=2,gravity=True, radial_cut=(1.0, 10.0)),
            ]
        if  self.which_diagnostics == 'ryan':
            return [
                dict(quantity="time"),
                dict(quantity="semimajor-axis"),
                dict(quantity="eccentricity"),
                dict(quantity="density_floor"),
                dict(quantity="pressure_floor"),
                dict(quantity="Accreted_energy"),
                dict(quantity="optical"),
                dict(quantity="infared"),
                dict(quantity="bolometric"),
                dict(quantity="uncounted_cells_in_lc"),
                dict(quantity="mdot", which_mass=1, accretion=True),
                dict(quantity="mdot", which_mass=2, accretion=True),
                dict(quantity="uv"),
                dict(quantity="xray"),
                dict(quantity="max_temperature"),
            ]

        else:
            return [
                dict(quantity="time"),
                dict(quantity="mdot", which_mass=1, accretion=True),
                dict(quantity="mdot", which_mass=2, accretion=True),
            ]

    from math import sqrt
    @property
    def solver(self):
        if self.is_isothermal:
            return "cbdiso_2d"
        elif self.is_gamma_law:
            return "cbdgam_2d"

    @property
    def boundary_condition(self):
        return "outflow"

    @property
    def reference_time_scale(self):
        return 2.0 * pi
    
    @property
    def Omega_0(self):
        return sqrt(self.GM/self.a0/self.a0/self.a0)
    
    @property
    def Phase_at_Start(self):
        return 0. # correct this later.... 

    @property
    def code_start_inspiral_time(self):
        return self.inspiral_start_time * self.reference_time_scale
    
    @property
    def MergeStart(self):
        return self.inspiral_time_list[-1] + self.code_start_inspiral_time
    
    @property
    def kick_speed(self):
        c_code         = np.sqrt(self.init_separation_rg)
        Ratio_v_over_c = (self.vkick * 1e5) / cgs['c']
        return Ratio_v_over_c * c_code

    # Ryan: The direction of the kick refers to the blackhole. i.e. in direction 2, BH is kicked in
    # positive x, so gas is kicked in negative x
    @property
    def kick_speed_direction_x(self):
        if ((self.direction == 1) or (self.direction == 3)):
            return 0.0
        elif self.direction == 4:
            return self.kick_speed
        elif self.direction == 2:
            return -1.0 * self.kick_speed
        else:
            return 0.0
        
    @property
    def kick_speed_direction_y(self):
        if ((self.direction == 2) or (self.direction == 4)):
            return 0.0
        elif self.direction == 1:
            return self.kick_speed
        elif self.direction == 3:
            return -1.0 * self.kick_speed
        else:
            return 0.0

    def check_if_inspiral(self, time):
        if (time <= self.code_start_inspiral_time):
            return 'Burn-in'
        elif (self.code_start_inspiral_time <= time <= self.code_start_inspiral_time + self.inspiral_time_list[-1]):
            return 'Inspiralling'
        elif (self.inspiral_time_list[-1] + self.code_start_inspiral_time <= time):
            return 'Merged'
    # Ryan: Interpolating the Buffer shocking condition
    def Buffer_Conditions_During_Inspiral(self,time):
        Merge_t = time - self.code_start_inspiral_time - self.gw_inspiral_time[-1]
        Post_Merge_Progress = Merge_t/self.integration_timestep
        Nstep = floor(Post_Merge_Progress)
        Position_in_Bracket_N0_N1 = Post_Merge_Progress - float(Nstep)

        Density_N0 = self.buffer_sigma_list[Nstep]
        Pressure_N0 = self.buffer_pressure_list[Nstep]

        Density_N1 = self.buffer_sigma_list[Nstep+1]
        Pressure_N1 = self.buffer_pressure_list[Nstep+1]

        Interpolated_Density = Density_N0 + Position_in_Bracket_N0_N1 * (Density_N1 - Density_N0)
        Interpolated_Pressure = Pressure_N0 + Position_in_Bracket_N0_N1 * (Pressure_N1 - Pressure_N0)

        return [Interpolated_Density, Interpolated_Pressure]
    
    def buffer_terms(self,time):
        if self.check_if_inspiral(time) == 'Merged':
            return self.Buffer_Conditions_During_Inspiral(time)
        else:
            return "Normal"




    def Orbital_Elements_During_Inspiral(self, time):
        if time < self.code_start_inspiral_time:
            raise ValueError("Orbital_Elements_During_Inspiral has been called before inspiral is set to occur")

        Inspiral_t                = time - self.code_start_inspiral_time
        Inspiral_Progress         = Inspiral_t/self.integration_timestep
        Nstep                     = floor(Inspiral_Progress)
        Position_in_Bracket_N0_N1 = Inspiral_Progress - float(Nstep)

        self.semi_major_axis_list.append(1e-5)
        self.eccentricity_list.append(1e-5)
        self.Eccentric_Anomalies.append(1e-5)
        
        try:
            SemiMajorAxis_N0 = self.semi_major_axis_list[Nstep]
            Eccentricity_N0  = self.eccentricity_list[Nstep]
            EcctricPhase_N0  = self.Eccentric_Anomalies[Nstep]

            SemiMajorAxis_N1 = self.semi_major_axis_list[Nstep+1]
            Eccentricity_N1  = self.eccentricity_list[Nstep+1]
            EcctricPhase_N1  = self.Eccentric_Anomalies[Nstep+1]

            Interpolated_SMA   = SemiMajorAxis_N0 + Position_in_Bracket_N0_N1 * (SemiMajorAxis_N1 - SemiMajorAxis_N0)
            Interpolated_ECC   = Eccentricity_N0  + Position_in_Bracket_N0_N1 * (Eccentricity_N1  - Eccentricity_N0)
            Interpolated_Anom  = EcctricPhase_N0  + Position_in_Bracket_N0_N1 * (EcctricPhase_N1  - EcctricPhase_N0)

            #Phase_at_Start     = self.Omega_0 * self.code_start_inspiral_time
            return [Interpolated_SMA , Interpolated_ECC , Interpolated_Anom+self.Phase_at_Start]
        
        except IndexError as e:
            return 'Merged'



    def orbital_elements(self, time):
        flag = self.check_if_inspiral(time)

        if flag == 'Burn-in':
            return OrbitalElements(
                semimajor_axis=self.a0,
                total_mass=1.0,
                mass_ratio=self.mass_ratio,
                eccentricity=self.init_eccentricity)

        elif flag =='Inspiralling':
            Inspiralling_Orbital_Elements = self.Orbital_Elements_During_Inspiral(time)
            return OrbitalElements(
                    semimajor_axis=Inspiralling_Orbital_Elements[0],
                    total_mass=1.0,
                    mass_ratio=self.mass_ratio,
                    eccentricity=Inspiralling_Orbital_Elements[1])
        
        elif flag == 'Merged':
            return 'Merged'



    def point_masses(self, time):
        from math import cos, sin, sqrt
        m1   = 1 / (1+self.mass_ratio)
        m2   = self.mass_ratio/ (1+self.mass_ratio)
        a1   = m2
        a2   = m1
        flag = self.check_if_inspiral(time)

        # Case 1: Still at the burn-in stage
        if flag == 'Burn-in':
            primary, secondary = self.orbital_elements(time).orbital_state(time)

            return (
                PointMass(
                    softening_length=self.softening_length,
                    sink_model=SinkModel[self.sink_model.upper()],
                    sink_rate=self.sink_rate,
                    sink_radius=self.sink_radius,
                    **primary._asdict(),),
                PointMass(
                    softening_length=self.softening_length,
                    sink_model=SinkModel[self.sink_model.upper()],
                    sink_rate=self.sink_rate,
                    sink_radius=self.sink_radius,
                    **secondary._asdict(),),
                    )

        # Case 2: Inspiral has begun
        elif flag == 'Inspiralling':
            semi_major, eccen, phase = self.Orbital_Elements_During_Inspiral(time)
            Current_Omega            = sqrt(self.GM/semi_major/semi_major/semi_major)
            dphase_dt                = Current_Omega / (1-eccen * cos(phase))
            
            x1  = a1 * semi_major * cos (phase) - a1 * semi_major * eccen
            y1  = a2 * semi_major * (1 - eccen**2)**0.5 * sin (phase)
            x2  = -x1 * self.mass_ratio
            y2  = -y1 * self.mass_ratio
            vx1 = - dphase_dt * (a1 * semi_major * sin (phase))
            vy1 =   dphase_dt * (a2 * semi_major * (1 - eccen**2)**0.5 * cos (phase))
            vx2 = -vx1 * self.mass_ratio
            vy2 = -vy1 * self.mass_ratio

            c1 = PointMass(m1, x1, y1, vx1, vy1, softening_length= self.softening_length,sink_model=SinkModel['TORQUE_FREE'],sink_rate=self.sink_rate,sink_radius=self.sink_radius,)
            c2 = PointMass(m2, x2, y2, vx2, vy2, softening_length= self.softening_length,sink_model=SinkModel['TORQUE_FREE'],sink_rate=self.sink_rate,sink_radius=self.sink_radius,)
            return (c1,c2)
        
        # Case 3: Merger has occurred (remains centered)
        elif flag =='Merged':
            x  = 0.0
            y  = 0.0
            vx = 0.0
            vy = 0.0
            
            

            c1 = PointMass((m1 + m2) * (1 - self.epsilon), x, y, vx, vy, softening_length=2 * self.softening_length, sink_model=SinkModel['ACCELERATION_FREE'], sink_rate=self.sink_rate, sink_radius=2 * self.sink_radius,)
            c2 = PointMass(0, x, y, vx, vy, softening_length=0, sink_model=SinkModel['INACTIVE'], sink_rate=0, sink_radius=0,)
            return (c1,c2)

    def checkpoint_diagnostics(self, time):
        return dict(point_masses=self.point_masses(time))

# Buffer conditions


class ReferenceState(SetupBase):
    r"""
    A circumbinary disk setup for GW-inspiralling binaries.
    """

    eos                   = param("gamma-law", "EOS type: either isothermal or gamma-law")
    domain_radius         = param(10.0, "half side length of the square computational domain")
    mass_ratio            = param(1.0, "component mass ratio m2 / m1 <= 1", mutable=True)
    sink_rate             = param(1.0, "component sink rate", mutable=True)
    sink_radius           = param(0.03, "component sink radius", mutable=True)
    softening_length      = param(0.03, "gravitational softening length", mutable=True)
    buffer_is_enabled     = param(True, "whether the buffer zone is enabled", mutable=True)
    sink_model            = param("acceleration_free", "sink [acceleration_free|force_free|torque_free]", mutable=True)
    alpha                 = param(0.1, "alpha-viscosity parameter (gamma-law)")
    nu                    = param(0.001, "kinematic viscosity parameter (isothermal)")
    gamma_law_index_gas   = param(5.0 / 3.0, "adiabatic index (gamma-law)")
    constant_softening    = param(True, "whether to use constant softening (gamma-law)")
    retrograde            = param(False, "is disk retrograde?")
    which_diagnostics     = param("none", "diagnostics set to get from solver [none|mdots]")
    vkick                 = param(530.0, "velocity of the kick given to the merger remnant in km/s")
    epsilon               = param(0.05, "fraction of merger remnant lost through gw emission")
    direction             = param(1, "Direction of the")
    # Cooling specific parameters
    central_mass_msun     = param(8e6, "Mass of the central object in solar masses")
    semimajoraxis_pc      = param(None, "Length scale in parsecs") # Correct this for inspirals 
    mach_number_a         = param(10, "Disk Mach number") 
    target_accretion_rate = param(1., "Fraction of Eddington the disk we remap to in post-processing", mutable=True) 
    beta                  = param(1., "Gas pressure fraction P_gas/P_tot where P_tot = P_gas+P_rad") 
    # Inspiral specific parameters
    init_separation_rg    = param(100.0, "initial semi-major axis in grav-radii")
    init_eccentricity     = param(0.0, "orbital eccentricity of the binary")
    kick_start_time       = param(1000., "how many orbits before inspiral starts")
    integration_timestep  = param(0.001, "timestep for integrating the inspiral")
    semi_major_axis_list  = param([]," List of all semi-major axes over the inspiral")
    eccentricity_list     = param([]," List of all eccentricities axes over the inspiral")
    inspiral_time_list    = param([]," List of all eccentricities axes over the inspiral")
    gw_inspiral_time      = param(0.," The circular inspiral time for a0 = 1 ")
    Eccentric_Anomalies   = param([]," Find the true anomaly given the mean anomaly")
    OpticalDepthFloor     = param(0., "Minimum optical depth to measure lightcurves", mutable=True) 

    # Code units
    a0 = 1.0
    GM = 1.0

    @property
    def Gravitational_Radius_pc(self):
        from cooling import cgs
        return cgs['G'] * self.central_mass_msun * cgs['msun'] / cgs['c'] / cgs['c'] / cgs['pc']

    @property
    def length_scale_pc(self):
        r_g = self.Gravitational_Radius_pc
        if self.semimajoraxis_pc is None:
            return r_g * self.init_separation_rg  
        else:
            return self.length_scale_pc

    @property
    def is_isothermal(self):
        return self.eos == "isothermal"

    @property
    def is_gamma_law(self):
        return self.eos == "gamma-law"

    @property
    def gamma_law_index(self):
        #return self.beta + (4-3*self.beta)**2 * (self.gamma_law_index_gas-1) / ( self.beta + 12 * (self.gamma_law_index_gas-1) * (1-self.beta) )
        return cooling.gamma_law_index(self.beta, self.gamma_law_index_gas)

    @property
    def SS73(self):
        SS73_Setup = cooling.ShakuraSunyaevDisk(
            central_mass_msun = self.central_mass_msun, 
            length_scale_pc   = self.length_scale_pc,
            mach_number_a     = self.mach_number_a,
            alpha             = self.alpha,
            gamma             = self.gamma_law_index
            )
        return SS73_Setup

    @property
    def AccretionRateRescaling(self):
        return self.target_accretion_rate / self.SS73._eddington_fraction

    @property
    def cooling_coefficient(self):
        CoolingCoefficient = self.SS73.cooling_coefficient
        return CoolingCoefficient

    def primitive(self, t, coords, primitive):
        x, y       = coords
        r          = sqrt(x * x + y * y)
        r_softened = sqrt(x * x + y * y + self.softening_length * self.softening_length)
        phi_hat_x  = -y / max(r, 1e-12)
        phi_hat_y  = +x / max(r, 1e-12)

        sign = 1.0
        if self.retrograde == True:
                sign = -1.

        if self.is_isothermal:
            primitive[0] = 1.0
            primitive[1] = sqrt(self.GM / r_softened) * phi_hat_x * sign
            primitive[2] = sqrt(self.GM / r_softened) * phi_hat_y * sign

        elif self.is_gamma_law: # No Cavity
            sigma    = self.SS73.surface_density_profile(r_softened)
            pressure = self.SS73.surface_pressure_profile(r_softened)

            primitive[0] = sigma
            primitive[1] = sign  * sqrt(self.GM / r_softened) * phi_hat_x
            primitive[2] = sign  * sqrt(self.GM / r_softened) * phi_hat_y
            primitive[3] = pressure
            

    def mesh(self, resolution):
        return PlanarCartesian2DMesh.centered_square(self.domain_radius, resolution)

    @property
    def default_resolution(self):
        return 3000

    @property
    def physics(self):
        if self.is_isothermal:
            return dict(
                eos_type=EquationOfState.LOCALLY_ISOTHERMAL,
                mach_number=self.mach_number_3a,
                point_mass_function=self.point_masses,
                buffer_is_enabled=self.buffer_is_enabled,
                buffer_driving_rate=100.0,
                buffer_onset_width=1.0,
                cooling_coefficient=0.0,
                constant_softening=self.constant_softening,
                viscosity_model=ViscosityModel.CONSTANT_NU if self.nu > 0.0 else ViscosityModel.NONE,
                viscosity_coefficient=self.nu,
                alpha=0.0,
                diagnostics=self.diagnostics,
                retrograde=self.retrograde,
            )

        elif self.is_gamma_law:
            return dict(
                eos_type=EquationOfState.GAMMA_LAW,
                gamma_law_index=self.gamma_law_index,
                point_mass_function=self.point_masses,
                buffer_is_enabled=self.buffer_is_enabled,
                buffer_driving_rate=1000.0,  # default value in circumbinary.py
                buffer_onset_width=0.1,  # default value in circumbinary.py
                cooling_coefficient=self.cooling_coefficient,
                constant_softening=self.constant_softening,
                viscosity_model=ViscosityModel.CONSTANT_ALPHA if self.alpha > 0.0 else ViscosityModel.NONE,
                viscosity_coefficient=0.0,
                alpha=self.alpha,
                diagnostics=self.diagnostics,
                retrograde=self.retrograde,
            )


    @property
    def diagnostics(self):
        if self.which_diagnostics == "david":
            return [
                dict(quantity="time"),
                dict(quantity="semimajor-axis"),
                dict(quantity="eccentricity"),
                dict(quantity="density_floor"),
                dict(quantity="pressure_floor"),
                dict(quantity="Accreted_energy"),
                dict(quantity="optical"),
                dict(quantity="infared"),
                dict(quantity="bolometric"),
                dict(quantity="uncounted_cells_in_lc"),
                dict(quantity="mdot", which_mass=1, accretion=True),
                dict(quantity="mdot", which_mass=2, accretion=True),
                dict(quantity="torque", which_mass='both', gravity=True),
                dict(quantity="torque", which_mass='both', accretion=True),
                dict(quantity="power" ,which_mass=1,gravity=True),
                dict(quantity="power" ,which_mass=2,gravity=True),
                dict(quantity="power" ,which_mass=1,accretion=True),
                dict(quantity="power" ,which_mass=2,accretion=True),
                dict(quantity="angular_momentum"),
                dict(quantity="uv"),
                dict(quantity="xray"),
                dict(quantity="max_temperature"),
                
                #dict(quantity="eccentricity_vector", radial_cut=(1.0, 6.0)),
                #dict(quantity="torque",which_mass='both',gravity=True, radial_cut=(0.0, 1.0)),
                #dict(quantity="torque",which_mass='both',gravity=True, radial_cut=(1.0, 10.0)),
                #dict(quantity="power",which_mass=2,gravity=True, radial_cut=(0.0, 10.0)),
                #dict(quantity="power",which_mass=2,accretion=True, radial_cut=(0.0, 10.0)),
                #dict(quantity="power",which_mass=1,gravity=True, radial_cut=(0.0, 1.0)),
                #dict(quantity="power",which_mass=1,gravity=True, radial_cut=(1.0, 10.0)),
                #dict(quantity="power",which_mass=2,gravity=True, radial_cut=(0.0, 1.0)),
                #dict(quantity="power",which_mass=2,gravity=True, radial_cut=(1.0, 10.0)),
            ]
        else:
            return [
                dict(quantity="time"),
                dict(quantity="mdot", which_mass=1, accretion=True),
                dict(quantity="mdot", which_mass=2, accretion=True),
            ]

    from math import sqrt
    @property
    def solver(self):
        if self.is_isothermal:
            return "cbdiso_2d"
        elif self.is_gamma_law:
            return "cbdgam_2d"

    @property
    def boundary_condition(self):
        return "outflow"

    @property
    def reference_time_scale(self):
        return 2.0 * pi
    
    @property
    def Omega_0(self):
        return sqrt(self.GM/self.a0/self.a0/self.a0)
    
    @property
    def Phase_at_Start(self):
        return 0. # correct this later.... 

    @property
    def code_start_kick_time(self):
        return self.kick_start_time * self.reference_time_scale
    
    @property
    def kick_speed(self):
        c_code         = np.sqrt(self.init_separation_rg)
        Ratio_v_over_c = (self.vkick) / cgs['c']
        return Ratio_v_over_c * c_code

    
    def check_if_kick(self, time):
        if (time <= self.code_start_kick_time):
            return 'Burn-in'
        else:
            return 'Merged'
        



    def point_masses(self, time):
        from math import cos, sin, sqrt
        M = 1.0
        flag = self.check_if_kick(time)

        # Case 1: Still at the burn-in stage
        if flag == 'Burn-in':
            x1  = 0.
            y1  = 0.
            x2  = 0.
            y2  = 0.
            vx1 = 0.
            vy1 = 0.
            vx2 = 0.
            vy2 = 0.

            c1 = PointMass(0.5, x1, y1, vx1, vy1, softening_length= self.softening_length,sink_model=SinkModel['ACCELERATION_FREE'],sink_rate=self.sink_rate,sink_radius= self.sink_radius,)
            c2 = PointMass(0.5, x2, y2, vx2, vy2, softening_length= self.softening_length,sink_model=SinkModel['ACCELERATION_FREE'],sink_rate=self.sink_rate,sink_radius= self.sink_radius,)
            return (c1,c2)

        # Case 2: Kicked
        elif flag =='Merged':
            x  = 0.0
            y  = 0.0
            vx = 0.0
            vy = 0.0
            
            c1 = PointMass((1 - self.epsilon), x, y, vx, vy, softening_length=2 * self.softening_length, sink_model=SinkModel['ACCELERATION_FREE'], sink_rate=self.sink_rate, sink_radius=2 * self.sink_radius,)
            c2 = PointMass(0., x, y, vx, vy, softening_length=0, sink_model=SinkModel['INACTIVE'], sink_rate=0, sink_radius=0,)
            return (c1,c2)

    def checkpoint_diagnostics(self, time):
        return dict(point_masses=self.point_masses(time))







