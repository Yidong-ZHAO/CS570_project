# Drucker-Prager

# main references - papers:
# 1. [Klar et al., Drucker-Prager sand simulation] (https://dl.acm.org/doi/10.1145/2897824.2925906)
# 2. [Klar et al., Drucker-Prager sand simulation - supplementary file] (https://www.seas.upenn.edu/~cffjiang/research/sand/tech-doc.pdf)
# 3. [Tampubolon et al., Multi-species simulation of porous sand and water mixtures] (https://dl.acm.org/doi/10.1145/3072959.3073651)

# main references - code:
# 1. A soil-water interaction code from Taichi GAMES201 course: (https://github.com/g1n0st/GAMES201/tree/master/hw2)
# 2. A soil-water interaction code writen in C++: (https://github.com/YiYiXia/Flame)

import taichi as ti
import numpy as np

# ti.init(arch=ti.gpu)
ti.init(arch=ti.cpu, default_fp=ti.f64)

# constants
pi = 3.141592653

@ti.data_oriented
class DruckerPrager:
    def __init__(self,
                 n_particles, # number of particles
                 dim, # problem dimension, 2 = 2D (plain strain assumption), 3 = 3D
                 E, # Young's modulus
                 nu, # Poisson's ratio
                 friction_angle, # friction angle in degree unit
                 cohesion
                 ):
        self.n_particles = n_particles
        self.dim = dim
        self.E = E
        self.nu = nu
        self.lame_mu, self.lame_lambda = E / (2*(1+nu)), E*nu / ((1+nu) * (1-2*nu)) # Lame parameters
        self.friction_angle = friction_angle
        self.cone_shift_due_to_cohesion = cohesion / (ti.tan(friction_angle*pi/180) * (9*self.lame_lambda + 6*self.lame_mu)) # TODO: check

        # Quantities declaration (some of them will be initialized in another funciton)
        self.F_elastic_array = ti.Matrix.field(3, 3, dtype = float, shape = n_particles)
        self.saturation_array = ti.field(dtype = float, shape = n_particles)
        self.volume_correction_array = ti.field(dtype = float, shape = n_particles) # volume correction for tension case, see section 4.3.4 in [ref paper 3] for more details
        self.friction_coeff_array = ti.field(dtype = float, shape = n_particles)
        self.q_array = ti.field(dtype = float, shape = n_particles) # hardening parameter, see section 7.3 in [ref paper 1] for more details
        self.plastic_multiplier = ti.field(dtype = float, shape = n_particles)


    # ============ MEMBER FUNCTIONS - taichi scope ============
    @ti.func
    def initialize(self):
        F_elastic_init = ti.Matrix.identity(float, 3)

        for p in range(self.n_particles):
            self.F_elastic_array[p] = F_elastic_init

            self.saturation_array[p] = 0
            self.volume_correction_array[p] = 0
            # self.friction_coeff_array[p] = ti.tan(self.friction_angle*pi/180)
            self.friction_coeff_array[p] = 1.0/3.0 * (2*ti.sqrt(6)*ti.sin(self.friction_angle*pi/180))/(3-ti.sin(self.friction_angle*pi/180)) # NOTICE there is a conversion due to different friction angel definitions
            self.q_array[p] = 0
            self.plastic_multiplier[p] = 0.0


    @ti.func
    def update_saturation(self, phi, p):
        self.saturation_array[p] = phi

    @ti.func
    def project(self, e_trial_0, p):
        # TODO: check whether there are some other ways to consider volume correction
        # e_trial = e_trial_0 + self.volume_correction_array[p]/3 * ti.Matrix.identity(float, 3) # volume correction
        e_trial = e_trial_0 # not consider volume correction

        # shift the elastic strain to consider cohesion
        e_trial -= self.cone_shift_due_to_cohesion * ti.Matrix.identity(float, 3)

        ehat = e_trial - e_trial.trace()/3 * ti.Matrix.identity(float, 3) # Eq. (27) in [ref paper 1]
        ehat_norm = ti.sqrt(ehat[0, 0] ** 2 + ehat[1, 1] ** 2 + ehat[2, 2] ** 2)

        delta_lambda = ehat_norm + (3 * self.lame_lambda + 2 * self.lame_mu) / (2 * self.lame_mu) * e_trial.trace() * self.friction_coeff_array[p] # Eq. (27) in [ref paper 1]

        new_e = ti.Matrix.zero(float, 3, 3)
        delta_q = 0.0

        # three cases
        if ehat_norm <= 0 or e_trial.trace() > 0: # case II, project to the tip
            new_e = ti.Matrix.zero(float, 3, 3)
            e_trial += self.cone_shift_due_to_cohesion * ti.Matrix.identity(float, 3)
            e_trial_norm = ti.sqrt(e_trial[0, 0] ** 2 + e_trial[1, 1] ** 2 + e_trial[2, 2] ** 2)
            delta_q = e_trial_norm

            self.plastic_multiplier[p] += delta_lambda
            # TODO: state
        elif delta_lambda <= 0: # case I, elastic
            new_e = e_trial_0
            delta_q = 0.0
            # TODO: state
        else: # case III, plastic
            new_e = e_trial + self.cone_shift_due_to_cohesion * ti.Matrix.identity(float, 3) - delta_lambda / ehat_norm * ehat # Eq. (28) in [ref paper 1]
            delta_q = delta_lambda

            self.plastic_multiplier[p] += delta_lambda
            # TODO: state

        return new_e, delta_q




    @ti.func
    def update_deformation_gradient(self, delta_F, p):
        # new_F_elastic_trial = (ti.Matrix.identity(float, self.dim) + dt * new_C) @ self.F_elastic_array[p]
        # new_C_3D = ti.Matrix.zero(float, 3, 3)
        delta_F_3D = ti.Matrix.zero(float, 3, 3)
        if self.dim == 2:
            for i, j in ti.static(ti.ndrange(2, 2)):
                # new_C_3D[i, j] = new_C[i, j]
                delta_F_3D[i, j] = delta_F[i, j]
            # new_C_3D[2, 2] = 0.0
            delta_F_3D[2, 2] = 1.0
        else:
            for i, j in ti.static(ti.ndrange(self.dim, self.dim)):
                # new_C_3D[i, j] = new_C[i, j]
                delta_F_3D[i, j] = delta_F[i, j]

        # new_F_elastic_trial = (ti.Matrix.identity(float, 3) + dt * new_C_3D) @ self.F_elastic_array[p]
        new_F_elastic_trial = delta_F_3D @ self.F_elastic_array[p]

        U, sig, V = ti.svd(new_F_elastic_trial)
        e = ti.Matrix.zero(float, 3, 3)
        for d in ti.static(range(3)):
            e[d, d] = ti.log(sig[d, d])
        new_e, dq = self.project(e, p)

        # TODO: hardening

        # get new elastic deformation gradient
        exp_new_e = ti.Matrix.zero(float, 3, 3)
        for d in ti.static(range(3)):
            exp_new_e[d, d] = ti.exp(new_e[d, d])
        new_F_elastic = U @ exp_new_e @ V.transpose()

        # update volume correction quantity
        self.volume_correction_array[p] += -ti.log(new_F_elastic.determinant()) + ti.log(new_F_elastic_trial.determinant()) # Eq. (26) in [ref paper 3]

        # update elastic deformation gradient
        self.F_elastic_array[p] = new_F_elastic


    @ti.func
    def get_Kirchhoff_stress(self, p):
        U, sig, V = ti.svd(self.F_elastic_array[p])
        inv_sig = sig.inverse()
        e = ti.Matrix.zero(float, 3, 3)
        for d in ti.static(range(3)):
            e[d, d] = ti.log(sig[d, d])
        stress = U @ (2 * self.lame_mu * inv_sig @ e + self.lame_lambda * e.trace() * inv_sig) @ V.transpose() # formula (26) in Klar et al., pk1 stress
        stress = stress @ self.F_elastic_array[p].transpose() # Kirchhoff stress

        return stress

    @ti.func
    def get_plastic_multiplier(self, p):
        return self.plastic_multiplier[p]











