# main function for dry sand collapse

# Yidong Zhao (ydzhao94@gmail.com)

# main references - papers:
# 1. [Jiang et al., APIC, 2015] (https://dl.acm.org/doi/10.1145/2766996)
# 2. [Jiang et al. JCP, 2017] (https://www.sciencedirect.com/science/article/pii/S0021999117301535)
# 3. [MPM SIGGRAPH 2016 course] (https://dl.acm.org/doi/10.1145/2897826.2927348)
# 4. [Hu et al., MLS-MPM] (https://dl.acm.org/doi/10.1145/3197517.3201293)
# 5. [Klar et al., Drucker-Prager sand simulation] (https://dl.acm.org/doi/10.1145/2897824.2925906)
# 6. [Klar et al., Drucker-Prager sand simulation - supplementary file] (https://www.seas.upenn.edu/~cffjiang/research/sand/tech-doc.pdf)
# 7. [Neto, Borja, ActaGeo, 2018] (https://link.springer.com/article/10.1007/s11440-018-0700-3)

# main references - code:
# 1. mpm99.py: from Taichi/examples/mpm99.py
# 2. A soil-water interaction code from Taichi GAMES201 course: (https://github.com/g1n0st/GAMES201/tree/master/hw2)
# 3. A soil-water interaction code writen in C++: (https://github.com/YiYiXia/Flame)


import taichi as ti
import numpy as np
import time
import sys
from numpy import savetxt

sys.path.append("..")

# materials
from materials.drucker_prager import *

ti.init(arch=ti.gpu) # Initialize Taichi according to hardware platform (for more information please refer to https://taichi.readthedocs.io/en/stable/hello.html?highlight=ti.gpu)
# ti.init(arch=ti.cpu, default_fp=ti.f64) # use double precision


# ============ PARTICLES, GRID QUANTITIES ============
quality = 1 # Larger value means higher resolution (more particles and finner mesh)
n_s_particles = ti.field(dtype=int, shape=())
max_num_s_particles = 1600 #25600 * quality ** 2 # MARK: magic number 25600 for d0=0.2m, h0=1.6m
n_grid = 100 #200 #400 #800 * quality # MARK: magic number 800 means how many grids in a direction

max_x = 2.0 #4.0 #8.0 # maximum x/y coordinate
dx, inv_dx = max_x/n_grid, float(n_grid/max_x)
dt = 5e-5/quality # time step dt

dim = 2 # problem dimension




# FLIP blending
FLIP_blending_ratio = 1.0




# Constants
standard_gravity = ti.Vector([0, -9.81])

boundary_friction_coeff = 0.3 #0.2 # 0.2 for 33 #0.17 for 37
tol = 1e-12

# Column collapse geometry (initial geometry sizes. Please see Fig. 2 in [Neto, Borja, ActaGeo, 2018])
d0 = 0.2
h0 = 0.4 #1.6
center_line_x = n_grid * dx / 2
start_x = center_line_x - d0 # the minimum x coordinate


p_vol = 2*d0*h0/max_num_s_particles


# Sand particles quantities
x_s = ti.Vector.field(dim, dtype=float, shape=max_num_s_particles) # sand position
x_s_plot = ti.Vector.field(dim, dtype=float, shape=max_num_s_particles) # sand position for drawing
v_s = ti.Vector.field(dim, dtype=float, shape=max_num_s_particles) # sand velocity
Affine_C_s = ti.Matrix.field(dim, dim, dtype=float, shape=max_num_s_particles) # Sand affine velocity C matrix, see section 5.3 in [Jiang et al., APIC, 2015] or section 2.2 in [Jiang et al. JCP, 2017] for the definition.

rho_s = 2.7 # sand initial density, t/m3
mass_s = p_vol * rho_s



# Sand grids quantities
grid_sv = ti.Vector.field(dim, dtype=float, shape=(n_grid, n_grid)) # grid sand momentum/velocity
grid_s_old_v = ti.Vector.field(dim, dtype=float, shape=(n_grid, n_grid)) # grid sand old momentum/velocity
grid_sm = ti.field(dtype=float, shape=(n_grid, n_grid)) # grid sand mass
grid_sf = ti.Vector.field(dim, dtype=float, shape=(n_grid, n_grid)) # grid sand force





# GUI setup
# scale_factor = 1.0 / max_x * 2.2
center_line_shift = max_x/2 - 1.0/2

scale_factor = 1.0/max_x * 1.2





# ====================================================




# ============ LOAD STEP ============
@ti.kernel
def load_step(step: ti.i32):
	gravity = min(500, step)/500 * standard_gravity # add gravity in 500 steps

	# reset grid quantities
	for i, j in grid_sm:
		# 2D
		grid_sv[i, j] = [0, 0]
		grid_s_old_v[i, j] = [0, 0]
		grid_sm[i, j] = 0
		grid_sf[i, j] = [0, 0]



	# P2G
	for p in range(n_s_particles[None]):
		base = (x_s[p] * inv_dx - 0.5).cast(int) # get grid index
		fx = x_s[p] * inv_dx - base.cast(float) # get reference coordinate, fx belongs to [0.5, 1.5]
		# Quadratic B-spline kernels, Eq. (123) in [MPM SIGGRAPH 2016 course]
		w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]

		# get stress
		stress_3D = sand_material.get_Kirchhoff_stress(p) # get sand Kirchhoff stress (3D)
		stress = ti.Matrix.zero(float, dim, dim)
		for i, j in ti.static(ti.ndrange(dim, dim)):
			stress[i, j] = stress_3D[i, j]

		# get affine
		affine = mass_s * Affine_C_s[p]

		# execute P2G - sand
		for i, j in ti.static(ti.ndrange(3, 3)):
			offset = ti.Vector([i, j])
			dpos = (offset.cast(float) - fx) * dx # difference between grid node position and particle position
			weight = w[i][0] * w[j][1]
			grid_sv[base + offset] += weight * (mass_s * v_s[p] + affine @ dpos)
			grid_sm[base + offset] += weight * mass_s
			grid_sf[base + offset] += weight * (-p_vol * 4 * inv_dx * inv_dx) * stress @ dpos # MARK: notice here is @







	# Momentum to velocity
	for i, j in grid_sm:
		if grid_sm[i, j] > 0:
			grid_sv[i, j] = (1 / grid_sm[i, j]) * grid_sv[i, j] # Momentum to velocity
			grid_s_old_v[i, j] = grid_sv[i, j]


	# Friction and BCs for old v
	for i, j in grid_sm:
		
		# BCs
		boundary_normal = ti.Vector.zero(float, dim)
		if grid_sm[i, j] > 0:
			if i < 3 and grid_s_old_v[i, j][0] < 0: 			boundary_normal = ti.Vector([1, 0])
			if i > n_grid - 3 and grid_s_old_v[i, j][0] > 0: boundary_normal = ti.Vector([-1, 0])
			if j < 3 and grid_s_old_v[i, j][1] < 0: 			boundary_normal = ti.Vector([0, 1])
			if j > n_grid - 3 and grid_s_old_v[i, j][1] > 0: boundary_normal = ti.Vector([0, -1])
		if boundary_normal[0] != 0 or boundary_normal[1] != 0:
			v_normal_mag = grid_s_old_v[i, j].dot(boundary_normal)
			v_normal = v_normal_mag * boundary_normal
			v_tangent = grid_s_old_v[i, j] - v_normal
			v_tangent_norm = v_tangent.norm()
			if v_tangent_norm > tol:
				# Coulomb friction
				if v_tangent_norm < abs(boundary_friction_coeff * v_normal_mag):
					grid_s_old_v[i, j] = ti.Vector([0 ,0])
				else:
					grid_s_old_v[i, j] = v_tangent
					grid_s_old_v[i, j] -= abs(boundary_friction_coeff * v_normal_mag) * (v_tangent/v_tangent_norm)

		# BCs (change normal v to zero)
		if grid_sm[i, j] > 0:
			if i < 3 and grid_s_old_v[i, j][0] < 0:          grid_s_old_v[i, j][0] = 0
			if i > n_grid - 3 and grid_s_old_v[i, j][0] > 0: grid_s_old_v[i, j][0] = 0
			if j < 3 and grid_s_old_v[i, j][1] < 0:          grid_s_old_v[i, j][1] = 0
			if j > n_grid - 3 and grid_s_old_v[i, j][1] > 0: grid_s_old_v[i, j][1] = 0

			if (i*dx <= start_x or i*dx >= start_x + 2*d0) and (step < 500): grid_s_old_v[i, j][0] = 0 # Add initial barrier





	# Explicit solver
	for i, j in grid_sm:

		# Velocity update
		if grid_sm[i, j] > 0:
			grid_sv[i, j] += dt * (gravity + grid_sf[i, j] / grid_sm[i, j])


		# Add boundary friction for sand
		boundary_normal = ti.Vector.zero(float, dim)
		if grid_sm[i, j] > 0:
			if i < 3 and grid_sv[i, j][0] < 0: 			boundary_normal = ti.Vector([1, 0])
			if i > n_grid - 3 and grid_sv[i, j][0] > 0: boundary_normal = ti.Vector([-1, 0])
			if j < 3 and grid_sv[i, j][1] < 0: 			boundary_normal = ti.Vector([0, 1])
			if j > n_grid - 3 and grid_sv[i, j][1] > 0: boundary_normal = ti.Vector([0, -1])
		if boundary_normal[0] != 0 or boundary_normal[1] != 0:
			v_normal_mag = grid_sv[i, j].dot(boundary_normal)
			v_normal = v_normal_mag * boundary_normal
			v_tangent = grid_sv[i, j] - v_normal
			v_tangent_norm = v_tangent.norm()
			if v_tangent_norm > tol:
				# Coulomb friction
				if v_tangent_norm < abs(boundary_friction_coeff * v_normal_mag):
					grid_sv[i, j] = ti.Vector([0 ,0])
				else:
					grid_sv[i, j] = v_tangent
					grid_sv[i, j] -= abs(boundary_friction_coeff * v_normal_mag) * (v_tangent/v_tangent_norm)



		# BCs (change normal v to zero)
		if grid_sm[i, j] > 0:
			if i < 3 and grid_sv[i, j][0] < 0:          grid_sv[i, j][0] = 0
			if i > n_grid - 3 and grid_sv[i, j][0] > 0: grid_sv[i, j][0] = 0
			if j < 3 and grid_sv[i, j][1] < 0:          grid_sv[i, j][1] = 0
			if j > n_grid - 3 and grid_sv[i, j][1] > 0: grid_sv[i, j][1] = 0

			if (i*dx <= start_x or i*dx >= start_x + 2*d0) and (step < 500): grid_sv[i, j][0] = 0 # Add initial barrier



	# G2P
	for p in range(n_s_particles[None]):
		base = (x_s[p] * inv_dx - 0.5).cast(int)
		fx = x_s[p] * inv_dx - base.cast(float)
		w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
		new_v = ti.Vector.zero(float, dim)
		new_C = ti.Matrix.zero(float, dim, dim)

		new_FLIP_v = v_s[p]

		for i, j in ti.static(ti.ndrange(3, 3)):
			offset = ti.Vector([i, j])
			dpos = (offset.cast(float) - fx) * dx
			g_v = grid_sv[base + ti.Vector([i, j])]
			g_old_v = grid_s_old_v[base + ti.Vector([i, j])]
			weight = w[i][0] * w[j][1]

			new_v += weight * g_v # get new particle velocity
			new_FLIP_v += weight * (g_v - g_old_v)
			new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx ** 2 # get new particle affine velocity matrix


		delta_F = (ti.Matrix.identity(float, dim) + dt * new_C)
		# update material
		sand_material.update_deformation_gradient(delta_F, p)


		# particle advection
		v_s[p], Affine_C_s[p] = (FLIP_blending_ratio * new_FLIP_v + (1 - FLIP_blending_ratio) * new_v), new_C
		x_s[p] += dt * new_v

		x_s_plot[p] = [center_line_x ,0] + (x_s[p] - [center_line_x, 0]) * scale_factor - [center_line_shift, 0]








# ===================================

# ============ INITIALIZATION ============
sand_material = DruckerPrager(n_particles=max_num_s_particles, dim=2, E=2.016e4, nu=0.3, friction_angle=35, cohesion=0.0) # plane strain, 2.016e4 kPa, friction angle = 35 degree, cohesion = 0.0

@ti.kernel
def initialize():
	n_s_particles[None] = 0
	# material initialization
	sand_material.initialize()

	# # randome position, velocity initialization
	# n_s_particles[None] = max_num_s_particles
	# for p in range(n_s_particles):
	# 	x_s[p] = [ti.random() * 2*d0 + start_x, ti.random() * h0 + 2*dx] # 2D
	# 	v_s[p] = ti.Matrix([0,0]) # 2D

	# Homogeneous initialization (2x2 PPC)
	new_particle_id = 0
	for i, j in grid_sm:
		if i*dx >= start_x and i*dx < start_x + 2*d0 and j >= 2 and j*dx < 2*dx + h0: # (2.0m x 2.0m computation domain)
			new_particle_id = ti.atomic_add(n_s_particles[None], 1)
			x_s[new_particle_id] = [i * dx + 0.25 * dx, j * dx + 0.25 * dx]
			v_s[new_particle_id] = ti.Matrix([0,0])

			new_particle_id = ti.atomic_add(n_s_particles[None], 1)
			x_s[new_particle_id] = [i * dx + 0.75 * dx, j * dx + 0.25 * dx]
			v_s[new_particle_id] = ti.Matrix([0,0])

			new_particle_id = ti.atomic_add(n_s_particles[None], 1)
			x_s[new_particle_id] = [i * dx + 0.25 * dx, j * dx + 0.75 * dx]
			v_s[new_particle_id] = ti.Matrix([0,0])

			new_particle_id = ti.atomic_add(n_s_particles[None], 1)
			x_s[new_particle_id] = [i * dx + 0.75 * dx, j * dx + 0.75 * dx]
			v_s[new_particle_id] = ti.Matrix([0,0])
	print('n_s_particles: ', n_s_particles[None])




color_s = ti.field(dtype = int, shape = max_num_s_particles)
@ti.kernel
def update_color():
	for p in range(n_s_particles[None]):
		random_number = ti.random()
		# color_s[p] = 225*256**2 + 169*256 + 95 # yellow

		# Random color
		if random_number<=0.85:
			color_s[p] = 225*256**2 + 169*256 + 95 # yellow
		elif random_number>0.85 and random_number<=0.95:
			color_s[p] = 107*256**2 + 84*256 + 30 # brow
		else:
			color_s[p] = 255*256**2 + 255*256 + 255 # white



# ============ MAIN LOOP ============
# initialization
initialize()

update_color()

# gui
gui = ti.GUI("Dry sand collapse", res = 1024, background_color = 0xFFFFFF)

# export file
export_file = '../results/dry_sand_collapse_plane_strain.ply'

# experimental solution
d_inf = 0
a_ratio = h0/d0
if a_ratio >= 0 and a_ratio <= 2.3:
	d_inf = (1+1.2*a_ratio) * d0
else:
	d_inf = (1+1.9*a_ratio ** (2/3)) * d0

print('center line x: ', center_line_x)
print('run out distance: ', d_inf)


total_step = 0
while True:
# for total_step in range(50000): # 150 frames
	for step in range(50):
		total_step += 1
		load_step(total_step)



	# gui.circles(x_s.to_numpy(), radius=3.5, color=color_s.to_numpy())
	gui.circles(x_s_plot.to_numpy(), radius=4.8*scale_factor, color=color_s.to_numpy())

	# # experimental run-out distance
	# gui.line(begin=(center_line_x+scale_factor*d_inf-center_line_shift, 0.0), end=(center_line_x+scale_factor*d_inf-center_line_shift, 0.5), color=0x0, radius=2)
	# gui.text(content=f'Experimental runout distance', pos=(0.6, 0.6), color=0x0, font_size=35)


	# # Centerline
	# gui.line(begin=(0.5,0), end=(0.5,0.5), color=0x0, radius=2)


	# gui.show()
	gui.show(f'../animation/sand_collapse_{gui.frame:06d}.png')

	# Show time
	current_time = max(total_step-500, 0) * dt
	gui.text(content=f'Time: {current_time:.2f} s' , pos=(0.4,0.8), color=0x0, font_size=42)



	x_s_to_numpy = x_s.to_numpy()
	np.savetxt(f'../data/sand_data_{gui.frame:06d}.csv', x_s_to_numpy)


	# Export ply file
	if total_step*dt > 1.5 and export_file:
	
		break








