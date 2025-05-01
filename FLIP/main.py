import taichi as ti
import numpy as np
import time
import cv2
#from attempt_cg import *
from cg import *

np.set_printoptions(precision=4, linewidth=300)

# Scale all of the four parameters together
NUM = 2 ** 14 # total number of particles
NUM_HALF = 2 ** 7 # must be square root of NUM
GRID_NUM = 128 # grid dimensions
SUBSTEPS = 40

vec2 = ti.math.vec2
GRAVITY = vec2(0.0, -9.8) 
RHO = 1000
FLIP = 0.9

h = 1.0 / 60 # frame rate
dt = 1.0 / 60 / SUBSTEPS # simulation rate

WINDOW_SIZE = 1024
WINDOW_R = 0.8 * 0.5 * 0.5 / NUM_HALF

USESOLID = False

@ti.dataclass
class Particle:
    x: vec2
    v: vec2
    m: ti.f32

ti.init(arch=ti.gpu)

# Particles
particles = Particle.field(shape=(NUM))

# MAC grid
u = ti.field(dtype=ti.f32, shape=(GRID_NUM+1,GRID_NUM))
v = ti.field(dtype=ti.f32, shape=(GRID_NUM,GRID_NUM+1))
uold = ti.field(dtype=ti.f32, shape=(GRID_NUM+1,GRID_NUM))
vold = ti.field(dtype=ti.f32, shape=(GRID_NUM,GRID_NUM+1))

# solid
solid = ti.field(dtype=ti.i32, shape=(GRID_NUM,GRID_NUM))
solid.fill(0)
pixels = ti.field(dtype=ti.f32, shape=(WINDOW_SIZE, WINDOW_SIZE))

# Parallel index
GN = GRID_NUM
grid_count = ti.field(dtype=ti.i32, shape=(GN, GN))
list_head = ti.field(dtype=ti.i32, shape=GN * GN)
list_cur = ti.field(dtype=ti.i32, shape=GN * GN)
list_tail = ti.field(dtype=ti.i32, shape=GN * GN)
column_sum = ti.field(dtype=ti.i32, shape=GN)
prefix_sum = ti.field(dtype=ti.i32, shape=(GN, GN))
particle_id = ti.field(dtype=ti.i32, shape=NUM)


# Initialize square of particles
@ti.kernel
def init(particles: ti.template()):
    # Spread particles in centerred square of length 0.5
    for i in particles:
        x = (i // NUM_HALF) / NUM_HALF
        y = (i % NUM_HALF) / NUM_HALF
        #pos = vec2(1./6 + 4./6 * x, 1./3 + 1./3 * y)
        pos = vec2(0.01 + 1./3 * x, 0.01 + 2./3 * y)
        particles[i].x = pos
        particles[i].m = RHO * 2. / 9. / NUM
        vel = vec2(0.0, 0.0) 
        particles[i].v = vel


@ti.kernel
def init_solid(solid: ti.template()):
    for i,j in solid:
        '''
        if solid[i,j] > 128:
            solid[i,j] = 0
        else:
            solid[i,j] = 1
        '''
        x = (i + 0.5) / GRID_NUM
        y = (j + 0.5) / GRID_NUM
        if (y + ti.abs(x - 0.5)) < 0.1:
            solid[i,j] = 1

@ti.kernel
def pixels_solid(solid: ti.template(), pixels: ti.template()):
    RATIO = WINDOW_SIZE // GRID_NUM
    for i,j in pixels:
        ir = i // RATIO
        jr = j // RATIO
        if solid[ir,jr]:
            pixels[i,j] = 0.4
        else:
            pixels[i,j] = 0.0

# Resolve boundary
@ti.func
def boundary(x):
    xnew = x
    dhat = 0.1 / GRID_NUM
    if x[0] > 1.0 - dhat:
        xnew[0] = ti.max(2.0 - 2*dhat - x[0], 0.0)
    elif x[0] < dhat:
        xnew[0] = ti.min(-x[0]+2*dhat, 1.0)
    if x[1] > 1.0 - dhat:
        xnew[1] = ti.max(2.0 - 2*dhat - x[1], 1.0)
    elif x[1] < dhat:
        xnew[1] = ti.min(-x[1]+2*dhat, 1.0)
    return xnew

@ti.func
def boundary_solid(x, solid):
    ui = ti.floor(x[0] * GRID_NUM, int)
    uj = ti.floor(x[1] * GRID_NUM, int)
    return solid[ui, uj]

@ti.func
def bounce_solid(x, solid):
    xnew = x
    dhat = 0.1 / GRID_NUM
    ui = ti.floor(x[0] * GRID_NUM, int)
    uj = ti.floor(x[1] * GRID_NUM, int)
    fi = x[0] - ui / GRID_NUM
    fj = x[1] - uj / GRID_NUM
    if ui > 0 and solid[ui-1,uj] and fi < dhat:
        xnew[0] += dhat
    if ui < GRID_NUM - 1 and solid[ui+1,uj] and fi > 1 - dhat:
        xnew[0] -= dhat
    if uj > 0 and solid[ui,uj-1] and fj < dhat:
        xnew[1] += dhat
    if uj < GRID_NUM - 1 and solid[ui,uj+1] and fj > 1 - dhat:
        xnew[1] -= dhat
    return xnew
    

# Advect particle with grid
@ti.kernel
def advect(particles: ti.template(), u: ti.template(), v: ti.template(), solid: ti.template()):
    for i in particles:
        xn = particles[i].x
        xhalf = boundary(xn + 0.5 * dt * grid_query(u, v, particles[i].x))
        particles[i].x = xhalf
        xn1 = boundary(xn + dt * grid_query(u, v, particles[i].x))
        particles[i].x = xn1
        if boundary_solid(xn1, solid):
            particles[i].x = bounce_solid(xn, solid)


#https://docs.taichi-lang.org/blog/acclerate-collision-detection-with-taichi
@ti.kernel
def init_grid(particles: ti.template()):
    grid_num = GRID_NUM

    # Grid counting
    grid_count.fill(0)
    for i in particles:
        grid_idx = ti.floor(particles[i].x * grid_num, int)
        grid_count[grid_idx] += 1

    # Scan count over 2d grid
    for i in range(grid_num):
        csum = 0
        for j in range(grid_num):
            csum += grid_count[i, j]
        column_sum[i] = csum

    prefix_sum[0, 0] = 0
    ti.loop_config(serialize=True)
    for i in range(1, grid_num):
        prefix_sum[i, 0] = prefix_sum[i - 1, 0] + column_sum[i - 1]

    for i in range(grid_num):
        for j in range(grid_num):
            if j == 0:
                prefix_sum[i, j] += grid_count[i, j]
            else:
                prefix_sum[i, j] = prefix_sum[i, j - 1] + grid_count[i, j]
            linear_idx = i * grid_num + j
            list_head[linear_idx] = prefix_sum[i, j] - grid_count[i, j]
            list_cur[linear_idx] = list_head[linear_idx]
            list_tail[linear_idx] = prefix_sum[i, j]

    # Particle id to grid index conversion
    for i in range(NUM):
        grid_idx = ti.floor(particles[i].x * grid_num, int)
        linear_idx = grid_idx[0] * grid_num + grid_idx[1]
        grid_location = ti.atomic_add(list_cur[linear_idx], 1)
        particle_id[grid_location] = i


# Partciel to grid velocity transfer
@ti.kernel
def particle_to_grid(particles: ti.template(), u: ti.template(), v: ti.template(), solid: ti.template()):

    grid_num = GRID_NUM
    #ti.loop_config(serialize=True)
    for i,j in u:
        if i > 0 and i < GRID_NUM:

            if solid[i-1,j] == 1 or solid[i,j] == 1:
                u[i,j] = 0

            else:
                cellm = 0.
                cellu = 0.
                for neigh_i in range(i - 1, i + 1):
                    for neigh_j in range(j - 1, j + 2):
                        if neigh_j >= 0 and neigh_j < grid_num:
                        
                            neigh_linear_idx = neigh_i * grid_num + neigh_j
                            for p_idx in range(list_head[neigh_linear_idx],
                                               list_tail[neigh_linear_idx]):

                                k = particle_id[p_idx]
                                particle = particles[k]
                                qx = ti.abs(grid_num * particle.x[0] - i)
                                qy = ti.abs(grid_num * particle.x[1] - j - 0.5)

                                rxy = ti.max(1. - qx, 0.) * ti.max(1. - qy, 0.)
                                cellm += rxy * particle.m
                                cellu += rxy * particle.m * particle.v[0]
                u[i,j] = cellu / cellm if cellm > 0.0 else 0.0
        else:
            u[i,j] = 0. # boundary edge, treat zero velocity


    for i,j in v:
        if j > 0 and j < GRID_NUM:

            if solid[i,j-1] == 1 or solid[i,j] == 1:
                v[i,j] = 0

            else:
                cellm = 0.
                cellv = 0.
                for neigh_i in range(i - 1, i + 2):
                    for neigh_j in range(j - 1, j + 1):
                        if neigh_i >= 0 and neigh_i < grid_num:

                            neigh_linear_idx = neigh_i * grid_num + neigh_j
                            for p_idx in range(list_head[neigh_linear_idx],
                                               list_tail[neigh_linear_idx]):

                                k = particle_id[p_idx]
                                particle = particles[k]
                                qx = ti.abs(grid_num * particle.x[0] - i - 0.5)
                                qy = ti.abs(grid_num * particle.x[1] - j)

                                rxy = ti.max(1. - qx, 0.) * ti.max(1. - qy, 0.)
                                cellm += rxy * particle.m
                                cellv += rxy * particle.m * particle.v[1]
                v[i,j] = cellv / cellm if cellm > 0.0 else 0.0
        else:
            v[i,j] = 0. # boundary edge, treat zero velocity


# Body forces
@ti.kernel
def body(u: ti.template(), v: ti.template(), solid: ti.template()):
    for i,j in v:
        if j > 0 and j < GRID_NUM:
            if solid[i,j-1] == 0 and solid[i,j] == 0:
                v[i,j] += dt * GRAVITY[1]


# Query velocity
@ti.func
def grid_query_u(u, pos):
    partu = 0.
    ui = ti.floor(pos[0] * GRID_NUM, int)
    uj = ti.floor(pos[1] * GRID_NUM - 0.5, int)
    if uj == -1 or uj == GRID_NUM - 1:
        neigh_j = 0 if (uj == -1) else GRID_NUM - 1
        for neigh_i in range(ui, ui+2):
            qx = ti.abs(GRID_NUM * pos[0] - neigh_i)
            qy = 0.
            rxy = ti.max(1. - qx, 0.) * ti.max(1. - qy, 0.)
            partu += u[neigh_i, neigh_j] * rxy
    else:
        for neigh_i in range(ui, ui+2):
            for neigh_j in range(uj, uj+2):
                qx = ti.abs(GRID_NUM * pos[0] - neigh_i)
                qy = ti.abs(GRID_NUM * pos[1] - neigh_j - 0.5)
                rxy = ti.max(1. - qx, 0.) * ti.max(1. - qy, 0.)
                partu += u[neigh_i, neigh_j] * rxy
    return partu
@ti.func
def grid_query_v(v, pos):
    partv = 0.
    vi = ti.floor(pos[0] * GRID_NUM - 0.5, int)
    vj = ti.floor(pos[1] * GRID_NUM, int)
    if vi == -1 or vi == GRID_NUM - 1:
        neigh_i = 0 if (vi == -1) else GRID_NUM - 1
        for neigh_j in range(vj, vj+2):
            qx = 0.
            qy = ti.abs(GRID_NUM * pos[1] - neigh_j)
            rxy = ti.max(1. - qx, 0.) * ti.max(1. - qy, 0.)
            partv += v[neigh_i, neigh_j] * rxy
    else:
        for neigh_i in range(vi, vi+2):
            for neigh_j in range(vj, vj+2):
                qx = ti.abs(GRID_NUM * pos[0] - neigh_i - 0.5)
                qy = ti.abs(GRID_NUM * pos[1] - neigh_j)
                rxy = ti.max(1. - qx, 0.) * ti.max(1. - qy, 0.)
                partv += v[neigh_i, neigh_j] * rxy
    return partv
@ti.func
def grid_query(u, v, pos):
    return vec2(grid_query_u(u, pos), grid_query_v(v, pos))


# grid to particle transfer
@ti.kernel
def grid_to_particle(particles: ti.template(), u: ti.template(), v: ti.template(), uold: ti.template(), vold: ti.template()):
    for k in particles:
        particle = particles[k]
        unew = grid_query(u, v, particle.x) 
        uoldold = grid_query(uold, vold, particle.x)
        particles[k].v = FLIP * (particles[k].v - uoldold) + unew


@ti.kernel
def calc_div(u: ti.template(), v: ti.template(), divu: ti.template(), grid_count : ti.template(), solid: ti.template()):
    for i,j in grid_count:
        if grid_count[i,j] > 0 and solid[i,j] == 0:
            divu[GRID_NUM*i+j] = u[i+1,j] - u[i,j] + v[i,j+1] - v[i,j]
        else:
            divu[GRID_NUM*i+j] = 0.0


@ti.kernel
def update_u(u: ti.template(), v: ti.template(), p: ti.template(), solid: ti.template()):
    for i,j in u:
        if i > 0 and i < GRID_NUM:
            if solid[i-1,j] == 1 or solid[i,j] == 1:
                u[i,j] = 0
            else:
                u[i,j] = u[i,j] + (p[i*GRID_NUM+j] - p[(i-1)*GRID_NUM+j])
        else:
            u[i,j] = 0.0

    for i,j in v:
        if j > 0 and j < GRID_NUM:
            if solid[i,j-1] == 1 or solid[i,j] == 1:
                v[i,j] = 0
            v[i,j] = v[i,j] + (p[i*GRID_NUM+j] - p[i*GRID_NUM+j-1])
        else:
            v[i,j] = 0.0


@ti.kernel
def fill(Adiag: ti.template(), Aoff: ti.template(), grid_count : ti.template(), solid: ti.template()):
    for i,j in grid_count:
        k = i * GRID_NUM + j

        Aoff[k].e1 = -1
        Aoff[k].e2 = -1
        Aoff[k].e3 = -1
        Aoff[k].e4 = -1
        Aoff[k].e5 = -1
        Aoff[k].e6 = -1
        Adiag[k] = 0

        if grid_count[i,j] > 0 and solid[i,j] == 0:
            Acoef = 0
            if i < GRID_NUM - 1 and solid[i+1,j] == 0:
                Acoef += 1
                if grid_count[i+1,j] > 0:
                    Aoff[k].e1 = (i+1) * GRID_NUM + j
            if i > 0 and solid[i-1,j] == 0:
                Acoef += 1
                if grid_count[i-1,j] > 0:
                    Aoff[k].e2 = (i-1) * GRID_NUM + j
            if j < GRID_NUM - 1 and solid[i,j+1] == 0:
                Acoef += 1
                if grid_count[i,j+1] > 0:
                    Aoff[k].e3 = i * GRID_NUM + j + 1
            if j > 0 and solid[i,j-1] == 0:
                Acoef += 1
                if grid_count[i,j-1] > 0:
                    Aoff[k].e4 = i * GRID_NUM + j - 1
            Adiag[k] = Acoef



gui = ti.GUI('FLIP', (WINDOW_SIZE, WINDOW_SIZE))
lines = GRID_NUM + 1
lines_h_b = np.stack([np.zeros(lines), np.linspace(0.0,1.0,num=lines)], axis=-1)
lines_h_e = np.stack([np.ones(lines), np.linspace(0.0,1.0,num=lines)], axis=-1)
lines_v_b = np.stack([np.linspace(0.0,1.0,num=lines), np.zeros(lines)], axis=-1)
lines_v_e = np.stack([np.linspace(0.0,1.0,num=lines), np.ones(lines)], axis=-1)

t = 0.0
frame = 0
init(particles)
if USESOLID:
    #im = cv2.imread("418.png")
    ##solid.from_numpy(im[:,:,0])
    init_solid(solid)

solver = CG(GRID_NUM * GRID_NUM)

overall_time = time.time()
solver_time = 0.0
while gui.running:

    for s in range(SUBSTEPS):

        # parallel scan for particle to grid transfer
        init_grid(particles) 

        # transfer velocity to grid
        particle_to_grid(particles, u, v, solid)
        uold.copy_from(u)
        vold.copy_from(v)
        
        # advect particle position with grid velocity, RK2
        advect(particles, u, v, solid)

        # apply gravity to grid
        body(u, v, solid)

        # pressure project grid
        start_time = time.time()
        calc_div(u, v, solver.b, grid_count, solid)
        fill(solver.Adiag, solver.Aoff, grid_count, solid)
        solver.solve()
        update_u(u, v, solver.x, solid)

        solver_time += time.time() - start_time

        # transfer velocity to particles
        grid_to_particle(particles, u, v, uold, vold)

        t += dt

    frame += 1

    pos = particles.x.to_numpy()
    pixels_solid(solid, pixels)
    ui.set_image(pixels)
    #gui.lines(begin=lines_h_b, end=lines_h_e, radius=1, color=0x068587)
    gui.lines(begin=lines_v_b, end=lines_v_e, radius=1, color=0x068587)
    gui.circles(pos, radius=(WINDOW_R * WINDOW_SIZE))

    filename = f'out/frame_{frame:03d}.png' 
    gui.show(filename)

    vtime = solver.vectortime
    mtime = solver.matmultime
    rtime = solver.reducetime
    cgiters = solver.cgiters
    total_time = time.time() - overall_time

    if frame % 10 == 0:
        print("Frame %s, total %s, solver %s, reduce %s, vector %s, matrix %s, cgiters %s" % (frame, total_time, solver_time, rtime, vtime, mtime, cgiters))

    if frame == 300:
        exit()


