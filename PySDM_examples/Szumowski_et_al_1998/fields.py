import numpy as np
from PySDM.state.arakawa_c import make_rhod, z_scalar_coord


def z_vec_coord(grid):
    nx = grid[0]
    nz = grid[1]+1
    xX = np.repeat(np.linspace(1/2, grid[0]-1/2, nx).reshape((nx, 1)), nz, axis=1) / grid[0]
    assert np.amin(xX) >= 0
    assert np.amax(xX) <= 1
    assert xX.shape == (nx, nz)
    zZ = np.repeat(np.linspace(0, grid[1], nz).reshape((1, nz)), nx, axis=0) / grid[1]
    assert np.amin(zZ) == 0
    assert np.amax(zZ) == 1
    assert zZ.shape == (nx, nz)
    return xX, zZ


def x_vec_coord(grid):
    nx = grid[0]+1
    nz = grid[1]
    xX = np.repeat(np.linspace(0, grid[0], nx).reshape((nx, 1)), nz, axis=1) / grid[0]
    assert np.amin(xX) == 0
    assert np.amax(xX) == 1
    assert xX.shape == (nx, nz)
    zZ = np.repeat(z_scalar_coord(grid).reshape((1, nz)), nx, axis=0) / grid[1]
    assert np.amin(zZ) >= 0
    assert np.amax(zZ) <= 1
    assert zZ.shape == (nx, nz)
    return xX, zZ

def nondivergent_vector_field_2d(grid, size, dt, stream_function: callable):
    dx = size[0] / grid[0]
    dz = size[1] / grid[1]
    dxX = 1 / grid[0]
    dzZ = 1 / grid[1]

    xX, zZ = x_vec_coord(grid)
    rho_velocity_x = -(stream_function(xX, zZ + dzZ/2) - stream_function(xX, zZ - dzZ/2)) / dz

    xX, zZ = z_vec_coord(grid)
    rho_velocity_z = (stream_function(xX + dxX/2, zZ) - stream_function(xX - dxX/2, zZ)) / dx

    rho_times_courant = [rho_velocity_x * dt / dx, rho_velocity_z * dt / dz]
    return rho_times_courant


class Fields:
    def __init__(self, environment, stream_function, initial_profiles):
        self.g_factor = make_rhod(environment.mesh.grid, environment.rhod_of)
        self.environment = environment
        self.stream_function = stream_function
        self.advector = None
        self.sample_advector()

        self.advectees = dict(
            (key, np.repeat(
                profile.reshape(1,-1),
                environment.mesh.grid[0],
                axis=0)
             ) for key, profile in initial_profiles.items()
        )

    def sample_advector(self):
        self.advector = nondivergent_vector_field_2d(
            self.environment.mesh.grid, self.environment.mesh.size, self.environment.dt, self.stream_function)
        self.courant_field = (
            self.advector[0] / self.environment.rhod_of(zZ=x_vec_coord(self.environment.mesh.grid)[-1]),
            self.advector[1] / self.environment.rhod_of(zZ=z_vec_coord(self.environment.mesh.grid)[-1])
        )
