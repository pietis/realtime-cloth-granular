"""Particle buffer for MPM granular flow.

Preallocated max-N design (no dynamic alloc) — Step 2.3 in 설명서.md.
`radius` field is required for JKR pull-off work in Operator A (Step 5).
`active` flag toggles when particle is absorbed into cloth surface reservoir.
"""

import taichi as ti


@ti.dataclass
class Particle:
    pos:    ti.types.vector(3, ti.f32)
    vel:    ti.types.vector(3, ti.f32)
    F:      ti.types.matrix(3, 3, ti.f32)   # deformation gradient
    C:      ti.types.matrix(3, 3, ti.f32)   # affine velocity
    Jp:     ti.f32                           # plastic volume ratio (Drucker-Prager)
    mass:   ti.f32
    radius: ti.f32                           # used by JKR pull-off work
    active: ti.i32                           # 1 = in MPM domain, 0 = absorbed into σ


def make_particle_field(n: int) -> ti.StructField:
    """Allocate a Particle struct field of size n (preallocated, no dynamic alloc)."""
    return Particle.field(shape=n)


@ti.kernel
def init_particles_box(
    particles: ti.template(),
    n_active: ti.i32,
    box_min: ti.types.vector(3, ti.f32),
    box_max: ti.types.vector(3, ti.f32),
    mass_per_particle: ti.f32,
    default_radius: ti.f32,
):
    """Initialize n_active particles uniformly inside a box, rest inactive."""
    for i in range(particles.shape[0]):
        if i < n_active:
            r = ti.Vector([ti.random(ti.f32), ti.random(ti.f32), ti.random(ti.f32)])
            particles[i].pos = box_min + r * (box_max - box_min)
            particles[i].vel = ti.Vector([0.0, 0.0, 0.0])
            particles[i].F = ti.Matrix.identity(ti.f32, 3)
            particles[i].C = ti.Matrix.zero(ti.f32, 3, 3)
            particles[i].Jp = 1.0
            particles[i].mass = mass_per_particle
            particles[i].radius = default_radius
            particles[i].active = 1
        else:
            particles[i].pos = ti.Vector([0.0, 0.0, 0.0])
            particles[i].vel = ti.Vector([0.0, 0.0, 0.0])
            particles[i].F = ti.Matrix.identity(ti.f32, 3)
            particles[i].C = ti.Matrix.zero(ti.f32, 3, 3)
            particles[i].Jp = 1.0
            particles[i].mass = 0.0
            particles[i].radius = 0.0
            particles[i].active = 0


@ti.kernel
def count_active(particles: ti.template()) -> ti.i32:
    n = 0
    for i in range(particles.shape[0]):
        if particles[i].active == 1:
            n += 1
    return n
