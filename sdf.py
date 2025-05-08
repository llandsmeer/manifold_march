import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

def normalize(x):
    return x / jnp.linalg.norm(x)

def vec(x, y, z):
    return jnp.array([x, y, z], dtype='float32')

screen_x, screen_y = jnp.meshgrid(jnp.linspace(-1, 1, 300), jnp.linspace(-1, 1, 300))
screen_z = 1 * jnp.ones_like(screen_x)

light_dir = normalize(vec(-1, -1, 1.))
light_origin = vec(2, 3, 1)
light_color = vec(1.0, 0.8, .2)
base_color = vec(0.5, 0.5, 0.5)
background_color = vec(0.1, 0.3, 0.9)

def sdf(x):
    # a, b, c = x
    # sigma = 10.0 + a
    # rho = 28.0 + b
    # beta = 8.0/3.0 + c
    # x, y, z = 1.0, 1.0, 1.0
    # dt = 0.1
    # for _ in range(10):
    #     dx = sigma * (y - x)
    #     dy = x * (rho - z) - y
    #     dz = x * y - beta * z
    #     x = x + dt * dx
    #     y = y + dt * dy
    #     z = z + dt * dz
    # return (jnp.sqrt(((vec(x, y, z) - vec(-30, 30 , 30))**2).sum()) - 10)/10
    return jnp.minimum(
           jnp.minimum(
            jnp.sqrt(((x - vec(-0.8, .1, 1.0))**2).sum()) - 0.3,
            jnp.sqrt(((x - vec(0.2, .1, 1.0))**2).sum()) - 0.5,
            ),
            x[1] + 1
            )

def sdf_normal(x):
    return normalize(jax.grad(sdf)(x))

rays = jnp.vstack([
    screen_x.flatten(),
    screen_y.flatten(),
    screen_z.flatten()
    ]).T

origins = jnp.zeros_like(rays)
rays = jax.vmap(normalize)(rays)

def march(origin, ray):
    def loop(at, _):
        d = sdf(at)
        at = at + d * ray
        return at, at
    at, _ = jax.lax.scan(loop, origin, length=100)
    hit = (sdf(at) < 0.01) & (jnp.isfinite(sdf(at)))
    c = (hit) * (
            base_color +
            - light_color * (sdf_normal(at) @ light_dir) * jnp.minimum(1, 1 / (0.03 * jnp.linalg.norm(at - light_origin)**2))
            ) + (1-hit) * background_color
    return c
    return jnp.tanh((c-.5)/2)/2 + .5

img = jax.vmap(march)(origins, rays).reshape(*screen_z.shape, 3)
plt.imshow(img[::-1],
           extent=(float(screen_x.min()), float(screen_x.max()),
                   float(screen_y.min()), float(screen_y.max())))
plt.axis('off')
plt.savefig('/tmp/out.png', bbox_inches='tight')
plt.show()

