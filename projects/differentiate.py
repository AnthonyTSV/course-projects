import numpy as np


def differentiate(u, dt):
    d = np.zeros(u.shape)
    for i in range(1, len(u)-1):
        d[i] = (u[i+1] - u[i-1])/(2*dt)
    d[0] = (u[1] - u[0])/dt
    d[-1] = (u[-1] - u[-2])/dt
    return d

def differentiate_vector(u, dt):
    d = np.zeros(u.shape)
    d[0] = (u[1] - u[0])/dt
    d[1:-1] = (u[2:] - u[0:-2])/(2*dt)
    d[-1] = (u[-1] - u[-2])/dt
    return d

def test_differentiate():
    t = np.linspace(0, 1, 10)
    dt = t[1] - t[0]
    u = t**2
    du1 = differentiate(u, dt)
    du2 = differentiate_vector(u, dt)
    assert np.allclose(du1, du2)

if __name__ == '__main__':
    test_differentiate()
    