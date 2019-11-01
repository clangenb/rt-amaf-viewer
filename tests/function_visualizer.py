import numpy as np
import matplotlib.pyplot as plt


def DoRotation(xspan, yspan, rotrad=0):
    """Generate a meshgrid and rotate it by RotRat radians"""
    rot = [[np.cos(rotrad), - np.sin(rotrad)],
           [np.sin(rotrad), np.cos(rotrad)]]

    x, y = np.meshgrid(xspan, yspan)
    return np.einsum('ji, mni -> jmn', rot, np.dstack([x, y]))




if __name__ == '__main__':
    # Argumentwerte als 1D Arrays erzeugen
    x_1d = np.linspace(-3,3,14)
    y_1d = np.linspace(-3,3,14)

    # Argumentwerte als 2D Arrays erzeugen
    x_2d, y_2d = DoRotation(x_1d, y_1d, 60/180*np.pi)

    # Interessante Daten erzeugen
    z_2d = 1/((x_2d)**2 + y_2d**2 + 1) * np.cos(np.pi * (x_2d+3)) * np.cos(np.pi * (y_2d+3))


    # Plotten
    plt.figure()
    plt.pcolormesh(x_2d, y_2d, 0.1*z_2d, cmap='hsv')
    plt.gca().set_aspect("equal") # x- und y-Skala im gleichen Maßstaab
    plt.colorbar()

    plt.show()

    print('Z:', z_2d)