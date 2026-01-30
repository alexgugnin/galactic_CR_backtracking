from crpropa import *
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    plt.figure(figsize=(10, 6))
    ax = plt.subplot(111, projection="mollweide")

    seed = 42
    sigma_dir = 15*np.pi/180 #1 degree directional uncertainty
    R = Random(seed)

    #RandVectorAroundMean test
    vectors = []
    for i in range(1000):
        mean_dir = Vector3d()
        mean_dir.setRThetaPhi(1, 30*np.pi/180, 30*np.pi/180)
        direction = R.randVectorAroundMean(mean_dir, sigma_dir)
        vectors.append(direction)

    directions = np.array([[v.x, v.y, v.z] for v in vectors])
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    unit_vectors = directions / norms

    x, y, z = unit_vectors.T

    # Longitude (RA): -pi to pi
    lon = np.arctan2(y, x) 

    # Latitude (Dec): -pi/2 to pi/2
    lat = np.arcsin(z)

    #ax.scatter(lon, lat, s=5, alpha=0.7, color='teal')

    #Fisher test
    kappa = 2.278/(sigma_dir**2)  # Concentration parameter
    #kappa = 1/(sigma_dir**2) 
    vectors_fisher = [] 
    for i in range(1000):
        mean_dir = Vector3d()
        mean_dir.setRThetaPhi(1, 30*np.pi/180, 30*np.pi/180)
        direction = R.randFisherVector(mean_dir, kappa)
        vectors_fisher.append(direction)

    directions_fisher = np.array([[v.x, v.y, v.z] for v in vectors_fisher])
    norms_fisher = np.linalg.norm(directions_fisher, axis=1, keepdims=True)
    unit_vectors_fisher = directions_fisher / norms_fisher

    x_fisher, y_fisher, z_fisher = unit_vectors_fisher.T

    # Longitude (RA): -pi to pi
    lon_fisher = np.arctan2(y_fisher, x_fisher) 

    # Latitude (Dec): -pi/2 to pi/2
    lat_fisher = np.arcsin(z_fisher)

    ax.scatter(lon_fisher, lat_fisher, s=5, alpha=0.7, color='red')

    ax.grid(True)
    ax.set_title("Mollweide Projection of Vector Directions")
    plt.show()