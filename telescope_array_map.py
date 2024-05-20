import matplotlib.pyplot as plt
from useful_funcs import eqToGal
import numpy as np


events = []
with open('data/TA2023_highE.dat', 'r') as infile:
    for line in infile:
        if line.split()[0] == '#': continue
        temp_event = line.split()
        events.append((temp_event[0], float(temp_event[6]), float(temp_event[7]), float(temp_event[8])))

#coords = [eqToGal(event[1], event[2]) for event in events]
#x, y = [coord[0] for coord in coords], [coord[1] for coord in coords]

x, y = np.array([coord[1]*np.pi/180 for coord in events]), np.array([coord[2]*np.pi/180 for coord in events])
x = np.pi - x
#x[x < 0] = -x[x < 0]

plt.figure(figsize=(12, 7))
plt.subplot(111, projection = 'hammer')
plt.grid(True)
#plt.grid(which = "both")
#plt.minorticks_on()
#plt.title('TITLE', y=1.1)

plt.scatter(x, y, s = 15)

x_tick_labels = ['360°', '300°', '240°', '180°', '120°', '60°', '0°']
#x_tick_positions = [0, -np.pi/4, -np.pi/2, -3*np.pi/4, -np.pi, np.pi, 3*np.pi/4, np.pi/2, np.pi/4]
x_tick_positions = [-np.pi, -2*np.pi/3, -np.pi/3, 0, np.pi/3, 2*np.pi/3, -np.pi]

y_tick_labels = ['-90°', '-60°', '-30°', '0°', '30°', '60°', '90°']
y_tick_positions = [-np.pi/2, -np.pi/3, -np.pi/6, 0, np.pi/6, np.pi/3, np.pi/2]

plt.xticks(x_tick_positions, labels=x_tick_labels)
plt.yticks(y_tick_positions, labels=y_tick_labels)

plt.savefig("TA_test.png", dpi=400, bbox_inches='tight')
plt.show()
