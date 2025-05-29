import astropy.units as u
from astropy.coordinates import SkyCoord

events = []
with open('Auger_lowE_shapley.dat', 'w') as outf:
    with open('../data/AugerApJS2022_lowE.dat', 'r') as infile:
        for line in infile:
            if line.split()[0][0] == '#': continue
            temp_event = line.split()
            if float(temp_event[7]) < 50.0: continue #Limiting energies to be from lgE_min = 19.7 - phys constraints
            cords = SkyCoord(ra=float(temp_event[5])*u.deg, dec=float(temp_event[6])*u.deg, frame='icrs').transform_to("galactic")
            #if cords.l.value > 270 and cords.l.value < 330 and cords.b.value > 15 and cords.b.value < 45:
            #if cords.l.value > 290 and cords.l.value < 320 and cords.b.value > 15 and cords.b.value < 50:
            if cords.l.value > 270 and cords.l.value < 330 and cords.b.value > 15 and cords.b.value < 50: #MAKE  > 10
                #if cords.b.value < 23 and cords.l.value < 305: continue
                outf.write(line)

    #Check N>3e19 , 7e19 + >10 . Also check if any updates in data, remove levels
    #For low energies two images in one, for high energies two images in one