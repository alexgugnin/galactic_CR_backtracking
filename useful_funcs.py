def countExecTime(func):
    '''
    Function to calculate the execution time of arg function
    '''
    import time

    def innerFunc(*args, **kwargs):
        '''
        Some func, execution time of wich to be calculated
        '''
        start_time = time.time()

        return_val = func(*args, **kwargs)

        end_time = time.time()
        total_time = end_time - start_time
        print(f"EXECUTION TIME of function {func.__name__} is : {total_time}\n")

        return return_val

    return innerFunc

def eqToGal(RA:float, DEC:float):
    '''
    Function to convert equatorial to galactic coordinates with(J2000) -> rad:
    1.GAL North Pole equat coords:
        a0 = 192.8595
        d0 = 27.1284
    2.Gal longtitude of the equat north pole:
        l0 = 122.9320
    CHECK THIS
    https://www.atnf.csiro.au/people/Tobias.Westmeier/tools_coords.php
    https://galaxiesbook.org/chapters/A.-Coordinate-systems.html
    '''
    import math
    a0 = 192.8595*math.pi/180
    d0 = 27.1284*math.pi/180
    l0 = 122.9320*math.pi/180
    #RA = RA*math.pi/180
    #DEC = DEC*math.pi/180

    l = 0 #longtitude
    b = 0 #latitude

    temp = (math.cos(DEC)*math.sin(RA - a0))/(math.sin(DEC)*math.cos(d0) - math.cos(DEC)*math.sin(d0)*math.cos(RA - a0))
    l = math.atan((math.tan(l0)-temp)/(1+temp*math.tan(l0)))
    b = math.asin(math.sin(DEC)*math.sin(d0) + math.cos(DEC)*math.cos(d0)*math.cos(RA-a0))

    '''
    FOR THE 0-360 coords
    if l < 0:
        l = 2*math.pi + l
    '''
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    cords = SkyCoord(ra=RA, dec=DEC, frame='icrs', unit='deg')
    temp_l = cords.galactic.l.radian
    temp_b = cords.galactic.b.radian
    #if temp_l >= 180: temp_l -= math.pi*2
    return temp_l, temp_b #Longtitude from 0 to 360
    #return l,b#in ragians

if __name__ == '__main__':
    '''
    Some tests
    '''
    import numpy as np
    l,b = eqToGal(178, 13)
    print(l*180/np.pi,b*180/np.pi)
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    cords = SkyCoord(ra=178, dec=13, frame='icrs', unit='deg')
    print(cords.galactic.l.deg, cords.galactic.b.deg)
