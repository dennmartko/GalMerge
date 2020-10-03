#Code for this class was obtained from:
#https://stackoverflow.com/questions/41522890/how-to-save-frequently-used-physical-constants-in-python

class Constant(float):
    def __new__(cls, value, units, doc):
        self = float.__new__(cls, value)
        self.units = units
        self.doc = doc
        return self

# Physical constants
Msol = Constant(1.9891 * 10 ** (30), "kg", "Solar mass in kg")
G = Constant(6.67430 * 10 ** (-11), "m3 kg-1 s-2", "Gravitational constant (G)")
#Galt = Constant(4.49376*10**(-6), "kpc3 Msol-1 Gyr-2", "Alternative gravitational constant (G)")
kpc2m = Constant(3.08567758*10**19, "m kpc-1", "Conversion factor for kpc to m")
m2kpc = Constant(1/kpc2m, "kpc m-1", "Conversion factor for m to kpc")
s2Gyr = Constant(60*60*365*10**9, "Gyr s-1", "Conversion factor for s to Gyr")
G_ = Constant(4.49376*10**(-6), "kpc3 Msol−1 Gyr−2", "Conversion to correct units G")