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