import sys
import os
from multiprocessing import cpu_count

#command line argument parser
class argv_parser(tuple):
    def __new__(cls):
        #create self
        self = tuple.__new__(cls)
        
        #defaults
        self.N_CPU = cpu_count()
        self.fname = "animation"
        self.outpath = os.path.dirname(os.path.abspath(__file__))
        self.debug = False

        #argument dictionaries
        options = {"n" : self.n, "o" : self.o, "f" : self.f}
        flags = {"d" : self.d, "-debug" : self.d, "h" : self.h, "-help" : self.h}

        #parser
        option = None
        for argv in sys.argv[1:]:
            if argv[0] == "-":
                if option is not None:
                    raise SystemError(f"Missing command line argument for '-{option}'!")
                argv = argv[1:]
                if argv in flags.keys():
                    flags[argv]()
                elif argv in options.keys():
                    option = argv
                else:
                    raise SystemError(f"Flag or option '-{argv}' undefined!")
            elif option is not None:
                options[option](argv)
                option = None

        return (self.N_CPU, self.fname, self.outpath, self.debug)

    #option functions
    def n(self, x):
        self.N_CPU = int(x)

    def o(self, x):
        self.fname = x

    def f(self, x):
        self.outpath = os.path.realpath(x)

    #flag functions
    def d(self):
        self.debug = True

    def h(self):
        print("""SYNTAX python3 main.py [flags] [options [arguments]]
    FLAGS:
        -d/--debug : Toggle debugging (default: False). If debugging is on it will store useful information in a log file.
        -h/--help : Prints help.
    OPTIONS:
        -n [cores] : The number of cores to be used for force calculation (defaults to the total number of CPUs in the system).
        -o [filename] : Filename used for storing the animation (defaults to "animation").
        -f [path] : Path to the directory where all output files will be stored (defaults to root directory of the program).""")
        sys.exit()

if __name__ == "__main__":
    print(argv_parser())