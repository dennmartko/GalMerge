import os
import sys
from multiprocessing import cpu_count

#file IO functions
def listfiles(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file

def check_and_generate_fname(fname, path=None):
    if path is None:
        path = os.path.realpath(".")

    name, ext = os.path.splitext(fname)

    matches = []
    for file in listfiles(path):
        tmp = os.path.splitext(file)
        if tmp[1] == ext and tmp[0][:len(name)] == name:
            matches.append(tmp[0])

    if matches != []:
        i = 1
        cond = True
        while cond:
            newname = name + str(i)
            if newname in matches:
                i += 1
            else:
                return newname + ext
    return fname

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
        self.verbose = False

        #argument dictionaries
        options = {"n" : self.n, "o" : self.o, "f" : self.f}
        flags = {"d" : [self.d], "v" : [self.v], "dv" : [self.d, self.v], "vd" : [self.d, self.v], "-debug" : [self.d], "h" : [self.h], "-help" : [self.h]}

        #parser
        option = None
        for argv in sys.argv[1:]:
            if argv[0] == "-":
                if option is not None:
                    raise SystemError(f"Missing command line argument for '-{option}'!")
                argv = argv[1:]
                if argv in flags.keys():
                    for func in flags[argv]:
                        func()
                elif argv in options.keys():
                    option = argv
                else:
                    raise SystemError(f"Flag or option '-{argv}' undefined!")
            elif option is not None:
                options[option](argv)
                option = None

        return (self.N_CPU, self.fname, self.outpath, self.debug, self.verbose)

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

    def v(self):
        self.verbose = True

    def h(self):
        print("""SYNTAX python3 main.py [flags] [options [arguments]]
    FLAGS:
        -d/--debug : Toggle debugging (default: False). If debugging is on it will store useful information in a log file.
        -v : Controls the verbosity of the debugger.
        -h/--help : Prints help.
    OPTIONS:
        -n [cores] : The number of cores to be used for force calculation (defaults to the total number of CPUs in the system).
        -o [filename] : Filename used for storing the animation (defaults to "animation").
        -f [path] : Path to the directory where all output files will be stored (defaults to root directory of the program).""")
        sys.exit()

#function to write debug messages
def debugmsg(file, message, write_mode='a', verbose=False, writer=None):
    with open(file, write_mode) as f:
        f.write(message + "\n")
    
    if verbose:
        if writer is None:
            print(message)
        else:
            writer(message)



if __name__ == "__main__":
    print(argv_parser())
    print(check_and_generate_fname("debug_log.txt", "./logs/"))