import os,sys
from pyraf import iraf

IN = "/tmp/cootoxy.in"
OUT = "/tmp/cootoxy.out"

def cootoxy(image):
    for f in (IN,OUT):
        try:
            os.remove(f)
        except OSError:
            pass
    RA = raw_input("\nObject RA (hh:mm:ss.ss):\n").strip()
    DEC = raw_input("Object DEC ([+/-]dd:mm:ss.ss):\n").strip()

    with open(IN,"w") as infile:
        infile.write("{} {}".format(RA,DEC))

    iraf.wcsctran(input = IN,
                  output = OUT,
                  image = image,
                  inwcs = "world",
                  outwcs = "logical",
                  units="h n",
                  formats="%8.3f %8.3f",
                  columns = "1 2")
    
    print "\nObject pixel coordinates (x,y):"
    print open(OUT,"r").readlines()[-1].lstrip()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print "USAGE:\npython cootoxy.py image.fits"
        sys.exit(2)
    try:
        open(sys.argv[1])
    except IOError:
        print "couldn't open {}".format(sys.argv[1])
    else:
        cootoxy(sys.argv[1])