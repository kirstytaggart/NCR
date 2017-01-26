#! /usr/bin/env python
"""
jdl - 15/09/2013
rmb - ?

Run the NCR analysis on an image.

"""

import argparse
import math
import sys
import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import pyfits


__version__ = '1.2.2'
__author__ = 'jdl@astro.livjm.ac.uk'

class Logger(object):
    """
    Quick logger class to save all stdout to a logfile as well as printing
    to terminal
    """
    def __init__(self, filename,quiet=False):
        self.terminal = sys.stdout
        self.log = open(filename, "w")
        self.quiet = quiet

    def write(self, message):
        if not self.quiet:
            self.terminal.write(message)
        self.log.write(message)

class NCR(object):
    """
    Initialise class to perform NCR analysis.

    Args:
    fitsfile: path to a valid fits file from which to read data (string)
    snlocation: x and y coordinates of sn in zero-indexed pixel number or
                ra and dec as hh:mm:ss and dd:mm:ss (string,list,tuple)
    KWargs:
    section: pixel coordinates of the section to cut from the image as
             x1,y1,x2,y2, default=None - i.e. whole image (string,list,tuple)
    ext: extension to retrieve data from in fitsfile, default=0 (int)
    bin: the binning factor for pixels, default=1 (int)
    negcut: set pixels >5sigma below background to zero, default=False (bool)
    starcentres: pixel coordinates of stars to mask given as x1,y1,...,xn,yn
                 deault=None (string,list,tuple)
    masksize: radius of the mask to apply to star centres, default=10 (int)
    plot: create various plots of results, default=True (bool)
    makeimages: create fits files of the sectioned and binned arrays, 
                default=False (bool)
    constant: constant value to *subtract* from all pixel values, default=0
              (float)
    ds9: used ds9 to interactively select the section, starcentres and masksize,
         requires pyds9 (bool)
    quiet: supress output to stdout - output just directed to log file, 
           default=False (bool)

    Returns length-5 tuple consisting of:
    ncrvalue: NCR value of SN
    snpixvalue: pixel value of snlocation in fitsfile
    (binsnx,binsny): snlocation in the sectioned,binned array
    binarray: the sectioned and binned array of pixel values
    cumulativearray: normalised cumulative sum of pixel values in sectioned
                     and binned array


    snlocation,section and starcentres can be lists/tuples of the values or
    comma-separated strings.
    e.g. section="10,10,50,50" or section=[10,10,50,50]

    ALL pixel coordinates should be given zero-indexed!

    NB on sections:
    - The program will reduce the section as appropriate to ensure each 
      axis is divisible by the binning number.
    - The section will be shrunk so that the SN location in the section falls
      into the central pixel of a super pixel for accuracy during binning
    - The section will always be made smaller to avoid any OutOfBounds issues
    tl;dr: Your section may have a few pixel shifts applied to it.
    """
    def __init__(self,fitsfile,snlocation,section=None,ext=0,bin=1,negcut=False,
                 starcentres=None,masksize=10,plot=True,makeimages=False,
                 constant=0,ds9=False,quiet=False):
        # get the data array from fitsfile
        try:
            self.imagearray = pyfits.getdata(fitsfile,ext)
            self.headerarray = pyfits.getheader(fitsfile,ext)
            
        except (IOError,IndexError):
            raise Exception("cannot get data from {0} (extension {1})"
                                 .format(fitsfile,ext))

        self.xsize,self.ysize = self.imagearray.shape[::-1]
        self.basename = os.path.splitext(fitsfile)[0]

        if isinstance(snlocation,str):
            snloc = snlocation.split(",")
        else:
            snloc = map(str,snlocation)
        if len(snloc) != 2:
            raise Exception("snlocation must be comma separated string or a"\
                            "length=2 tuple or list of x and y")
        self.snra,self.sndec = None,None
        if len(snloc[0].split(":")) == 3 and len(snloc[1].split(":")) == 3:
            # assign ra and dec and use when displaying ds9 to show DSS image
            self.snra = snloc[0]
            self.sndec = snloc[1]
            snx,sny = cootoxy(fitsfile,snloc[0],snloc[1])
            if not (0 <= snx <= self.xsize) or not (0 <= sny <= self.ysize):
                raise Exception("SN not within image!\nReturned pixel "\
                                "coordinates: {0},{1}".format(snx,sny))
            # need to ensure we're still on zero-indexing! 
            # cootoxy returns 1-index
            snloc[0],snloc[1] = snx-1,sny-1
        elif len(snloc[0].split(":")) != 1 and len(snloc[1].split(":")) != 1:
            raise Exception("SN coordinates must be pixel values (x,y) or RA"\
                            " and DEC in the form hh:mm:ss,dd:mm:ss")
        #snloc = map(int,snloc)
        snloc = [int(round(float(loc))) for loc in snloc] 

        if ds9:
            # overwrite section,starcentres and masksize using values from
            # ds9 interaction
            import pyds9 as pyds9
            self.d = pyds9.ds9(wait=12,start=True)
            section,starcentres,masksize = self._use_ds9(fitsfile,ext,snloc,
                                                         masksize)
            
        if section == None:
            # default to size of image in that case
            x1,y1,x2,y2 = 0,0,self.xsize,self.ysize
        else:
            if isinstance(section,str):
                section = section.split(",")
            x1,y1,x2,y2 = map(int,section)    
        if x1 >= x2 or y1 >= y2:
            raise Exception("check section parameters are valid (x1<x2,y1<y2)")

        if starcentres is not None:
            # convert the "x1,y1,x2,y2.." user string to [(x1,y1),(x2,y2)..]
            # list of tuples for NCR class
            c = starcentres
            if isinstance(c,str):
                c = starcentres.split(",")
            if len(c)%2 != 0:
                raise Exception("length of starcentres not divisible by 2 ")
            starcentres = [(int(c[i]),int(c[i+1])) for i in range(0,len(c),2)]

        if bin < 1 or bin%2 != 1:
            raise Exception("bin must be positive and odd")

        # reduce size of section so axes are divisible by bin
        x2 = x2 -(x2-x1)%bin 
        y2 = y2 -(y2-y1)%bin 
        # this shouldn't happen now due to above lines, but you never know...
        if len(range(x1,x2))%bin != 0 or\
           len(range(y1,y2))%bin != 0:
            raise Exception("section needs to be multiple of bin ({0}) in both "
                            "axes.".format(bin))

        if not (x1 < snloc[0] <= x2) or not (y1 <snloc[1] < y2):
            raise Exception("snlocation not within section!")

        if masksize < 1:
            raise Exception("masksize must be positive")

        # assign as class attributes after checking
        self.fitsfile = fitsfile
        self.snloc = snloc
        self.snx = snloc[0]
        self.sny = snloc[1]
        self.section = section
        self.x1,self.y1,self.x2,self.y2 = x1,y1,x2,y2
        self.bin = bin
        self.ext = ext
        self.negcut = negcut
        self.starcentres = starcentres
        self.masksize = masksize
        self.plot = plot
        self.makeimages = makeimages
        self.constant = constant
        self.quiet = quiet


    def main(self):
        """
        Perform the NCR analysis.
        """
        # duplicate stdout to a logfile
        oldstdout = sys.stdout
        sys.stdout = Logger(self.basename+"_NCR.log",self.quiet)
        wholearray = self.imagearray

        # subtract the constant
        if self.constant != 0:
            wholearray -= self.constant

        # get pixel value at SN location, just for interest
        self.snpixvalue = wholearray[self.sny,self.snx]

        # mask over saturated stars with a clipped median value of 
        # surrounding sky.

        # OLD MASKING ROUTINE!
        """
        m = self.masksize
        if self.starcentres != None:
            for star in self.starcentres:
                sx,sy = [int(i) for i in star]
                wa = wholearray
                # create a frame around masking box 3 pixels wide to compute
                # a clipped median to fill box with
                frame = np.concatenate(
                            (wa[sy-m-3:sy+m+4,sx-m-3:sx-m].ravel(),
                             wa[sy-m-3:sy+m+4,sx+m+1:sx+m+4].ravel(),
                             wa[sy-m-3:sy-m,sx-m:sx+m+1].ravel(),
                             wa[sy+m+1:sy+m+4,sx-m:sx+m+1].ravel()))
                frameclip = clipdata(frame)
                # replace values in 
                wa[sy-m:sy+m+1,sx-m:sx+m+1] = frameclip[3]
        """
        # this is much better:
        if self.starcentres != None:
            for star in self.starcentres:
                sx,sy = [int(i) for i in star]
                # because of way numpy+pyfits reading in data, we need to switch
                # x and y when calling mask_circle
                wholearray = mask_circle(wholearray,sy,sx,self.masksize)
            
        # update SN location onto section coordinates 
        sectsnx = self.snx - self.x1
        sectsny = self.sny - self.y1

        # ensure SN location pixel in section array will be the central pixel
        # in a binned pixel for accuracy of SN location in binned array
        # this is a crap way...
        """
        if sectsnx%self.bin == 0:
            shiftx=2
        elif sectsnx%self.bin == 2:
            shiftx=1
        else:
            shiftx=0
        if sectsny%self.bin == 0:
            shifty=2
        elif sectsny%self.bin == 2:
            shifty=1
        else:
            shifty=0
        """
        # this is a much better way and works for any odd +ve bin:
        shiftx = getshift(sectsnx,self.bin)
        shifty = getshift(sectsny,self.bin)
        if shiftx:
            self.x1 += shiftx
            self.x2 -= (self.bin-shiftx)
            # update SN location onto new section coordinates 
            sectsnx = self.snx - self.x1
        if shifty:
            self.y1 += shifty
            self.y2 -= (self.bin-shifty)
            # update SN location onto new section coordinates
            sectsny = self.sny - self.y1

        # grab the required section from wholearray
        # (numpy + pyfits makes x and y designations counter-intuitive)
        sectionarray = wholearray[self.y1:self.y2,self.x1:self.x2]

        # cut all negative pixels < -5*stddev of background to 0
        if self.negcut:
            sectstddev,sectmean,sectdatamin,sectmedian = clipdata(sectionarray)
            cutpixels = len(sectionarray[sectionarray<sectdatamin].ravel())
            sectionarray[sectionarray<sectdatamin] = 0

        if self.makeimages:
            # write section array as fits file
            try:
                pyfits.writeto(self.basename+"_NCRsect.fits",data=sectionarray)
            except IOError:
                os.remove(self.basename+"_NCRsect.fits")
                pyfits.writeto(self.basename+"_NCRsect.fits",data=sectionarray)

        # bin the section if required
        if self.bin > 1:
            sectionx,sectiony = sectionarray.shape
            binx = sectionx/self.bin
            biny = sectiony/self.bin
            binarray = rebin(sectionarray,(binx,biny))
            # update SN location again to account for binning
            # -1 to put on zero-index scale
            self.binsnx = int(math.ceil(float(sectsnx+1.0)/self.bin))-1
            self.binsny = int(math.ceil(float(sectsny+1.0)/self.bin))-1
            if self.makeimages:
                # write binned array as fits file
                try:
                    pyfits.writeto(self.basename+"_NCRbin.fits",data=binarray)
                except IOError:
                    os.remove(self.basename+"_NCRbin.fits")
                    pyfits.writeto(self.basename+"_NCRbin.fits",data=binarray)
        # otherwise our binarray == sectionarray (.i.e no binning)
        else:
            binarray = sectionarray
            self.binsnx = sectsnx
            self.binsny = sectsny

        self.plotarray = binarray.copy()

        # get binned pixel value at SN location
        # because of way pyfits reads data, and numpy, index is [y,x]
        snbinpixvalue = binarray[self.binsny,self.binsnx]
        # make a sorted 1d list of pixel values
        pixelvalues = binarray.ravel()
        pixelvalues.sort()
        self.sortedpixelvalues = pixelvalues

        # find the index of the sorted array corresponding to the snlocation
        self.snpixnumber = np.where(pixelvalues==snbinpixvalue)

        # create cumulative sum distribution normalised to total sum of pixels
        totalvalue = float(np.sum(pixelvalues))
        self.cumarray = np.cumsum(pixelvalues)/totalvalue
        # find value of this cumulative sum at snpixnumber
        self.snncr = self.cumarray[self.snpixnumber][0]

        if totalvalue < 0:
            print "WARNING!: Total sum of pixel values in the selection is"
            print "          negative, setting NCR as NaN!."
            self.snncr = np.nan

        # find where value of cumsum is zero
        try:
            from scipy import interpolate
        except ImportError:
            print "ImportError:scipy.interpolate - ",
            print "NCRselection.eps may be bogus"
            # find the pixel nearest to zero in the cumsum if we have to..
            absc = np.abs(self.cumarray)
            minabsc = np.min(abscumarray)
            self.firstpospixelvalue = pixelvalues[np.where(absc==minabsc)][0]
        else:
            # .. but better to find the roots of a spline
            f = interpolate.UnivariateSpline(np.arange(self.cumarray.size),
                                         self.cumarray,s=0)
            try:
                self.firstpospixelvalue = pixelvalues[int(math.ceil(f.roots()\
                                                                    [-1]))]
            except IndexError:
                # i.e. the spline doesn't cross zero
                # This will ensure all pixels are highlighted if cumsum all +ve,
                # or none if cumsum all -ve
                self.firstpospixelvalue = 0.0
        
        if self.plot:
            self.makeplots()

        if self.negcut:
            print "\nNegative pixel clipping stats:"
            print ("\tmedian: %.4f, mean: %.4f, stddev: %.4f, datamin: %.4f,\n"\
                   "\tnumber of -ve pixels cut: %i" 
                   % (sectmedian,sectmean,sectstddev,sectdatamin,cutpixels))  

        print "\nFits file:         ",self.fitsfile+"[%i]" % self.ext
        print "Original SN coords:",self.snx,self.sny
        print "Original SN value:  %.4f" % self.snpixvalue
        print "Star centres:      ",self.starcentres
        print "Mask size:         ",self.masksize
        print "Section taken:     ","%i,%i,%i,%i" %\
                                               (self.x1,self.y1,self.x2,self.y2)
        print "Section SN coords: ",  sectsnx,sectsny   
        print "Binning factor:    ",self.bin
        print "Binned array size: ",binarray.shape[1],binarray.shape[0]
        print "Binned SN coords:  ",self.binsnx,self.binsny
        print "Binned SN value:    %.4f" % snbinpixvalue
        print "                    (pixel indices are zero-indexed!)"
        print "\nNCR value:          %.4f" % self.snncr

        # write some history cards to fits header if we used a fits file
        #self.writehistory()

        # revert stdout to normal
        sys.stdout = oldstdout

        return (self.snncr,self.snpixvalue,(self.binsnx,self.binsny),
                binarray,self.cumarray)

    def writehistory(self):
            
        now = datetime.now()
        hdu = pyfits.open(self.fitsfile,mode="update")
        history = hdu[self.ext].header.get_history()
        newhistory = [h for h in history if not any(str(h)[8:].startswith(x)\
                      for x in ("NCR","SN pixel","Section:"))]
        try:
            del hdu[self.ext].header["history"]
        except KeyError:
            # some pyfits versions will complain if "history" not in header
            pass
        for h in newhistory:
            try:
                hdu[self.ext].header.add_history(str(h).rstrip())
            except ValueError:
                pass
        hdu[self.ext].header.add_history("NCR = {0:.3f} on {1}"\
                                             .format(float(self.snncr),
                                                     now.strftime("%d %b %Y")))
        hdu[self.ext].header.add_history("SN pixel location (0-indexed): \
                                         {0},{1}".format(self.snx,self.sny))
        hdu[self.ext].header.add_history("Section: {0},{1},{2},{3}"\
                                         .format(self.x1,self.y1, 
                                                 self.x2,self.y2))
        hdu.flush()
        hdu.close()

    def makeplots(self):
        # plot the cumulative sum with the location of the SN's pixel marked
        plt.clf()
        x = np.arange(len(self.cumarray))
        plt.plot(x,self.cumarray,"k-",label="cumulative pixel distribution")
        plt.plot(self.snpixnumber[0]+1,self.cumarray[self.snpixnumber[0]],"ro",
                 label="SN")
        plt.axhline(y=0,color="r",linestyle="--")
        plt.ylim(np.min(self.cumarray),1)
        plt.xlim(0,len(self.cumarray))
        plt.ylabel("Cumulative fraction")
        plt.xlabel("Pixel number")
        plt.title(self.fitsfile+","+str(self.section)+","+str(self.bin),
                  fontsize=10)
        name = self.basename+"_NCRpix.eps"
        plt.savefig(name)

        # plot the image data (binned and sectioned as appropriate) with an
        # overlay showing those pixels that count positively to the cumsum
        plt.clf()
        sortedarray = np.sort(self.plotarray.ravel())
        npix = len(sortedarray)
        cutpix = math.ceil(0.05*npix)
        low_cut = sortedarray[cutpix]
        high_cut = sortedarray[npix-cutpix]
        newplotarray = np.clip(self.plotarray,low_cut,high_cut)

        plt.imshow(newplotarray,origin="lower",
                   interpolation="nearest",cmap="binary")
        plt.colorbar()
        # over plot the positions where pixels are above positive in cumsum
        overplotarray = newplotarray.copy()
        # make non-positive counting pixel transparent and all others equal
        overplotarray[np.where(self.plotarray<self.firstpospixelvalue)] = np.nan
        overplotarray[np.where(self.plotarray>=self.firstpospixelvalue)] = 1
        plt.imshow(overplotarray,origin="lower",
                   interpolation="nearest",cmap="autumn",alpha=0.8)
        plt.title("Pixels used in NCR analysis")
        plt.savefig(self.basename+"_NCRselection.png")

        # make a heatmap plot of NCR values
        plt.clf()
        sortindices = np.searchsorted(self.sortedpixelvalues,self.plotarray)
        heatplotarray = self.cumarray[sortindices]
        heatplotarray[heatplotarray<=0] = 0
        plt.imshow(heatplotarray,origin="lower",interpolation="nearest",
                   cmap="Spectral")
        self.heatplotarray = heatplotarray
        ax1 = plt.axes()
        ax1.autoscale(False)
        ax1.get_yaxis().set_visible(False)
        ax1.get_xaxis().set_visible(False)
        plt.plot(self.binsnx,self.binsny,marker="*",markersize=20,
                 markerfacecolor="none",markeredgewidth=1)
        color_bar = plt.colorbar(orientation='horizontal')
        cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
        plt.setp(cbytick_obj, color='k')
        plt.savefig(self.basename+"_NCRheatmap.eps",bbox_inches="tight")

        # rtmp: make heatmap of file using original headers. N.B. WCS will be invalid if file is sectioned
        hdulist = pyfits.open(self.fitsfile)
        hdrs = hdulist[0].header
        pyfits.writeto(self.basename+"_NCRheatmap.fits", data=self.heatplotarray, header=hdrs)
        hdulist.close()

    def _use_ds9(self,fitsfile,ext,snloc,masksize):
        """
        Allow interaction with ds9 to provide the section, star centres and 
        masksize.
        Can also show a DSS image of the region if coorindates are in WCS form.
        """
        masksize_original = masksize
        def initialsetup():
            self.d.set("frame frameno 1")
            self.d.set("file {0}".format(fitsfile))
            self.d.set("scale mode 99.5")
            #Centre on the SN
            self.d.set("pan to {0} {1} image".format(snloc[0],snloc[1]))
            #north up, east left
            self.d.set("align yes")
            #Mark the SN location
            self.d.set('regions', 'image; circle({0},{1},2) # width=2 color=red select=0'.format(snloc[0],snloc[1]))

        initialsetup()

        dssloaded = False
        x1,y1,x2,y2 = None,None,None,None
        section_drawn = False
        starcentres = []
        masksize = masksize_original

        print "Move cursor in ds9 window to set parameters"
        print " 's' - set corners of section (use twice)"
        print " 'w' - set section as whole image"
        print " 'c' - mark centres of regions to mask"
        print " 'z' - decrease the mask radius"
        print " 'x' - increase the mask radius"
        print " 'r' - reset frame (i.e. start over)"
        print " 'd' - download DSS frame and display adjacent, SN coordinates"
        print "       must be given in WCS for this to work!"
        print " 'q' - finish and continue to NCR calculation\n"
        while True:
            try:
                key,coox,cooy = self.d.get("imexam key coordinate image").split()
            except AttributeError:
                print "Problem contacting ds9 (closed?)"
                sys.exit()
            except ValueError:
                continue
            if key == "q":
                if all((x1,y1,x2,y2)):
                    break
                else:
                    print "section not defined!"
                    continue
            if key == "d":
                if self.snra and self.sndec and not dssloaded:
                    self.d.set("tile yes")
                    self.d.set("tile grid layout 2 1")
                    self.d.set("frame frameno 2")
                    self.d.set("dsssao frame current")
                    self.d.set("dsssao coord {0} {1}".format(self.snra,
                                                           self.sndec))
                    self.d.set("frame first")
                    dssloaded = True
                elif not dssloaded:
                    print "Need to give SN coordinates as WCS for DSS image"
            if key == "r":
                initialsetup()        
                x1,y1,x2,y2 = None,None,None,None
                section_drawn = False
                starcentres = []
                masksize = masksize_original
                print "section and mask centres cleared, masksize reset"
            if key == "s" and not all((x1,y1)):
                x1,y1 = [coox,cooy]
                self.d.set('regions', 'image; point({0} {1}) # point=x '
                           'color=blue width=2'.format(coox,cooy))
                print "first section corner at {0} {1}".format(coox,cooy)
            elif key == "s" and not all((x2,y2)):
                x2,y2 = [coox,cooy]
                self.d.set('regions', 'image; point({0} {1}) # point=x '
                           'select=0 color=blue width=2'.format(coox,cooy))
                print "second section corner at {0} {1}".format(coox,cooy)
            if key == "w" and not all((x1,y1,x2,y2)):
                # define in 1-indexed manner for region placement
                x1,y1,x2,y2 = 1,1,self.xsize-1,self.ysize-1
                print "section set to whole image"
            if all((x1,y1,x2,y2)) and not section_drawn:
                x1,y1,x2,y2 = map(float,(x1,y1,x2,y2))
                xl = min(x1,x2)
                xu = max(x1,x2)
                yl = min(y1,y2)
                yu = max(y1,y2)
                self.d.set('regions', 'image; box({0} {1} {2} {3} 0) # '
                           'select=0 width=2 color=blue'.format((xl+xu)/2,
                                                               (yl+yu)/2,
                                                               xu-xl,yu-yl))
                # correct to zero-indexed for NCR usage
                section = [xl-1,yl-1,xu-1,yu-1]
                section_drawn = True
            if key == "c":
                starcentres.extend([float(coox)+1,float(cooy)+1])
                print "mask region added at {0} {1}".format(coox,cooy)
                self.d.set('regions', 'image; point({0} {1}) # point=cross '
                           'select=0 color=yellow width=2'.format(coox,cooy))
            if key == "x":
                masksize += 1
                print "mask radius increased to {0} pixels".format(masksize)
            if key == "z":
                if masksize > 1:
                    masksize -= 1
                    print "mask radius decreased to {0} pixels".format(masksize)

        return (section,starcentres,masksize)

def rebin(a, shape):
    """
    Rebin an array, a, to shape. Uses the mean of the pixels in each
    bin as the value of the super pixel
    """
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)

def getshift(px,bin):
    """
    Calculates shift to apply to the section such that SN location is always
    the centre of a super pixel. The returned value needs to be cut from the
    lower bound of section in the axes being checked. (bin - shift) will be cut
    from the upper bound to ensure section length%bin == 0 still.
    """
    return ((px%bin)-range(bin)[(bin-1)/2])%bin


def mask_circle(array,xc,yc,r,dr=2):
    """
    Mask a circular refion from `array` centered on xc,yc with radius r.
    The pixel distribution to replace the masked region with is determined
    from an annulus of thickness `dr` arround the masked region.
    xc and yc must be >= r+dr from edge of array!
    """
    # cut a patch form the array centred on the region to mask
    patch_array = array[xc-r-dr:xc+r+dr+1,yc-r-dr:yc+r+dr+1]
    # get centres of this array relative to the centre
    x_centres = np.arange(-r-dr+0.5,r+dr+1,1)
    y_centres = np.arange(-r-dr+0.5,r+dr+1,1)
    xx,yy = np.meshgrid(x_centres,y_centres)

    # set pixels as mask region or annulus, as appropriate
    dist = xx * xx + yy * yy 
    mask = dist < r*r
    annulus = (dist > r*r) & (dist < (r+dr)*(r+dr))

    # instead of using all data in annulus...
    #std_ann = np.std(patch_array[annulus])
    #med_ann = np.median(patch_array[annulus])
    # ...do some quick clipping to remove and spikes in annulus
    ann_stats = clipdata(patch_array[annulus],nclip=2)
    # replace mask region with a pixel distribution from the annulus
    # seed so test will pass - still random for each mask in real image
    np.random.seed(abs(int(np.sum(patch_array)))) 
    patch_array[mask] = ann_stats[0]*np.random.randn(np.sum(mask))+ann_stats[3]
    # place the patch over the original array
    array[xc-r-dr:xc+r+dr+1,yc-r-dr:yc+r+dr+1] = patch_array

    return array


def clipdata(a,nclip=3,sigmaclip=3,nmin=5):
    """
    Iterate over an array `nclip` times, each time removing values > or < 
    `sigmaclip`*stddev away from mean.
    Return stddev and mean of this clipped array as well as datamin (the value
    `nmin`*stddev below clipped mean)
    """
    data3 = a.copy()
    for i in range(nclip-1):
        length = float(len(data3.ravel()))
        stddev = np.std(data3)
        mean = np.sum(data3)/length
        # remove any values < or > sigmaclip from mean
        clipmin = mean - sigmaclip * stddev
        clipmax = mean + sigmaclip * stddev
        # clip data
        data3 = data3[data3>clipmin] # returns ravelled array anyway
        data3 = data3[data3<clipmax]
    if length < 2:
        print "clipped data too small, using all pixels to calculate stats"
        data3 = a.copy()

    # calculate new stddev,mean and datamin
    stddev = np.std(data3)
    mean = np.sum(data3)/len(data3.ravel())
    median = np.median(data3)
    datamin = mean - nmin * stddev
    return stddev,mean,datamin,median

def cootoxy(image,ra,dec):
    """
    Given an ra and dec in sexigesimal form, this will return the pixel
    coordinates of that location in the image, make sure the WCS is correct!
    """
    from pyraf import iraf
    IN = "/tmp/cootoxy.in"
    OUT = "/tmp/cootoxy.out"
    for f in (IN,OUT):
        try:
            os.remove(f)
        except OSError:
            pass

    with open(IN,"w") as infile:
        infile.write("{0} {1}".format(ra,dec))
    # create a file with pixel coordinates
    iraf.wcsctran(input = IN,
                  output = OUT,
                  image = image,
                  inwcs = "world",
                  outwcs = "logical",
                  units="h n",
                  formats="%8.3f %8.3f",
                  columns = "1 2")
    # return x and y as floats
    return map(float,open(OUT,"r").readlines()[-1].strip().split())


def test():
    """
    Simple test to check that the NCR calculations remain correct
    """
    pyfits.writeto("/tmp/ncrtest.fits",np.arange(1600).reshape(40,40),
                   clobber=True)
    n = NCR(fitsfile="/tmp/ncrtest.fits",
            snlocation=(18,15),
            section=[3,7,30,25],
            ext=0,
            bin=3,
            starcentres=[10,20],
            masksize=3,     
            plot=False,
            makeimages=False)
    ret =  n.main()

    print "\nRunning tests..."
    assert ret[0] == 0.40689226780457494 # NCR value
    assert ret[1] == 618 # pixel value
    assert ret[2] == (4,2) # sectioned/binned coords
    assert ret[3].shape == (5,8) # sectioned/binned array size
    assert len(ret[4]) == 40 # length of cumulative distribution
    print "Passed!"

if __name__ == "__main__":
    # just run test if required
    if sys.argv[-1] == "-t":
        test()
        sys.exit()

    parser = argparse.ArgumentParser(
                      description='Perform NCR analysis on a FITS file. Allows'
                      ' trimming of image and binning, as well as masking of '
                      'star residuals and clipping of abnormally negative '
                      'pixel values. If -t is last argument, just run test and'
                      ' exit.')
    parser.add_argument('fitsfile',type=str,help='FITS file for analysis.')
    parser.add_argument('snlocation',type=str,help='image pixel coordinates '
                        'of SN in format `x,y`, zero-indexed! Alternatively '
                        'ra and dec of SN can be given as `hh:mm:ss,dd:mm:ss`'
                        '(requires PyRAF).')
    parser.add_argument('-s','--section',type=str,default=None,help='section '
                        'of image for analysis in the format:`x1,y1,x2,y2`, '
                        'limits half-open, zero-indexed! Default = wholeimage.')
    parser.add_argument('-b','--bin',type=int,default=1,help='bin pixels into '
                        '`bin`x`bin` squares before analysis. Default = 1.')
    parser.add_argument('-p','--plot',action='store_true',help='plot the '
                        'cumulative sum of pixel value and image of pixels '
                        'used as well as a heatmap of NCR values. '
                        'saved in image\'s dir with filenames.'
                        '*_NCRpix.eps, *_NCRselection.png, *_NCRheatmap.eps.')
    parser.add_argument('-i','--makeimages',action='store_true',help='create '
                        'fits files of the cropped section and binned section. '
                        ' saved in image\'s dir with filenames '
                        '*_NCRsect.fits *_NCRbin.fits.')
    parser.add_argument('-nc','--negcut',action='store_true',help='set pixels '
                        '< mean-5*skystddev to 0.')
    parser.add_argument('-c','--constant',type=float,default=0,help='value to '
                        'SUBTRACT from all image pixel values. Default = 0.')
    parser.add_argument('-sc','--starcentres',type=str,default=None,help='list '
                        'of x,y coordinates to mask over of the form '
                        'x1,y1,x2,y2... Must be > masksize+2 pixels from edge '
                        'of image, otherwise prepare for index errors. '
                        'zero-indexed! Default = None.')
    parser.add_argument('-m','--masksize',type=int,default=10,help='radius of'
                        ' mask region to patch over star centres.'
                        ' Default = 10.') 
    parser.add_argument('-e','--ext',type=int,default=0,help='extension of the '
                        'fits files which contains the data for NCR analysis.')
    parser.add_argument('--ds9',action='store_true',help='use ds9 to select '
                        'the section,starcentres and masksize, using this '
                        'option will override -s,-sc and -m (requires pyds9)')
    parser.add_argument('-q','--quiet',action='store_true',help='supress '
                        'output to stdout (log file still created)')
    arg_dict = vars(parser.parse_args())

    # call class and run the main program
    n = NCR(**arg_dict) 
    n.main()

