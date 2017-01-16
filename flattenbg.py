#! /usr/bin/env python

#TODO Allow negative numbers for boxsize to mean 'split into this many sections'
#TODO Add griddata interpolation to use linear/cubic and interpolate at every
#     pixel (cpu time too long?)
import os,sys
import argparse
import shutil
from datetime import datetime

import pyfits
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as si


CLIP_OPTIONS = ("objmask","sigma")
INTERP_OPTIONS = ("smoothspl","linear","rectspl",) #FIXME
RV = np.nan  # replacement value for bad points found in median mesh

class MyParser(argparse.ArgumentParser):
    """
    wrapper for argparse.ArgumentParser

    custom error behaviour - will automatically show usage help on argument
    error
    """
    def error(self, message):
        sys.stderr.write('ERROR\t%s\n' % message)
        self.print_usage()
        sys.stderr.write('type -h or --help to display full help\n')
        sys.exit(2) 


class SurfaceFit(object):
    """
    Fit a surface to background pixels of an image and output a version
    of that image file with the surface fit subtracted from the pixel values
    in the specified extension.
    Probably a bit shit if you have big extended objects in the image
    """
    def __init__(self,image,output=None,ext=0,boxsize=101,skim=False,
                 cliptype="objmask",interp="rect",xo=3,yo=3,nsigma=3,nclip=3):

        try:
            open(image)
        except IOError:
            raise Exception("Input image cannot be opened: {}".format(image))
        else:
            self.image = image
        if output is None:
            self.output = os.path.splitext(self.image)[0]+".surf.fits"
        else:
            self.output = output
        try:
            open(self.output)
        except:
            pass
        else:
            raise Exception("output file already exists: {}".format(self.output))

        self.ext = ext

        if boxsize%2 == 0:
            raise Exception("boxsize must be odd")
        self.boxsize = boxsize

        self.skim = skim

        self.smooth = 0.1

        if cliptype not in CLIP_OPTIONS:
            raise Exception("clipping choice must be: '{}'"\
                                            .format("','".join(CLIP_OPTIONS)))
        self.cliptype = cliptype

        if interp not in INTERP_OPTIONS:
            raise Exception("interpolation choice must be: '{}'"\
                                            .format("','".join(INTERP_OPTIONS)))
        self.interp = interp

        if xo not in range(1,6) or yo not in range(1,6):
            raise Exception("xorder and yorder must be between 1 and 5")
        self.xo,self.yo = xo,yo

        self.nsigma = nsigma
        self.nclip = nclip

    def getclippeddata(self,array=None):
        """
        Return an array based on the input image that has been clipped of sources
        """
        if self.cliptype == "objmask":
            return objmaskdata(self.image)
        elif self.cliptype == "sigma":
            if array is None:
                try:
                    array = pyfits.getdata(self.image,self.ext)
                except IndexError:
                    raise Exception("couldn't get data from extension {} in {}"\
                            .format(self.ext,self.image))
            return sigmaclipdata(array,nclip=self.nclip,nsigma=self.nsigma)
        else:
            raise Exception("Bad value of cliptype: {}".format(self.cliptype))

    def main(self):

        print "getting data"
        imagedata = self.getclippeddata()

        xsize,ysize = imagedata.shape

        # find out size of border that is set to zero so we ignore it when
        # removing surface from image
        trim = np.sum(imagedata[xsize/2,:]==0)/2 # better way to do this?

        ignoretrim = True if not trim else False

        # make a grid of points about which to find locan medians
        # TODO ensure we have at least a few in each direction so we don't have
        #      empty arrays!
        print "filling sparesly sampled mesh with local medians"
        box = self.boxsize
        xcentres = np.arange(box/2+trim,xsize-(box+1)/2-trim,box)
        if xcentres[-1] != xsize-(box+1)/2-trim:
            xcentres = np.append(xcentres,xsize-(box+1)/2-trim)
        ycentres = np.arange(box/2+trim,ysize-(box+1)/2-trim,box)
        if ycentres[-1] != ysize-(box+1)/2-trim:
            ycentres = np.append(ycentres,ysize-(box+1)/2-trim)

        # make a mesh of coordinates to hold the indices of these points
        ymesh,xmesh = np.meshgrid(ycentres,xcentres)
        zmesh = np.zeros(xmesh.shape)

        # iterate over these points and take a local median and use this
        # as a value in out sparesly sampled grid
        c = 0
        r = 0
        badareas = 0
        maxc = xmesh.shape[1]
        for xcent,ycent in zip(xmesh.ravel(),ymesh.ravel()): 
            imsect = imagedata[xcent-box/2:xcent+(box+1)/2,
                               ycent-box/2:ycent+(box+1)/2]
            if np.sum(np.isnan(imsect)) > box*box/5.:
                badareas+=1
                zmesh[r,c] = RV
            else:
                medval = np.median(imsect[~np.isnan(sigmaclipdata(imsect,
                                                                  1,2.5,1))])
                zmesh[r,c] = medval
            c += 1
            if c == maxc:
                c = 0
                r += 1
        print "found {} mesh point(s) dominated by objects".format(badareas)

        print "plotting median mesh"
        # plot median values
        zmask = np.ma.array(zmesh, mask=(np.isnan(zmesh)))
        plt.clf()
        plt.imshow(zmask,origin="lower",interpolation="nearest",cmap="jet")
        plt.colorbar()
        ax1 = plt.axes()
        ax1.get_yaxis().set_visible(False)
        ax1.get_xaxis().set_visible(False)
        plt.savefig("medianpoints.eps",bbox_inches="tight")

        # interpolate over this grid with a finer mesh of 1 point per pixel
        # (x/y orders reversed due to way numpy reads in fits array)
        print "interpolating median mesh over all pixels"
        if self.interp == "smoothspl":
            zmesh_values = ~np.isnan(zmesh.ravel())
            spl = si.SmoothBivariateSpline(xmesh.ravel()[zmesh_values],
                                           ymesh.ravel()[zmesh_values],
                                           zmesh.ravel()[zmesh_values],
                                           kx=self.yo,ky=self.xo,
                                           bbox=[0,xsize,0,ysize],
                                           s=len(zmesh_values)*self.smooth)
            zn = spl(np.arange(xsize),np.arange(ysize))
        elif self.interp == "linear":
            grid_x,grid_y= np.mgrid[0:xsize,0:ysize]
            zn = si.griddata((xmesh.ravel(),ymesh.ravel()),zmesh.ravel(),
                             (grid_x,grid_y),fill_value=0)
        elif self.interp == "rectspl":
            zmesh = np.nan_to_num(zmesh)
            spl = si.RectBivariateSpline(xcentres, ycentres, zmesh,
                                         kx=self.yo,ky=self.xo,
                                         bbox=[0,xsize,0,ysize])
            zn = spl(np.arange(xsize),np.arange(ysize))

        # plot the interpolation function
        print "plotting interpolation"
        plt.clf()
        iplt = plt.imshow(zn[::30,::30],origin="lower",interpolation="bilinear")
        iplt.set_clim(np.min(zmask),np.max(zmask))
        plt.colorbar()
        ax1 = plt.axes()
        ax1.get_yaxis().set_visible(False)
        ax1.get_xaxis().set_visible(False)
        plt.savefig("pixelpoints.eps",bbox_inches="tight")

        # copy image and remove the interpolation function from pixel values
        print "subtracting interpolated function from image"
        shutil.copy(self.image,self.output)
        hdu = pyfits.open(self.output,mode="update")
        # just remove the form of the surface, not the absolute value
        if self.skim:
            zn -= np.median(zn)
        if ignoretrim:
            hdu[self.ext].data -= zn
        else:
            hdu[self.ext].data[trim:-trim,trim:-trim] -= zn[trim:-trim,trim:-trim]
        now = datetime.now()
        hdu[self.ext].header.add_history("flattened with flattenbg.py on {}".format(
                (now.strftime("%d %b %Y"))))
        hdu[self.ext].header.add_history("boxsize:{},xo:{},yo:{},cliptype:{}"\
                                          .format(self.boxsize,self.xo,self.yo,
                                                  self.cliptype))
        
        hdu.flush()
        hdu.close()
        print "flattened image is at: {}".format(self.output)
        print "flattening done!"

def sigmaclipdata(array,nclip=3,nsigma=3,nsect=1):
    """
    Iterate over an array `nclip` times, each time removing values > or < 
    `sigmaclip`*stddev away from mean. nsect value will split the 
    original array into `nsect` slices along each axis and perform
    median/stddev and clipping in each section serparately before
    rejoining the array. nsect=1 just uses the full array as normal
    Return an array that is zero where values have been clipped
    """
    #print "sigma clipping data"
    data = array.copy()
    clippeddata = array.copy()

    numclipped = 0

    for i in range(nclip):
        #print "clip {}/{}".format(i+1,nclip),
        sections = [np.array_split(j,nsect,1) for j in np.array_split(data,nsect)]
        maskrow = []
        for row in sections:
            maskcol = []
            for column in row:
                cdata = column
                
                stddev = np.std(cdata[~np.isnan(cdata)])
                median = np.median(cdata[~np.isnan(cdata)])
                clipmin = median - nsigma * stddev
                clipmax = median + nsigma * stddev
                # flag data outside these limits
                belowclipmin = (cdata < clipmin)
                aboveclipmax = (cdata > clipmax)
                # we'll assume a value cannot be < clipmin AND > clipmax
                mask = (belowclipmin != aboveclipmax)
                maskcol.append(mask)
            maskrow.append(np.hstack(maskcol))
        maskfull = np.vstack(maskrow)

        # set all values outside the clips to zero
        clippeddata[maskfull] = RV
        clippeddata[maskfull] = RV

        newclipped = np.sum(maskfull)
        #print "\tremoved {} pixels".format(newclipped)
        numclipped += newclipped
        if newclipped == 0:
            break
        # copy this newly clipped data for another iteration if required
        data = clippeddata.copy()    

    return clippeddata

def objmaskdata(image):
    """
    Return the data from an image that is masked of objects by the iraf
    task nproto.objmasks
    """
    print "iraf object masking data"
    from pyraf import iraf
    iraf.nproto(Stdout=1)
    data = pyfits.getdata(image)
    try:
        os.remove("flattenbg_objmask.fits")
    except OSError:
        pass
    om = iraf.objmasks(image,"flattenbg_objmask.fits",omtype="boolean",
                       Stdout=1)
    mask = pyfits.getdata("flattenbg_objmask.fits",1)
    mask = mask==1
    data[mask] = RV
    os.remove("flattenbg_objmask.fits")

    return data


if __name__ == '__main__':

    parser = MyParser(description='Subtraction pipeline for'
                                  ' transient observations')
    parser.add_argument('image', type=str, help='image to flatten')
    parser.add_argument('-outimage', type=str, default=None,
                        help='output file name (default: inimage.surf.fits'
                        'remove if directory exists)')
    parser.add_argument('-e', dest='ext', type=int, default=0,
                        help='extension in image to flatten (default: 0)')
    parser.add_argument('-i', dest='interp', type=str, default='smoothspl',
                        help='interpolation to use "smoothspl","rectspl" or '
                        '"linear". rect and linear require artifical insetion '
                        ' of zero values at meshpoints dominated by objects '
                        '(default: "smoothspl")')
    parser.add_argument('-c', dest='cliptype', type=str, default='objmask',
                        help='clipping to use to mask objects. Either remove'
                        ' pixel `nsigma` stddeviations away from image median '
                        'with `nclip` iterations -"sigma", or use iraf object'
                        ' masking that is probably more adept to completely '
                        'remove objects from background estimation - '
                        '"objmask" (default: "objmask")')
    parser.add_argument('-b', dest='boxsize', type=int, default=101,
                        help='size of boxes in pixels used to sample '
                        'median across image (default: 101)')
    parser.add_argument('-s', dest='skim', action='store_true',
                        help='just remove the form of the surface and not the'
                        ' absolute value (removes the median surface value '
                        'before subtracting it from the image')
    parser.add_argument('-xo',dest='xorder',type=int, default=3,
                        help='order of spline to fit in x direction')
    parser.add_argument('-yo',dest='yorder',type=int, default=3,
                        help='order of spline to fit in y direction')
    parser.add_argument('-nsigma',type=float, default=3,
                        help='if cliptype is `sigma`, remove data more than '
                        'nsigma stddeviations from image median before fitting'
                        ' background. default=3')
    parser.add_argument('-nclip',type=int, default=3,
                        help='number of iterations of the nsigma clip. default=3')

    args = parser.parse_args()

    surfit = SurfaceFit(args.image,args.outimage,args.ext,args.boxsize,
                        args.skim,args.cliptype,args.interp,args.xorder,
                        args.yorder,args.nsigma,args.nclip)
    surfit.main()



