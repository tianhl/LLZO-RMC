# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 12:54:41 2020

@author: blues
"""
ppdf={1:'Li-Li', 2:'Li-O', 3:'Li-Zr', 4:'Li-La', 5:'O-O', 6:'O-Zr', 7:'O-La', 8:'Zr-Zr', 9:'Zr-La', 10:'La-La'}
ppdfwithLi={1:'Li-Li', 2:'Li-O', 3:'Li-Zr', 4:'Li-La'}
ppdfwithoutLi={5:'O-O', 6:'O-Zr', 7:'O-La', 8:'Zr-Zr', 9:'Zr-La', 10:'La-La'}



total = 56.0+24.0+16.0+96.0

NLifi=-1.9
XLifi=3.0
Lici=56.0/total

NLafi=8.24
XLafi=57.0
Laci=24.0/total

NZrfi=7.16
XZrfi=40.0
Zrci=16.0/total

NOfi=5.803
XOfi=8.0
Oci=96.0/total

Nfi={'Li':NLifi, 'La':NLafi, 'Zr':NZrfi, 'O':NOfi}
Xfi={'Li':XLifi, 'La':XLafi, 'Zr':XZrfi, 'O':XOfi}
Ci={'Li':Lici, 'La':Laci, 'Zr':Zrci, 'O':Oci}



class ConstantValue:
    TEM_LIST=['293','450','600','750','900','1100']
    
    COLOR_293   = 'b'
    COLOR_450   = 'c'
    COLOR_600   = 'g'
    COLOR_750   = 'y'
    COLOR_900   = 'm'
    COLOR_1100  = 'r'
    
    DLPOLY_293   ='../md.data/mdplotfiles/VAFDAT_293K_0'
    DLPOLY_450   ='../md.data/mdplotfiles/VAFDAT_450K_0'
    DLPOLY_600   ='../md.data/mdplotfiles/VAFDAT_600K_0'
    DLPOLY_750   ='../md.data/mdplotfiles/VAFDAT_P750K_0'
    DLPOLY_900   ='../md.data/mdplotfiles/VAFDAT_P900K_0'
    DLPOLY_1100  ='../md.data/mdplotfiles/VAFDAT_P1100K_0'
    
    GSAS_293   = '../gsas.results/293K'
    GSAS_450   = '../gsas.results/450K'
    GSAS_600   = '../gsas.results/600K'
    GSAS_750   = '../gsas.results/P750K'
    GSAS_900   = '../gsas.results/P900K'
    GSAS_1100  = '../gsas.results/1100K'
    
    RMC_293   = '../rmc.data/rmcplotfiles/LLZO_RMC6_293K'
    RMC_450   = '../rmc.data/rmcplotfiles/LLZO_RMC6_450K'
    RMC_600   = '../rmc.data/rmcplotfiles/LLZO_RMC6_600K'
    RMC_750   = '../rmc.data/rmcplotfiles/LLZO_RMC6_P750K'
    RMC_900   = '../rmc.data/rmcplotfiles/LLZO_RMC6_P900K'
    RMC_1100  = '../rmc.data/rmcplotfiles/LLZO_RMC6_P1100K'
    
    
    XPDF_293  = '../i15-1.xpdf/01-027/i15-1-18935_tth_det2_0.dofr'  
    XPDF_450  = '../i15-1.xpdf/01-177/i15-1-18936_tth_det2_0.dofr'  
    XPDF_600  = '../i15-1.xpdf/01-327/i15-1-18937_tth_det2_0.dofr'  
    XPDF_750  = '../i15-1.xpdf/01-477/i15-1-18938_tth_det2_0.dofr'  
    XPDF_900  = '../i15-1.xpdf/01-627/i15-1-18939_tth_det2_0.dofr'  
    XPDF_1100 = '../i15-1.xpdf/01-827/i15-1-18940_tth_det2_0.dofr'  
   
    rmc_npdf_suffix = '_PDF1.csv'
    rmc_xpdf_suffix = '_FT_XFQ1.csv'
    rmc_partialpdf_suffix = '_PDFpartials.csv'
    rmc_rmc6f_suffix = '.rm6f'
    gsas_lst_suffix='.LST'
    
    cellParameters={1100:13.08634,900:13.04123,750:13.00967,600:12.97967,450:12.95247,293:12.92195}   

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc
c = ConstantValue


def getDataByTemperature(temperature='293'):
    x,y,e=getDataByTemperatureByIdx(temperature,idx='0')
    for i in range(1,6):
        x1,y1,e1 =getDataByTemperatureByIdx(temperature,idx=str(i))
        y=y+y1
    y=y/6
    return x,y,e    

def getDataByTemperatureByIdx(temperature='300',idx='0'):

    #dlpoly_path = getattr(c,'DLPOLY_'+temperature)
    rmc_path    = getattr(c,'RMC_'+temperature)
    filename = rmc_path+'_'+idx+c.rmc_npdf_suffix
    print(filename)
    rmc_x,rmc_pdf, exp_pdf=rmcloaddata(filename)
    #return dl_x, dl_npdf, rmc_x, rmc_pdf, exp_pdf
    return rmc_x, rmc_pdf, exp_pdf

def rmcloaddata(filename):
    if((filename is None)and(suffix is not None)):
        filename = self.default+suffix
    print('Read  file: ' + filename)
        
    f=open(filename)
    l=f.readlines()
    x_array = []
    rmc_array = []
    exp_array = []
    for line in l:
        i = line.split(',')
        if(not is_number(i[1])):
            continue
        if(len(i)<3):
            continue
        x_array.append(float(i[0]))
        rmc_array.append(float(i[1]))
        exp_array.append(float(i[2]))
    return np.array(x_array), np.array(rmc_array), np.array(exp_array)

def definePlot(xup=25, xlow=0, grid = True):
    if grid == True:
        plt.grid()
    plt.xticks(fontsize=75)
    plt.yticks(fontsize=75)
    plt.xlim((xlow,xup))
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(4)
    ax.spines['left'].set_linewidth(4)
    ax.spines['right'].set_linewidth(4)
    ax.spines['top'].set_linewidth(4)

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False






def getRmcGsasBragg(temperature='293'):
    x,g,r=getRmcGsasBraggByIdx(temperature,idx='0')
    for i in range(1,6):
        x1,g1,r1 =getRmcGsasBraggByIdx(temperature,idx=str(i))
        r=r+r1
    r=r/6
    return x,g,r    

def getRmcGsasBraggByIdx(temperature='293', idx='0',filename = False):
    rmc_path=getattr(c,'RMC_'+temperature)
    filename=rmc_path+'_'+idx+'.braggout'
    print('read rmc braggout file: '+filename)
    return getRmcGsasBraggByName(filename)

def getRmcGsasBraggByName(filename):
    f=open(filename)
    lines=f.readlines()
    x=[]
    g=[]
    r=[]
    for line in lines:
        x.append(float(line.split()[0])/1000.)
        g.append(float(line.split()[1]))
        r.append(float(line.split()[2]))
    
    return np.array(x),np.array(g),np.array(r)


    


def readtable(filename,scale=1):
    f=open(filename)
    l=f.readlines()
    x_array = []
    y_array = []
    for line in l:
        if(line[0].startswith("#")):
            continue
        i = line.split()
        if(len(i)<2):
            continue
        x_array.append(float(i[0]))
        y_array.append(float(i[1]))
    return np.array(x_array), np.array(y_array)*scale

def getXPDFdofr(temperature, scale=1):
    c = ConstantValue
    filename=getattr(c,'XPDF_'+temperature)
    print('read file: '+filename)
    return readtable(filename,scale) 
    

def getRMCXPDFDataByTemperature(temperature='293'):
    x,y=getRMCXPDFDataByTemperatureByIdx(temperature,idx='0')
    for i in range(1,6):
        x1,y1, =getRMCXPDFDataByTemperatureByIdx(temperature,idx=str(i))
        y=y+y1
    y=y/6
    return x,y*x    

def getRMCXPDFDataByTemperatureByIdx(temperature='293', idx='1'):
    c = ConstantValue
    rmc_path    = getattr(c,'RMC_'+temperature)
    filename = rmc_path+'_'+idx+c.rmc_xpdf_suffix
    print(filename)
    rmc_x, rmc_pdf, exp_pdf = rmcloaddata(filename)
    return rmc_x, rmc_pdf#, dlp_x, dlp_pdf*1E-4

def getNSoqByTemperature(temperature='293'):
    x,c,e = getNSoqByTemperatureByIdx(temperature=temperature, idx = '0')
    for i in range(1,6):
        x1,c1,e1 = getNSoqByTemperatureByIdx(temperature=temperature, idx = str(i))
        c = c+c1
        e = e+e1
    c=c/6
    e=e/6
    return x,c,e    
    pass


def getNSoqByTemperatureByIdx(temperature='293', idx='0'):
    c = ConstantValue
    soq_path=getattr(c,'RMC_'+temperature)
    filename = soq_path+'_'+idx+'_SQ1.csv'
    print(filename)
    return getNSoqByTemperatureByName(filename)
    
def getNSoqByTemperatureByName(filename):
    f=open(filename)
    lines=f.readlines()
    x=[]
    c=[]
    e=[]
    for line in lines[1:]:
        x.append(float(line.split(',')[0]))
        c.append(float(line.split(',')[1]))
        e.append(float(line.split(',')[2]))
    
    return np.array(x),np.array(c),np.array(e)

def getXSoqByTemperature(temperature='293'):
    x,c,e = getXSoqByTemperatureByIdx(temperature=temperature, idx = '0')
    for i in range(1,6):
        x1,c1,e1 = getXSoqByTemperatureByIdx(temperature=temperature, idx = str(i))
        c = c+c1
        e = e+e1
    c=c/6
    e=e/6
    return x,c,e    
    pass

def getXSoqByTemperatureByIdx(temperature='293', idx = '0'):
    c = ConstantValue
    soq_path=getattr(c,'RMC_'+temperature)
    filename = soq_path+'_'+idx+'_XFQ1.csv'
    print(filename)
    return getXSoqByTemperatureByName(filename)
    
def getXSoqByTemperatureByName(filename):    
    f=open(filename)
    lines=f.readlines()
    x=[]
    c=[]
    e=[]
    for line in lines[1:]:
        x.append(float(line.split(',')[0]))
        c.append(float(line.split(',')[1]))
        e.append(float(line.split(',')[2]))
    
    return np.array(x),np.array(c),np.array(e)

def getRMCGR(dirname='../rmc.data/rmcplotfiles/', temperature = '293', idx='0'):
    if int(temperature) > 700:
        temperature = 'P'+temperature
    filename=dirname+'LLZO_RMC6_'+temperature+'K_'+idx+'_PDFpartials.csv'
    print(filename)
    f     =open(filename)
    lines =f.readlines()
    x    =[]
    LiLi =[]
    LiO  =[]
    LiZr =[]
    LiLa =[]
    OO   =[]
    OZr  =[]
    OLa  =[]
    ZrZr =[]
    ZrLa =[]
    LaLa =[]
    ys   = {}
    for line in lines:
        i = line.split(',')
        if(not is_number(i[1])):
            continue
        x.append(float(i[0]))
        LiLi.append(float(i[1])) 
        LiO.append(float(i[2])) 
        LiZr.append(float(i[3])) 
        LiLa.append(float(i[4])) 
        OO.append(float(i[5])) 
        OZr.append(float(i[6])) 
        OLa.append(float(i[7])) 
        ZrZr.append(float(i[8])) 
        ZrLa.append(float(i[9])) 
        LaLa.append(float(i[10]))
    ys['Li-Li']   = np.array(LiLi)    
    ys['Li-O']    = np.array(LiO)
    ys['Li-Zr']   = np.array(LiZr)
    ys['Li-La']   = np.array(LiLa)
    ys['O-O']     = np.array(OO)
    ys['O-Zr']    = np.array(OZr)
    ys['O-La']    = np.array(OLa)
    ys['Zr-Zr']   = np.array(ZrZr)
    ys['Zr-La']   = np.array(ZrLa)
    ys['La-La']   = np.array(LaLa)
    return x, ys

def getDlPolyGR(dirname='../md.data/mdplotfiles/', temperature = '293', idx='0'):
    if int(temperature) > 700:
        temperature = 'P'+temperature
    filename=dirname+'RDFDAT_'+temperature+'K_'+idx
    print('read file: '+filename)
    fi = open(filename)
    ys   = {}
    keys = ''
    x    = []
    y    = []
    LiLi =[]
    LiO  =[]
    LiZr =[]
    LiLa =[]
    OO   =[]
    OZr  =[]
    OLa  =[]
    ZrZr =[]
    ZrLa =[]
    LaLa =[]
    li = fi.readlines()
    for line in li[2:]:
        items = line.split()
        if is_number(items[0]):
            x.append(float(items[0]))
            y.append(float(items[1]))
            pass
        else:
            if keys is '':
                #print('first')
                pass
            else:
                #print('save: '+keys)
                ys[keys]=np.array(y)
            keys='-'.join(items)
            x = []
            y = []
            #print(keys)
    
    
    #return x, LiLay ,LaLay ,ZrLay ,OLay  ,ZrZry ,LiZry ,OZry  ,LiLiy ,LiOy  ,OOy   
    return x, ys


def correlationfunction(dirname='../md.data/mdplotfiles/', temperature = '293', idx='0', element='Li'):
    if int(temperature) > 700:
        temperature = 'P'+temperature
    filename=dirname+'VAFDAT_'+temperature+'K_'+idx+'_'+element
    print('open: '+filename)
    f = open(filename)
    lines = f.readlines()
    x=[]
    y=[]
    for line in lines[2:]:
        items = line.split()
        x.append(float(items[0]))
        y.append(float(items[1]))
    return np.array(x), np.array(y)    

def powerSpectra(x,y):
#    fft_y1 = fft.dst(y,type=2,norm='ortho')
#    fft_y2 = np.fft.fft(y)
    import math
    from scipy import fftpack as fft
    fft_y = fft.dct(y)
    xbin  = 1/math.pi
    fft_x =np.array([i*xbin for i in range(len(fft_y))])
    print(fft_x)
    diffuse_const = fft_y[0]
    return fft_x, fft_y, diffuse_const


def plotPowerSpectra(element='Li', has_xlabel=False):
    c  = ConstantValue

    definePlot(xup=50.0, xlow=0)
    #x_ticks = np.arange(0,50, 10)

    for temp in c.TEM_LIST:
        color    = getattr(c,'COLOR_'+temp)
        x,y = correlationfunction(temperature = temp,element = element)
        for i in range(1,6):
            x2,y2 = correlationfunction(temperature = temp, idx=str(i),element = element)
            y = y+y2
        y=y/6.0    

        fft_x,fft_y,diffusion = powerSpectra(x,y)
        #d.append(diffusion)
        #t.append(float(temperature))
        
        plt.plot(fft_x,fft_y, color, linewidth=5)
        if has_xlabel:
            plt.xlabel('frequency($\mathrm{THz}$)',fontsize=100)
        plt.ylabel(r'Z$_{'+element+'}$($\mathrm{THz}$)',fontsize=100)
        #plt.xticks(x_ticks)

    #plt.grid() 


def plotPartialGofr(item='Li-Li', has_xlabel = False):
    definePlot(12)
    offset = 0.
    for temp in c.TEM_LIST:
        color = getattr(c,'COLOR_'+temp)
        rmcx,rmcys=getRMCGR(temperature=temp, idx='0')
        mdx ,mdys =getDlPolyGR(temperature=temp, idx='0')
        for i in range(1,6):
            mdx2, mdys2  = getDlPolyGR(temperature=temp, idx=str(i))
            rmcx2,rmcys2 = getRMCGR(temperature=temp, idx=str(i))
            rmcys[item]= rmcys[item]+rmcys2[item]
            mdys[item] = mdys[item]+mdys2[item]
        mdys[item] = mdys[item]/6.
        rmcys[item]= rmcys[item]/6
        rmcy = np.array(rmcys[item])
        mdy  = np.array(mdys[item])
        plt.plot(rmcx, rmcy+offset, color+'o',linewidth=5)
        plt.plot(mdx,  mdy+offset, 'black',linewidth=5)
        offset += 5.0
    if has_xlabel:    
        plt.xlabel(r'r ($\mathrm{\AA}$)',fontsize=100)
    plt.ylabel('$\mathrm{g_{'+item+'}}$',fontsize=100)
    

def getPartialNPDFbyTemperature(item='Li-Li', temperature='293'):

    rmcx,rmcys=getRMCGR(temperature=temperature, idx='0')
        #mdx ,mdys =getDlPolyGR(temperature=temp, idx='0')
    for i in range(1,6):
            #mdx2, mdys2  = getDlPolyGR(temperature=temp, idx=str(i))
        rmcx2,rmcys2 = getRMCGR(temperature=temperature, idx=str(i))
        rmcys[item]= rmcys[item]+rmcys2[item]
            #mdys[item] = mdys[item]+mdys2[item]
        #mdys[item] = mdys[item]/6.
    rmcys[item]= rmcys[item]/6
    rmcy = np.array(rmcys[item])
        #mdy  = np.array(mdys[item])
    element0, element1 = item.split('-')    
    dofr = getDofR(rmcx, rmcy, Ci[element0], Ci[element1], Nfi[element0], Nfi[element1])
    return rmcx, dofr

def getTotalNPDFbyTemperature(temperature='293'):
    ppdfs={1:'Li-O', 2:'O-O', 3:'O-Zr', 4:'O-La', 5:'Li-Zr', 6:'Li-La', 7:'Li-Li', 8:'Zr-La', 9:'Zr-Zr', 10:'La-La'}
    rmcx, dofr = getPartialNPDFbyTemperature(ppdfs[1], temperature=temperature)
    for key in range(2,11):
        x,y = getPartialNPDFbyTemperature(ppdfs[key],temperature)
        dofr = y+dofr
    return rmcx, dofr    

def getDofR(np_x, np_y, ci, cj, fi, fj):
    pi=3.14
    rho=5.07
    factor=4*pi*rho*2*fi*fj*ci*cj

    tmp0 = np.subtract(np_y, 1.0)
    tmp1 = np.multiply(tmp0, np_x)
    #tmp2 = np.multiply(factor, tmp1)
    dr = np.multiply(tmp1, factor)
    #dr   = np.subtract(tmp3, offset)
    return dr


##############################################################################

def plotCellParametersVstemperature():
    definePlot(1150,250)
    ts=c.TEM_LIST
    cells=[]
    temps=[]
    for t in ts:
        cells.append(c.cellParameters[int(t)])
        temps.append(int(t))
    cells=np.array(cells)
    temperatures=np.array(temps)
    print(cells)
    print(temperatures)
    
    rg=np.polyfit(temperatures, cells, 1)
    ry=np.polyval(rg,temperatures)
    
#    plt.grid()
#    plt.xticks(fontsize=20)
#    plt.yticks(fontsize=20)    
    plt.plot(temperatures, cells, 'ko', markersize=30)  
    plt.plot(temperatures, ry, color='black', linewidth=10)
    plt.xlabel('temperature (K)',fontsize=100)
    plt.ylabel(r'a ($\mathrm{\AA}$)',fontsize=100)    


    
def plotAllBraggsInOneFig():
    definePlot(18,3)
    #c = ConstantValue
    offset = 0.
    legends = []
    labels  = []
    for temperature in c.TEM_LIST:
        color = getattr(c,'COLOR_'+temperature)
        rx,ro,rc=getRmcGsasBragg(temperature)
        #gx,go,gc=getGSASLSTfile(temperature)
        #print(go)
        print(temperature)
        #gsas, = plt.plot(gx[350:],gc[350:]+offset,'black',linewidth=5)
        gsas, = plt.plot(rx,ro+offset,'black',linewidth=5)
        rmc,  = plt.plot(rx,rc+offset,'red',linewidth=5)
        #raw,  = plt.plot(gx,go+offset,color+'o')
        offset += 1.0
        labels.append(temperature+' K')
        #legends.append(raw)
    plt.xlabel('TOF (ms)',fontsize=100)
    plt.ylabel('Bragg Intensity',fontsize=100)  
    legends.reverse()
    labels.reverse()
    #plt.legend(legends,labels, loc='upper center', bbox_to_anchor=(0.76 ,0.99), ncol=3, fontsize=40) 

def plotNPDFInOneFig():

    definePlot()
    offset = 0
    
    legends = []
    labels  = []
    
    for temperature in c.TEM_LIST:
    #for temperature in [0,1,2,3,4,5]:
        color = getattr(c,'COLOR_'+temperature)
        rmc_x, rmc_pdf, exp_pdf=getDataByTemperature(temperature)
        rmc, = plt.plot(rmc_x,rmc_pdf+offset,'k',linewidth=5)
        exp, = plt.plot(rmc_x,exp_pdf+offset, color+'o')

        offset += 1.0

    plt.xlabel(r'r ($\mathrm{\AA}$)',fontsize=100)
    plt.ylabel('D(r)',fontsize=100) 


def plotXPDFInOneFig():
    #c = ConstantValue
    scale = 10.
    definePlot(25)
    offset = 0
    for temperature in c.TEM_LIST:
        color = getattr(c,'COLOR_'+temperature)    
        xpdf_r,xpdf_dofr  = getXPDFdofr(temperature,scale) 
        rmc_r, rmc_dofr   = getRMCXPDFDataByTemperature(temperature)
        plt.plot(rmc_r, rmc_dofr+offset, 'k', linewidth=5)
        plt.plot(xpdf_r,xpdf_dofr+offset, color+'o')
        #plt.plot(dlp_x, dlp_pdf +offset, 'black')
        offset += 40.0
    plt.xlabel(r'r ($\mathrm{\AA}$)',fontsize=100)
    plt.ylabel('D(r)',fontsize=100)         
    
    

def plotNSoqInOneFig():
    definePlot(20)
    offset = 0.
    ct = ConstantValue
    for temperature in ct.TEM_LIST:
        color = getattr(ct,'COLOR_'+temperature) 
        x,c,e=getNSoqByTemperature(temperature)
        soq_c, = plt.plot(x,c+offset,'black',linewidth=5)
        soq_e, = plt.plot(x,e+offset,color+'o', linewidth=5)
        offset += 1.0
    
    plt.xlabel(r'Q ($\mathrm{\AA^{-1}}$)',fontsize=100)
    plt.ylabel('i(Q)',fontsize=100)      
            

def plotXSoqInOneFig():
    definePlot(11)
    offset = 0.
    ct = ConstantValue
    for temperature in ct.TEM_LIST:
        color = getattr(ct,'COLOR_'+temperature)
        x,c,e  = getXSoqByTemperature(temperature)
        soq_c, = plt.plot(x,c+offset,'black',linewidth=5)
        soq_e, = plt.plot(x,e+offset,color+'o',linewidth=5)
        offset += 3.0
        
    plt.xlabel(r'Q ($\mathrm{\AA^{-1}}$)',fontsize=100)
    plt.ylabel('i(Q)',fontsize=100)      

     
def plotPowerSpectraInOneFig():
    plt.figure() 
    pss={1:'Li', 2:'La', 3:'Zr', 4:'O'}
    x_label = False
    for key in pss:
        print(key)
        plt.subplot(2,2,key)
        
        if key<3:
            x_label = False
        else:
            x_label = True
        plotPowerSpectra(pss[key], x_label)

def plotPartialGofrInOneFig():
    plt.figure() 
    ppdfs={1:'Li-O', 2:'O-O', 3:'O-Zr', 4:'O-La'}
    x_label = False
    for key in ppdfs:
        print(key)
        plt.subplot(2,2,key)  
        if key<3:
            x_label = False
        else:
            x_label = True
        plotPartialGofr(ppdfs[key],x_label)

def plotAllMSD():
    import matplotlib.pyplot as plt
    import math
    definePlot(32.0,5.0)
    ct = ConstantValue
    slope=[]
    for temperature in ct.TEM_LIST:
        t,l,o,z,a=readSTATIS(temperature=temperature)
        color = getattr(ct,'COLOR_'+temperature)
        start = math.ceil(len(t)/10) 
        end   = math.ceil(len(t)/2)
        plt.plot(t[start:end],l[start:end],color=color,marker='o')
        rg=np.polyfit(t[start:end],l[start:end], 1)
        ry=np.polyval(rg,t[start:end])
        plt.plot(t[start:end],ry,color='k',linewidth=5)
        deltaY = ry[-1]   - ry[0]
        deltaX = t[end] - t[start]
        slope.append(deltaY/deltaX)
    plt.xlabel('time (ps)',fontsize=100)
    plt.ylabel('MSD ($\mathrm{\AA^2}$)',fontsize=100)
    print(slope)    
    

def readSTATIS(dirname='../md.data/mdplotfiles/', temperature = '293'):
    import math
    if int(temperature) > 700:
        temperature = 'P'+temperature
    filename=dirname+'STATIS_'+temperature+'K'
    print(filename)
    f = open(filename)
    line = f.readline()
    line = f.readline()
    line = f.readline()
    time = []
    dofl = []
    dofo = []
    dofz = []
    dofa = []
    while(line):
        # record 1
        items = line.split()
        nframe, ntime, nitem = int(items[0]), float(items[1]), int(items[2])
        time.append(ntime)
        nrecord = math.ceil(nitem/5)
        for i in range(5): # record 2-6
            line = f.readline()
        items    = f.readline().split() # record 7
        DLi, DO  = float(items[3]), float(items[4])
        items    = f.readline().split() # record 8
        DZr, DLa = float(items[0]), float(items[1])
        dofl.append(DLi)
        dofo.append(DO)
        dofz.append(DZr)
        dofa.append(DLa)
        for i in range(nrecord - 7): # rest records
            line = f.readline()
        line = f.readline() # next frame    
    return np.array(time), np.array(dofl), np.array(dofo), np.array(dofz), np.array(dofa)

def plotD():
    
    plt.axes(yscale = "log")  
    
    
    plt.ylim(1.E-13, 1.E-11)
    #definePlot(xup=3.5, xlow=0.5)
    # my=np.array([0.031075532625208274, 0.05198720372587821, 0.1328615809979784, 0.28705397833485197, 0.6056943962610198, 0.9520218588231469])
    # mx=np.array([293.,450.,600.,750.,900.,1100.])
    # mx=1000/mx
    # my=my*1E-11
    # plt.plot(mx,my,'o', markerfacecolor='none', markeredgewidth=5., color='k', markersize=40)
    
    # mrg=np.polyfit(mx, my, 1)
    # mry=np.polyval(mrg,mx)
    #plt.plot(mx,mry,color='k',linewidth=5)
    
    definePlot(xup=3.5, xlow=2.6, grid = False)
    # DLi (m2s-1)
    #ex=np.array([2.68097,2.75482,2.83286,2.91545,3.003,3.19489,3.41297])
    ey=np.array([6.02091E-12,3.6901E-12,3.056E-12,2.08526E-12,1.38574E-12,6.57443E-13,1.31293E-13])
    ex=np.array([2.68097,2.75482,2.83286,2.91545,3.003,3.19489,3.41297])
    plt.plot(ex,ey, 'o', color='k', markersize=20)
    # erg=np.polyfit(ex, ey, 1)
    # ery=np.polyval(erg,ex)
    # plt.plot(ex,ery,color='k',linewidth=5)
    
    plt.xlabel('1000/T (K$^{-1}$)',fontsize=60)
    plt.ylabel('$\mathrm{D_{Li}(m^2 s^{-1}}$)',fontsize=60)


def plotDiffuseCoeff():
    # definePlot(xup=3.7, xlow=.6)
    # my=np.array([0.031075532625208274, 0.05198720372587821, 0.1328615809979784, 0.28705397833485197, 0.6056943962610198, 0.9520218588231469])
    # mx=np.array([293.,450.,600.,750.,900.,1100.])
    # my=my/6
    # my=np.log(my*mx)
    # mx=1000/mx
    # plt.plot(mx,my, 'o', markerfacecolor='none',  markeredgewidth=5.,color='k', markersize=40)
    
    #ln [s x K] (S x cm-1 x K)
    
    definePlot(xup=3.5, xlow=2.6)
    ex=np.array([2.68097,2.75482,2.83286, 2.91545, 3.003,   3.19489, 3.41297])
    ey=np.array([1.05308,0.5635, 0.37495,-0.00726,-0.41592,-1.16155,-2.77248])
    plt.xlabel('1000/T (K$^{-1}$)',fontsize=60)
    #plt.ylabel('ln($\mathrm{D_{Li}T}$)',fontsize=100)
    plt.ylabel('ln[s X K] (S X cm$^{-1}$ X K)', fontsize = 60)
    plt.plot(ex,ey, 'o', color='k', markersize=20)
    plt.annotate('$\mathrm{E_a}$='+format(0.36,'0.2f')+' eV', fontsize=60, xy=(3.0,0.5))
    # mrg=np.polyfit(mx, my, 1)
    # mry=np.polyval(mrg,mx)
    # plt.plot(mx,mry,color='k',linewidth=5)
    # mae=-((mry[-1]-mry[0])/(mx[-1]-mx[0]))*8.314/96.484  # R = 8.314  J/mol=>eV = 96.484
    # print(mae)
    # plt.annotate('$\mathrm{E_a}$='+format(mae,'0.2f')+' eV', fontsize=100, xy=(2.5,4.0))
    
    
    
    erg=np.polyfit(ex, ey, 1)
    ery=np.polyval(erg,ex)
    plt.plot(ex,ery,color='k',linewidth=5)
    # eae=-((ery[-1]-ery[0])/(ex[-1]-ex[0]))*8.314/96.484  # R = 8.314  J/mol=>eV = 96.484
    # plt.annotate('$\mathrm{E_a}$='+format(eae,'0.2f')+' eV', fontsize=100, xy=(1.5,0.2))
    # print(eae)
    # plt.grid()
    
    
    