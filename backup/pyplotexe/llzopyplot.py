# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 12:54:41 2020

@author: blues
"""
ppdf={1:'Li-Li', 2:'Li-O', 3:'Li-Zr', 4:'Li-La', 5:'O-O', 6:'O-Zr', 7:'O-La', 8:'Zr-Zr', 9:'Zr-La', 10:'La-La'}
ppdfwithLi={1:'Li-Li', 2:'Li-O', 3:'Li-Zr', 4:'Li-La'}
ppdfwithoutLi={5:'O-O', 6:'O-Zr', 7:'O-La', 8:'Zr-Zr', 9:'Zr-La', 10:'La-La'}

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
    
    RMC_293   = '../rmc.data/rmcplotfiles/LLZO_RMC6_293K_3'
    RMC_450   = '../rmc.data/rmcplotfiles/LLZO_RMC6_450K_3'
    RMC_600   = '../rmc.data/rmcplotfiles/LLZO_RMC6_600K_1'
    RMC_750   = '../rmc.data/rmcplotfiles/LLZO_RMC6_P750K_3'
    RMC_900   = '../rmc.data/rmcplotfiles/LLZO_RMC6_P900K_2'
    RMC_1100  = '../rmc.data/rmcplotfiles/LLZO_RMC6_P1100K_4'
    
    
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

def getDataByTemperature(temperature='300'):

    #dlpoly_path = getattr(c,'DLPOLY_'+temperature)
    rmc_path    = getattr(c,'RMC_'+temperature)
    #rmc_path    = 'RMC6/LLZO_RMC6_293K-'+str(temperature)+'/LLZO-293K'
    #dlpoly_path = 'DLPOLY/LLZO-300k/'
    #print(dlpoly_path)
    #print(rmc_path+c.rmc_npdf_suffix)

    #dl_x,dl_npdf,dl_npdf_Li=plotnpdf(dlpoly_path)
    rmc_x,rmc_pdf, exp_pdf=rmcloaddata(rmc_path+c.rmc_npdf_suffix)
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

def definePlot(xup=25, xlow=0):
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


def getRmcGsasBragg(temperature='300'):
    rmc_path=getattr(c,'RMC_'+temperature)
    filename=rmc_path+'.braggout'
    print('read rmc braggout file: '+filename)
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

def getRMCXPDFDataByTemperature(temperature='300'):

    #dlpoly_path = getattr(c,'DLPOLY_'+temperature)
    rmc_path    = getattr(c,'RMC_'+temperature)
    #print(dlpoly_path)
    print(rmc_path+c.rmc_npdf_suffix)

    #dl_x,dl_npdf,dl_npdf_Li=plotnpdf(dlpoly_path)
    rmc_x, rmc_pdf, exp_pdf = rmcloaddata(rmc_path+c.rmc_xpdf_suffix)
    #dlp_x, dlp_pdf, dlp_Li  = plotxpdf(dirname=dlpoly_path)
   # return dl_x, dl_npdf, rmc_x, rmc_pdf, exp_pdf
    return rmc_x, rmc_pdf*rmc_x#, dlp_x, dlp_pdf*1E-4

def getSoqByTemperature(temperature='300'):
    c = ConstantValue
    soq_path=getattr(c,'RMC_'+temperature)
    f=open(soq_path+'_SQ1.csv')
    lines=f.readlines()
    x=[]
    c=[]
    e=[]
    for line in lines[1:]:
        x.append(float(line.split(',')[0]))
        c.append(float(line.split(',')[1]))
        e.append(float(line.split(',')[2]))
    
    return np.array(x),np.array(c),np.array(e)

def getXSoqByTemperature(temperature='300'):
    c = ConstantValue
    soq_path=getattr(c,'RMC_'+temperature)
    f=open(soq_path+'_XFQ1.csv')
    lines=f.readlines()
    x=[]
    c=[]
    e=[]
    for line in lines[1:]:
        x.append(float(line.split(',')[0]))
        c.append(float(line.split(',')[1]))
        e.append(float(line.split(',')[2]))
    
    return np.array(x),np.array(c),np.array(e)

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
        #dlp, = plt.plot(dl_x,dl_npdf*0.00005+offset,'black')
#        elb = 'Exp '+temperature+' K'
#        rlb = 'RMC '+temperature+' K'
#        dlb = 'MD  '+temperature+' K'
#        legends.append(exp)
#        legends.append(rmc)
#        legends.append(dlp)
#        labels.append(elb)
#        labels.append(rlb)
#        labels.append(dlb)
        offset += 1.0
        
#    legends.reverse()    
#    labels.reverse()
#    plt.legend(legends,labels)
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
        x,c,e=getSoqByTemperature(temperature)
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
        x,c,e=getXSoqByTemperature(temperature)
        soq_c, = plt.plot(x,c+offset,'black',linewidth=5)
        soq_e, = plt.plot(x,e+offset,color+'o',linewidth=5)
        offset += 3.0
        
    plt.xlabel(r'Q ($\mathrm{\AA^{-1}}$)',fontsize=100)
    plt.ylabel('i(Q)',fontsize=100)      

def getDlPolyGR(dirname=''):
    x,LiLay =getGR(dirname+'/_gr_Li_La.csv')
    x,LaLay =getGR(dirname+'/_gr_La_La.csv')
    x,ZrLay =getGR(dirname+'/_gr_Zr_La.csv')
    x,OLay  =getGR(dirname+'/_gr_O_La.csv')
    x,ZrZry =getGR(dirname+'/_gr_Zr_Zr.csv')
    x,LiZry =getGR(dirname+'/_gr_Li_Zr.csv')
    x,OZry  =getGR(dirname+'/_gr_O_Zr.csv')
    x,LiLiy =getGR(dirname+'/_gr_Li_Li.csv')
    x,LiOy  =getGR(dirname+'/_gr_Li_O.csv')
    x,OOy   =getGR(dirname+'/_gr_O_O.csv')
    
    #return x, LiLay ,LaLay ,ZrLay ,OLay  ,ZrZry ,LiZry ,OZry  ,LiLiy ,LiOy  ,OOy   
    return x, LiLiy, LiOy, LiZry, LiLay, OOy, OZry, OLay, ZrZry, ZrLay, LaLay
    
def getRMCGR(filename):
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
    return np.array(x), np.array(LiLi), np.array(LiO),  np.array(LiZr), np.array(LiLa), np.array(OO), np.array(OZr),  np.array(OLa), np.array(ZrZr), np.array(ZrLa), np.array(LaLa)
 
def plotPartialGofr(item=0):

    #plt.xlim((0,10))
    definePlot(10)
    #print('plot partial pdf: '+ppdf[item])
    offset = 0
    #num = 0

    for temperature in c.TEM_LIST:
    #for item in range(10):
        dlpoly_path = getattr(c,'DLPOLY_'+temperature)
        rmc_path    = getattr(c,'RMC_'+temperature)
        #rmc_path    = 'RMC6/LLZO_RMC6_293K-'+str(num)+'/LLZO-293K'
        #dlpoly_path = 'DLPOLY/LLZO-300k/'
        #num        += 1

        color = getattr(c,'COLOR_'+temperature)
        rmc_values = getRMCGR(rmc_path+c.rmc_partialpdf_suffix)
        #dl_values  = getDlPolyGR(dlpoly_path)
        rmclegend, = plt.plot(rmc_values[0], rmc_values[item]+offset, color+'o')
        #dlplegend, = plt.plot(dl_values[0],   dl_values[item]+offset, 'black')

        
        offset += 2.0
    #plt.title(ppdf[item], fontsize=40)
    plt.xlabel(r'r ($\mathrm{\AA}$)',fontsize=40)
    plt.ylabel('$\mathrm{g_{'+ppdf[item]+'}(r)}$',fontsize=40)
    
def plotPartialGofrInOneFigWithLi():
    plt.figure() 
    for key in ppdfwithLi.keys():
        plt.subplot(2,2,key)  
        plotPartialGofr(key)

def plotPartialGofrInOneFigWithoutLi():
    plt.figure() 
    for key in ppdfwithoutLi.keys():
        plt.subplot(3,2,key-4)  
        plotPartialGofr(key)
  
def correlationfunction(dirname='', element='Li'):
    f = open(dirname+'_'+element)
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
    from scipy import fftpack as fft
    fft_y = fft.dct(y)
    xbin  = 1.#getOmega(x)
    fft_x =np.array([i*xbin for i in range(len(fft_y))])
    diffuse_const = fft_y[0]
    return fft_x, fft_y, diffuse_const
    
def getOmega(x):
    from scipy import fftpack as fft
    cval = np.cos(x)
    fftc = np.abs(fft.dct(cval))
    print(fftc)
    idx  = np.argmax(fftc)
    print(idx)
    omega=1./idx
    return omega

def plotPowerSpectra(element='Li'):
    c  = ConstantValue
    fig   = plt.figure()
    left, bottom, width, height = 0.1,0.1, 0.8, 0.8
    ax1   = fig.add_axes([left,bottom,width,height])
    ax1.grid() 
    ax=fig.gca()
    ax.spines['bottom'].set_linewidth(4)
    ax.spines['left'].set_linewidth(4)
    ax.spines['right'].set_linewidth(4)
    ax.spines['top'].set_linewidth(4)
    
    d = []
    t = []
    for temperature in c.TEM_LIST:
        dlp_path = getattr(c,'DLPOLY_'+temperature)
        color    = getattr(c,'COLOR_'+temperature)
        x,y      = correlationfunction(dlp_path, element)
        x=x[1:-10]
        y=y[1:-10]
        #ax1.plot(x[:-10],y[:-10],color)
        print(x)
        fft_x,fft_y,diffusion = powerSpectra(x,y)
        #d.append(diffusion)
        #t.append(float(temperature))
        ax1.plot(fft_x[:200],fft_y[:200], color)  
        #ax1.set_xlabel('$\mathrm{\omega}$',fontsize=40)
        #ax1.set_ylabel(r'Z($\mathrm{\omega}$) of '+element,fontsize=40)
        #ax1.xaxis.set_tick_params(labelsize=40)
        #ax1.yaxis.set_tick_params(labelsize=40)

    #t=np.array(t)
    #t=1/t
    #d=np.log(d)    
    
    #rg=np.polyfit(t, d, 1)
    #ry=np.polyval(rg,t)
    #ae=-((ry[-1]-ry[0])/(t[-1]-t[0]))*8.314/1000/96.484  # R = 8.314  J/mol=>eV = 96.484
    #print(ae)


    #left, bottom, width, height = 0.55,0.55, 0.3, 0.3
    #ax2  = fig.add_axes([left,bottom,width,height])
    #ax=fig.gca()
    #ax.spines['bottom'].set_linewidth(3)
    #ax.spines['left'].set_linewidth(3)
    #ax.spines['right'].set_linewidth(3)
    #ax.spines['top'].set_linewidth(3)
    #ax2.grid()
    #ax2.plot(t,d,'k')
    #ax2.set_xlabel('temperature (K)',fontsize=30)
    #ax2.set_ylabel('Diffusion constant of '+element,fontsize=30)
    #ax2.xaxis.set_tick_params(labelsize=30)
    #ax2.yaxis.set_tick_params(labelsize=30)                  