#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 11:26:53 2020

@author: kissami
"""
from mpi4py import MPI
import numpy as np
from numba import njit
from manapy import ddm
from sedima.fvm import *
from sedima.fonction_utils import *
from sedima.localstructure import *
from sedima.meshpartitioning import *
minus


COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()


@njit
def compute_flux_advection(flux, fleft, fright, wleft, wright, normal):

    sol = 0
    vel = np.zeros(2)

    vel[0] = 0.5*(wleft.hu + wright.hu)
    vel[1] = 0.5*(wleft.hv + wright.hv)

    sign = np.dot(vel, normal)

    if sign >= 0:
        sol = wleft.h
    else:
        sol = wright.h

    flux.h = sign * sol
    flux.hu = 0
    flux.hv = 0
    flux.Z = 0

    return flux





@njit(fastmath=True)
def compute_flux_shallow_srnh(flux, fleft, fright, w_l, w_r, normal, mesure, grav):

    wn_l = w_l
    wr_l = w_r

    epsilon = 1e-4
    
    As=1
    p=0.5
    m=1
    zeta=1/(1-p)

    if wn_l.h < 0:
        wn_l.h = np.fabs(wn_l.h)
    if wr_l.h < 0:
        wr_l.h = np.fabs(wr_l.h)


    ninv = np.zeros(2)
    ninv[0] = -1*normal[1]
    ninv[1] = normal[0]
    w_dif = np.zeros(4)

    if np.fabs(wn_l.h)  < epsilon and np.fabs(wr_l.h) < epsilon:
        flux.h = 0.
        flux.hu = 0.
        flux.hv = 0.
        flux.Z = 0.
    else:
   
        if np.fabs(wn_l.h)  < epsilon and np.fabs(wr_l.h) >  epsilon:            

            u_h = (wr_l.hu / wr_l.h )
        
            v_h = (wr_l.hv / wr_l.h )
        

            un_h = u_h*normal[0] + v_h*normal[1]
            un_h = un_h / mesure
            vn_h = u_h*ninv[0] + v_h*ninv[1]
            vn_h = vn_h / mesure
        
            hroe = (wr_l.h)/2
            uroe = un_h
            vroe = vn_h
            
               
        elif np.fabs(wn_l.h)  > epsilon and np.fabs(wr_l.h) < epsilon:

            u_h = (wn_l.hu / wn_l.h )
        
            v_h = (wn_l.hv / wn_l.h )
        
            
            un_h = u_h*normal[0] + v_h*normal[1]
            un_h = un_h / mesure
            vn_h = u_h*ninv[0] + v_h*ninv[1]
            vn_h = vn_h / mesure
        
            hroe = (wn_l.h)/2
            uroe = un_h
            vroe = vn_h
        
        elif np.fabs(wn_l.h)  > epsilon and np.fabs(wr_l.h) >  epsilon:
           
            u_h = (wn_l.hu / wn_l.h * np.sqrt(wn_l.h)
                + wr_l.hu / wr_l.h * np.sqrt(wr_l.h)) /(np.sqrt(wn_l.h) + np.sqrt(wr_l.h))
    
            v_h = (wn_l.hv / wn_l.h * np.sqrt(wn_l.h)
                    + wr_l.hv / wr_l.h * np.sqrt(wr_l.h)) /(np.sqrt(wn_l.h) + np.sqrt(wr_l.h))
        
            
            #uvh = np.array([uh, vh])
            un_h = u_h*normal[0] + v_h*normal[1]
            un_h = un_h / mesure
            vn_h = u_h*ninv[0] + v_h*ninv[1]
            vn_h = vn_h / mesure
        
            hroe = (wn_l.h+wr_l.h)/2
            uroe = un_h
            vroe = vn_h

        uleft = wn_l.hu*normal[0] + wn_l.hv*normal[1]
        uleft = uleft / mesure
        vleft = wn_l.hu*ninv[0] + wn_l.hv*ninv[1]
        vleft = vleft / mesure
    
        uright = wr_l.hu*normal[0] + wr_l.hv*normal[1]
        uright = uright / mesure
        vright = wr_l.hu*ninv[0] + wr_l.hv*ninv[1]
        vright = vright / mesure

        w_lrh = (wn_l.h  + wr_l.h)/2
        w_lrhu = (uleft + uright)/2
        w_lrhv = (vleft + vright)/2
        w_lrz = (wn_l.Z + wr_l.Z)/2
        
                            
        w_dif[0] = wr_l.h - wn_l.h
        w_dif[1] = uright - uleft
        w_dif[2] = vright - vleft
        w_dif[3] = wr_l.Z - wn_l.Z
      
        #les valeurs propre
        
        d=As*zeta*(3*un_h**2 + vn_h**2)
        sound = np.sqrt(grav * hroe)
        P=(un_h**2+3*grav*(hroe+d))/9
        R=un_h*(9*grav*(2*hroe-d)-2*un_h**2)/54
        teta=np.arccos(R/(np.sqrt(P**3)))
        if P<0:
            print(P)
        # lambda1=2*np.sqrt(P)*np.cos(teta/3)+(2/3)*un_h
        # lambda2=2*np.sqrt(P)*np.cos((teta+2*np.pi)/3)+(2/3)*un_h
        # lambda3=2*np.sqrt(P)*np.cos((teta+4*np.pi)/3)+(2/3)*un_h
        # lambda4=un_h
          
    

    #     if lambda1 == 0:
    #         sign1 = 0.
    #     else:
    #         sign1 = lambda1 / np.fabs(lambda1)

    #     if lambda2 == 0:
    #         sign2 = 0.
    #     else:
    #         sign2 = lambda2 / np.fabs(lambda2)

    #     if lambda3 == 0:
    #         sign3 = 0.
    #     else:
    #         sign3 = lambda3 / np.fabs(lambda3)
    
    #     if lambda4 == 0:
    #         sign4 = 0.
    #     else:
    #         sign4 = lambda4 / np.fabs(lambda4)    
          
    #     rmat = np.zeros((4, 4))
    #     rmati = np.zeros((4, 4))
    #     slmat = np.zeros((4, 4))
    #     rslmat = np.zeros((4, 4))
    #     smmat = np.zeros((4, 4))
    
    #     slmat[0][0] = sign1
    #     slmat[1][1] = sign2
    #     slmat[2][2] = sign3
    #     slmat[3][3] = sign4
        
    
    #     rmat[0][0] = 1.
    #     rmat[1][0] = (lambda1 - un_h)/hroe
    #     rmat[2][0] = 0
    #     rmat[3][0] = ((lambda1 - un_h)/sound)**2 - 1.
    
    #     rmat[0][1] = 1.
    #     rmat[1][1] = (lambda2 - un_h)/hroe
    #     rmat[2][1] =0
    #     rmat[3][1] = ((lambda2 - un_h)/sound)**2 - 1.
        
    #     rmat[0][2] = 1.
    #     rmat[1][2] = (lambda3 - un_h)/hroe
    #     rmat[2][2] =0
    #     rmat[3][2] = ((lambda3 - un_h)/sound)**2 - 1.
        
    #     rmat[0][3] = -1.
    #     rmat[1][3] = 0
    #     rmat[2][3] =1/(2*As*zeta*vn_h)
    #     rmat[3][3] = 1
    
    #     alpha1=(lambda2-un_h)*(lambda3-un_h)
    #     alpha2=(lambda1-un_h)*(lambda3-un_h)
    #     alpha3=(lambda2-un_h)*(lambda1-un_h)
            
    #     rmati[0][0] = sound**2 + alpha1
    #     rmati[1][0] = - sound**2 - alpha2
    #     rmati[2][0] =  sound**2 + alpha3
    #     rmati[3][0] = 0
    
    
    #     rmati[0][1] = -hroe*(lambda2 + lambda3 - 2*un_h)
    #     rmati[1][1] =  hroe*(lambda1 + lambda3 - 2*un_h)
    #     rmati[2][1] = -hroe*(lambda2 + lambda1 - 2*un_h)
    #     rmati[3][1] = 0
        
    #     rmati[0][2] =  2.*As*zeta*alpha1*vn_h
    #     rmati[1][2] = -2.*As*zeta*alpha2*vn_h
    #     rmati[2][2] =  2.*As*zeta*alpha3*vn_h
    #     rmati[3][2] =  2.*As*zeta*vn_h
        
    #     rmati[0][3] =  sound**2
    #     rmati[1][3] = -sound**2
    #     rmati[2][3] =  sound**2
    #     rmati[3][3] =  0
    

    #     rslmat = matmul(rslmat, rmat, slmat)
    #     smmat = matmul(smmat, rslmat, rmati)

    #     hnew = 0.
    #     unew = 0.
    #     vnew = 0.
    #     znew = 0.
        
    #     # for i in range(4):
    #     #     hnew += smmat[0][i] * w_dif[i]
    #     #     unew += smmat[1][i] * w_dif[i]
    #     #     vnew += smmat[2][i] * w_dif[i]
    #     #     znew += smmat[3][i] * w_dif[i]
    
    #     # u_h = hnew/2
    #     # u_hu = unew/2
    #     # u_hv = vnew/2
    #     # u_z = znew/2

    #     # w_lrh = w_lrh  - u_h
    #     # w_lrhu = w_lrhu - u_hu
    #     # w_lrhv = w_lrhv - u_hv
    #     # w_lrz = w_lrz  - u_z
    
    #     # unew = 0.
    #     # vnew = 0.
        
        
       
    #     # if w_lrh > epsilon:
            
    #     #     Qbx=As*w_lrhu*normal[0]*(np.sqrt(w_lrhu*w_lrhu*normal[0]+w_lrhv*w_lrhv*normal[1])/w_lrh)**(m-1)/w_lrh
    #     #     Qby=As*w_lrhv**normal[1]*(np.sqrt(w_lrhu*w_lrhu*normal[0]+w_lrhv*w_lrhv*normal[1])/w_lrh)**(m-1)/w_lrh
            
            
    #     #     # Qbx=As*w_lrhu*(np.sqrt(w_lrhu**2+w_lrhv**2)/w_l)**(m-1)/w_lrh
    #     #     # Qby=As*w_lrhv*(np.sqrt(w_lrhu**2+w_lrhv**2)/w_l)**(m-1)/w_lrh
    
    #     #     unew = w_lrhu * normal[0] + w_lrhv * -1*normal[1]
    #     #     unew = unew / mesure
    #     #     vnew = w_lrhu * -1*ninv[0] + w_lrhv * ninv[1]
    #     #     vnew = vnew / mesure
        
        
    #     #     w_lrhu = unew
    #     #     w_lrhv = vnew
        
    #     #     q_s = normal[0] * unew + normal[1] * vnew
            
        
    #     #     flux.h = q_s
    #     #     flux.hu = q_s * w_lrhu/w_lrh + 0.5 * grav * w_lrh * w_lrh * normal[0]
    #     #     flux.hv = q_s * w_lrhv/w_lrh + 0.5 * grav * w_lrh * w_lrh * normal[1]
    #     #     flux.Z =normal[0] * Qbx + normal[1] * Qby            
    #     # else:
    #     #     flux.h = 0
    #     #     flux.hu = 0
    #     #     flux.hv = 0
    #     #     flux.Z = 0.

    flux.h = 0
    flux.hu = 0
    flux.hv = 0
    flux.Z = 0.
    return flux

@njit(fastmath=True)
def term_source_srnh(w_c, w_ghost, w_halo, w_x, w_y, wx_halo, wy_halo, nodeidc, faceidc,
                     cellidc, centerc, volumec, normalc, cellidf, nodeidf, normalf, centerf, namef,
                     ghostcenterf, vertexn, halofid, centerh, mystruct, order, grav, source):

    if order == 2:
        lim_bar = np.zeros(len(w_c))
        lim_bar = ddm.barthlimiter(w_c, w_x, w_y, lim_bar, cellidf, faceidc, centerc, centerf)

    elif order == 3:
        lim_alb = np.zeros(1, dtype=mystruct)[0]

    #source = np.zeros(len(w_c), dtype=mystruct)
    trv = np.zeros(1, dtype=mystruct)[0]
    nbelement = len(w_c)
    hi_p = np.zeros(3)
    zi_p = np.zeros(3)

    zv = np.zeros(3)
    #eta = 0.

    mata = np.zeros(3)
    matb = np.zeros(3)


    for i in range(nbelement):
        ns = np.zeros((3, 2))
        ss = np.zeros((3, 2))

        G = centerc[i]
        c_1 = 0
        c_2 = 0

        for j in range(3):
            f = faceidc[i][j]
            ss[j] = normalc[i][j]
            

            if namef[f] == 10 and SIZE > 1:
                
                trv = w_halo[halofid[f]]
                
                if order == 1:
                    h_1p = w_c[i].h
                    z_1p = w_c[i].Z
                    
                    h_p1 = trv.h
                    z_p1 = trv.Z
                
                if order == 2:
                    w_x_halo = wx_halo[halofid[f]]
                    w_y_halo = wy_halo[halofid[f]]

                    r_l = np.array([centerf[f][0] - centerc[i][0], centerf[f][1] - centerc[i][1]])
                    r_r = np.array([centerf[f][0] - centerh[halofid[f]][0], 
                                    centerf[f][1] - centerh[halofid[f]][1]])
                    
                    h_1p = w_c[i].h + lim_bar[i]*(w_x[i].h*r_l[0] + w_y[i].h*r_l[1])                    
                    z_1p = w_c[i].Z + lim_bar[i]*(w_x[i].Z*r_l[0] + w_y[i].Z*r_l[1])

                    h_p1 = trv.h    + lim_bar[i]*(w_x_halo.h*r_r[0] + w_y_halo.h*r_r[1])
                    z_p1 = trv.Z    + lim_bar[i]*(w_x_halo.Z*r_r[0] + w_y_halo.Z*r_r[1])
                
                if order == 3 :
                    
                    w_x_halo = wx_halo[halofid[f]]
                    w_y_halo = wy_halo[halofid[f]]
                    
                    lim_alb = ddm.albada(w_c[i], trv, w_x[i], w_y[i], centerc[i], centerh[halofid[f]], 
                                     lim_alb)
                    
                    h_1p = w_c[i].h  + 0.5*lim_alb["h"]
                    z_1p = w_c[i].Z  + 0.5*lim_alb["Z"]
                    
                    lim_alb = ddm.albada(trv, w_c[i], w_x_halo, w_y_halo, centerh[halofid[f]], centerc[i], 
                                     lim_alb)                    
                    h_p1 = trv.h    + 0.5*lim_alb["h"]
                    z_p1 = trv.Z    + 0.5*lim_alb["Z"]
                    
                                
            elif namef[f] == 0:
                vois = cellidc[i][j]
                trv = w_c[vois]
                
                if order == 1:
                    h_1p = w_c[i].h
                    z_1p = w_c[i].Z
                    
                    h_p1 = trv.h
                    z_p1 = trv.Z                

                if order == 2:
                    r_l = np.array([centerf[f][0] - centerc[i][0], centerf[f][1] - centerc[i][1]])
                    r_r = np.array([centerf[f][0] - centerc[vois][0], centerf[f][1] - centerc[vois][1]])
                
                    h_1p = w_c[i].h + lim_bar[i]*(w_x[i].h*r_l[0] + w_y[i].h*r_l[1])                    
                    z_1p = w_c[i].Z + lim_bar[i]*(w_x[i].Z*r_l[0] + w_y[i].Z*r_l[1])

                    h_p1 = trv.h    + lim_bar[vois]*(w_x[vois].h*r_r[0] + w_y[vois].h*r_r[1])
                    z_p1 = trv.Z    + lim_bar[vois]*(w_x[vois].Z*r_r[0] + w_y[vois].Z*r_r[1])
                
                elif order == 3:
                    lim_alb = albada(w_c[i], trv, w_x[i], w_y[i], centerc[i], centerc[vois], lim_alb)
                        
                    h_1p = w_c[i].h  + 0.5*lim_alb["h"]
                    z_1p = w_c[i].Z  + 0.5*lim_alb["Z"]
                    
                    lim_alb = np.zeros(1, dtype=mystruct)[0]
                    lim_alb = albada(trv, w_c[i], w_x[vois], w_y[vois], centerc[vois], centerc[i], 
                                     lim_alb)
                    
                    h_p1 = trv.h    + 0.5*lim_alb["h"]
                    z_p1 = trv.Z    + 0.5*lim_alb["Z"]

            else:
                trv = w_ghost[f]

                if order == 1:
                    h_1p = w_c[i].h
                    z_1p = w_c[i].Z

                elif order == 2:
                    r_l = np.array([centerf[f][0] - centerc[i][0], centerf[f][1] - centerc[i][1]])
                
                    h_1p = w_c[i].h + lim_bar[i]*(w_x[i].h*r_l[0] + w_y[i].h*r_l[1])                    
                    z_1p = w_c[i].Z + lim_bar[i]*(w_x[i].Z*r_l[0] + w_y[i].Z*r_l[1])

                elif order == 3:                                   
                    lim_alb = albada(w_c[i], trv, w_x[i], w_y[i], centerc[i], ghostcenterf[f],
                                         lim_alb)
                            
                    h_1p = w_c[i].h  + 0.5*lim_alb["h"]
                    z_1p = w_c[i].Z  + 0.5*lim_alb["Z"]

                
                h_p1 = trv.h
                z_p1 = trv.Z

            zv[j] = z_p1
            mata[j] = h_p1*ss[j][0]
            matb[j] = h_p1*ss[j][1]
            c_1 = c_1 + pow(0.5*(h_1p + h_p1), 2)*ss[j][0]
            c_2 = c_2 + pow(0.5*(h_1p + h_p1), 2)*ss[j][1]
            
            hi_p[j] = h_1p
            zi_p[j] = z_1p
            

        c_3 = 3.0 * h_1p
            
        delta = (mata[1]*matb[2]-mata[2]*matb[1]) - (mata[0]*matb[2]-matb[0]*mata[2]) + (mata[0]*matb[1]-matb[0]*mata[1])

        deltax = c_3*(mata[1]*matb[2]-mata[2]*matb[1]) - (c_1*matb[2]-c_2*mata[2]) + (c_1*matb[1]-c_2*mata[1])

        deltay = (c_1*matb[2]-c_2*mata[2]) - c_3*(mata[0]*matb[2]-matb[0]*mata[2]) + (mata[0]*c_2-matb[0]*c_1)

        deltaz = (mata[1]*c_2-matb[1]*c_1) - (mata[0]*c_2-matb[0]*c_1) + c_3*(mata[0]*matb[1]-matb[0]*mata[1])
        
        if np.fabs(delta) > 1e-6:

            h_1 = deltax/delta
            h_2 = deltay/delta
            h_3 = deltaz/delta
                
            z_1 = zi_p[0] + hi_p[0] - h_1
            z_2 = zi_p[1] + hi_p[1] - h_2
            z_3 = zi_p[2] + hi_p[2] - h_3
    
            b = np.array([vertexn[nodeidc[i][1]][0], vertexn[nodeidc[i][1]][1]])
    
            ns[0] = np.array([(G[1]-b[1]), -(G[0]-b[0])])
            ns[1] = ns[0] - ss[1]  #  N23
            ns[2] = ns[0] + ss[0]  #  N31
    
            s_1 = 0.5*h_1*(zv[0]*ss[0] + z_2*ns[0] + z_3*(-1)*ns[2])
            s_2 = 0.5*h_2*(zv[1]*ss[1] + z_1*(-1)*ns[0] + z_3*ns[1])
            s_3 = 0.5*h_3*(zv[2]*ss[2] + z_1*ns[2] + z_2*(-1)*ns[1])
            
    #            ufric = w_c[i].hu/w_c[i].h
    #            vfric = w_c[i].hv/w_c[i].h
    #            hfric = w_c[i].h
    #            
    #            s_fx = -eta**2 * ufric * np.sqrt(ufric**2 + vfric**2)/(pow(hfric, 4/3))
    #            s_fy = -eta**2 * vfric * np.sqrt(ufric**2 + vfric**2)/(pow(hfric, 4/3))
    
            source[i].h = 0
            source[i].hu = -grav*( (s_1[0] + s_2[0] + s_3[0]))# - hfric*s_fx)
            source[i].hv = 0#-grav*( (s_1[1] + s_2[1] + s_3[1]))# - hfric*s_fy)
            source[i].Z = 0.
    else:
        source[i].h = 0
        source[i].hu = 0
        source[i].hv = 0
        source[i].Z = 0.


    return source



#@njit(fastmath=True)
@njit(fastmath=True)
def explicitscheme_convective(w_c, w_x, w_y, w_ghost, w_halo, wx_halo, wy_halo, cellidf, faceidc,
                              nodeidc, centerc, cellidc, centerh, mesuref, centerf, normal, halofid,
                              name, ghostcenterf, cellidn, mystruct, order, grav):

    rezidus = np.zeros(len(w_c), dtype=mystruct)
    w_l = np.zeros(1, dtype=mystruct)[0]
    w_r = np.zeros(1, dtype=mystruct)[0]
    w_ln = np.zeros(1, dtype=mystruct)[0]
    w_rn = np.zeros(1, dtype=mystruct)[0]
    nbface = len(cellidf)
    nbelement = len(w_c)

    flx = np.zeros(1, dtype=mystruct)[0]
    fleft = np.zeros(1, dtype=mystruct)[0]
    fright = np.zeros(1, dtype=mystruct)[0]


    if order == 1:

        for i in range(nbface):
            
            w_l = w_c[cellidf[i][0]]
            norm = normal[i]
            mesu = mesuref[i]

            if name[i] == 0:
                w_r = w_c[cellidf[i][1]]
                
                flx = compute_flux_shallow_srnh(flx, fleft, fright, w_l, w_r, norm, mesu, grav)
                
                rezidus[cellidf[i][0]] = minus(rezidus[cellidf[i][0]], flx)
                rezidus[cellidf[i][1]] = add(rezidus[cellidf[i][1]], flx)

            elif name[i] == 10:
                w_r = w_halo[halofid[i]]

                flx = compute_flux_shallow_srnh(flx, fleft, fright, w_l, w_r, norm, mesu, grav)
                
                

            else:
                w_r = w_ghost[i]
                
                flx = compute_flux_shallow_srnh(flx, fleft, fright, w_l, w_r, norm, mesu, grav)
                
                
                rezidus[cellidf[i][0]] = minus(rezidus[cellidf[i][0]], flx)

    elif order == 2:

        psi = np.zeros(nbelement)
        psi = barthlimiter(w_c, w_x, w_y, psi, cellidf, faceidc, centerc, centerf)

        for i in range(nbface):
            w_l = w_c[cellidf[i][0]]
            norm = normal[i]
            mesu = mesuref[i]

            if name[i] == 0:
                w_r = w_c[cellidf[i][1]]

                center_left = centerc[cellidf[i][0]]
                center_right = centerc[cellidf[i][1]]

                w_x_left = w_x[cellidf[i][0]]
                w_y_left = w_y[cellidf[i][0]]
                psi_left = psi[cellidf[i][0]]

                w_x_right = w_x[cellidf[i][1]]
                w_y_right = w_y[cellidf[i][1]]
                psi_right = psi[cellidf[i][1]]

                r_l = np.array([centerf[i][0] - center_left[0], centerf[i][1] - center_left[1]])
                w_ln.h = w_l.h  + psi_left * (w_x_left.h  * r_l[0] + w_y_left.h  * r_l[1])
                w_ln.hu = w_l.hu + psi_left * (w_x_left.hu * r_l[0] + w_y_left.hu * r_l[1])
                w_ln.hv = w_l.hv + psi_left * (w_x_left.hv * r_l[0] + w_y_left.hv * r_l[1])
                w_ln.Z = w_l.Z  + psi_left * (w_x_left.Z  * r_l[0] + w_y_left.Z  * r_l[1])

                r_r = np.array([centerf[i][0] - center_right[0], centerf[i][1] - center_right[1]])
                w_rn.h = w_r.h  + psi_right * (w_x_right.h  * r_r[0] + w_y_right.h  * r_r[1])
                w_rn.hu = w_r.hu + psi_right * (w_x_right.hu * r_r[0] + w_y_right.hu * r_r[1])
                w_rn.hv = w_r.hv + psi_right * (w_x_right.hv * r_r[0] + w_y_right.hv * r_r[1])
                w_rn.Z = w_r.Z  + psi_right * (w_x_right.Z  * r_r[0] + w_y_right.Z  * r_r[1])

                flx = compute_flux_shallow_srnh(flx, fleft, fright, w_ln, w_rn, norm, mesu, grav)
                
                

                rezidus[cellidf[i][0]] = minus(rezidus[cellidf[i][0]], flx)
                rezidus[cellidf[i][1]] = add(rezidus[cellidf[i][1]], flx)

            elif name[i] == 10 and SIZE > 1:

                w_l = w_c[cellidf[i][0]]
                w_r = w_halo[halofid[i]]

                center_left = centerc[cellidf[i][0]]
                center_right = centerh[halofid[i]]

                w_x_left = w_x[cellidf[i][0]]
                w_y_left = w_y[cellidf[i][0]]
                psi_left = psi[cellidf[i][0]]

                w_x_right = wx_halo[halofid[i]]
                w_y_right = wy_halo[halofid[i]]
                psi_right = psi[cellidf[i][0]]

                r_l = np.array([centerf[i][0] - center_left[0], centerf[i][1] - center_left[1]])
                w_ln.h = w_l.h  + psi_left * (w_x_left.h  * r_l[0] + w_y_left.h  * r_l[1])
                w_ln.hu = w_l.hu + psi_left * (w_x_left.hu * r_l[0] + w_y_left.hu * r_l[1])
                w_ln.hv = w_l.hv + psi_left * (w_x_left.hv * r_l[0] + w_y_left.hv * r_l[1])
                w_ln.Z = w_l.Z  + psi_left * (w_x_left.Z  * r_l[0] + w_y_left.Z  * r_l[1])

                r_r = np.array([centerf[i][0] - centerh[halofid[i]][0],
                                centerf[i][1] - centerh[halofid[i]][1]])
                w_rn.h = w_r.h  + psi_right * (w_x_right.h  * r_r[0] + w_y_right.h  * r_r[1])
                w_rn.hu = w_r.hu + psi_right * (w_x_right.hu * r_r[0] + w_y_right.hu * r_r[1])
                w_rn.hv = w_r.hv + psi_right * (w_x_right.hv * r_r[0] + w_y_right.hv * r_r[1])
                w_rn.Z = w_r.Z  + psi_right * (w_x_right.Z  * r_r[0] + w_y_right.Z  * r_r[1])

                flx = compute_flux_shallow_srnh(flx, fleft, fright, w_ln, w_rn, norm, mesu, grav)
                rezidus[cellidf[i][0]] = ddm.minus(rezidus[cellidf[i][0]], flx)

            else:
                
                w_l = w_c[cellidf[i][0]]
                w_r = w_ghost[i]

                center_left = centerc[cellidf[i][0]]
                center_right = ghostcenterf[i]

                w_x_left = w_x[cellidf[i][0]]
                w_y_left = w_y[cellidf[i][0]]
                psi_left = psi[cellidf[i][0]]

                r_l = np.array([centerf[i][0] - center_left[0], centerf[i][1] - center_left[1]])
                w_ln.h = w_l.h  + psi_left * (w_x_left.h  * r_l[0] + w_y_left.h  * r_l[1])
                w_ln.hu = w_l.hu + psi_left * (w_x_left.hu * r_l[0] + w_y_left.hu * r_l[1])
                w_ln.hv = w_l.hv + psi_left * (w_x_left.hv * r_l[0] + w_y_left.hv * r_l[1])
                w_ln.Z = w_l.Z  + psi_left * (w_x_left.Z  * r_l[0] + w_y_left.Z  * r_l[1])

                flx = compute_flux_shallow_srnh(flx, fleft, fright, w_ln, w_r, norm, mesu, grav)
                rezidus[cellidf[i][0]] = minus(rezidus[cellidf[i][0]], flx)
                
    elif order == 3:

        lim = np.zeros(1, dtype=mystruct)[0]

        for i in range(nbface):
            w_l = w_c[cellidf[i][0]]
            norm = normal[i]
            mesu = mesuref[i]

            if name[i] == 0:
                w_r = w_c[cellidf[i][1]]

                center_left = centerc[cellidf[i][0]]
                center_right = centerc[cellidf[i][1]]
                
                w_x_left = w_x[cellidf[i][0]]
                w_y_left = w_y[cellidf[i][0]]


                lim = albada(w_l, w_r, w_x_left, w_y_left, center_left, center_right, lim)
                w_ln.h = w_l.h  + 0.5 * lim.h
                w_ln.hu = w_l.hu + 0.5 * lim.hu
                w_ln.hv = w_l.hv + 0.5 * lim.hv
                w_ln.Z = w_l.Z  + 0.5 * lim.Z

                w_x_right = w_x[cellidf[i][1]]
                w_y_right = w_y[cellidf[i][1]]

                lim = albada(w_r, w_l, w_x_right, w_y_right, center_right, center_left, lim)
                w_rn.h = w_r.h  + 0.5 * lim.h
                w_rn.hu = w_r.hu + 0.5 * lim.hu
                w_rn.hv = w_r.hv + 0.5 * lim.hv
                w_rn.Z = w_r.Z  + 0.5 * lim.Z

                flx = compute_flux_shallow_srnh(flx, fleft, fright, w_l, w_r, norm, mesu, grav)
                rezidus[cellidf[i][0]] = minus(rezidus[cellidf[i][0]], flx)
                rezidus[cellidf[i][1]] = add(rezidus[cellidf[i][1]], flx)

            elif name[i] == 10 and SIZE > 1:

                w_l = w_c[cellidf[i][0]]
                w_r = w_halo[halofid[i]]
                
                center_left = centerc[cellidf[i][0]]
                center_right = centerh[halofid[i]]
                
                w_x_left = w_x[cellidf[i][0]]
                w_y_left = w_y[cellidf[i][0]]

                lim = albada(w_l, w_r, w_x_left, w_y_left, center_left, center_right, lim)
                w_ln.h = w_l.h  + 0.5 * lim.h
                w_ln.hu = w_l.hu + 0.5 * lim.hu
                w_ln.hv = w_l.hv + 0.5 * lim.hv
                w_ln.Z = w_l.Z  + 0.5 * lim.Z

                w_x_right = wx_halo[halofid[i]]
                w_y_right = wy_halo[halofid[i]]

                lim = albada(w_r, w_l, w_x_right, w_y_right, center_right, center_left, lim)
                w_rn.h = w_r.h  + 0.5 * lim.h
                w_rn.hu = w_r.hu + 0.5 * lim.hu
                w_rn.hv = w_r.hv + 0.5 * lim.hv
                w_rn.Z = w_r.Z  + 0.5 * lim.Z

                flx = compute_flux_shallow_srnh(flx, fleft, fright, w_l, w_r, norm, mesu, grav)
                rezidus[cellidf[i][0]] = minus(rezidus[cellidf[i][0]], flx)

            else:
                w_l = w_c[cellidf[i][0]]
                w_r = w_ghost[i]

                center_left = centerc[cellidf[i][0]]
                center_right = ghostcenterf[i]

                w_x_left = w_x[cellidf[i][0]]
                w_y_left = w_y[cellidf[i][0]]
                
                lim = albada(w_l, w_r, w_x_left, w_y_left, center_left, center_right, lim)
                w_ln.h = w_l.h  + 0.5 * lim.h
                w_ln.hu = w_l.hu + 0.5 * lim.hu
                w_ln.hv = w_l.hv + 0.5 * lim.hv
                w_ln.Z = w_l.Z  + 0.5 * lim.Z
                
                flx = compute_flux_shallow_srnh(flx, fleft, fright, w_l, w_r, norm, mesu, grav)
                rezidus[cellidf[i][0]] = minus(rezidus[cellidf[i][0]], flx)
                

    return rezidus

@njit(fastmath=True)
def term_coriolis(w_c, f_c, mystruct):
    coriolis =  np.zeros(len(w_c), dtype=mystruct)
    nbelement = len(w_c)
    
    for i in range(nbelement):

        coriolis["h"][i] = 0.
        coriolis["hu"][i] = f_c*w_c["hv"][i]
        coriolis["hv"][i] = -f_c*w_c["hu"][i]
        coriolis["Z"][i] = 0.
    
    return coriolis
