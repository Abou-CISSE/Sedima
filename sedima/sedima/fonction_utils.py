#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 19:04:50 2020

@author: kissami
"""
import meshio
from mpi4py import MPI
import numpy as np
from numba import njit, jit


COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

@njit(fastmath=True)
def add(sol_1, sol_2):
    sol_1.h += sol_2.h
    sol_1.hu += sol_2.hu
    sol_1.hv += sol_2.hv
    sol_1.Z += sol_2.Z

    return sol_1

@njit(fastmath=True)
def minus(sol_1, sol_2):
    sol_1.h -= sol_2.h
    sol_1.hu -= sol_2.hu
    sol_1.hv -= sol_2.hv
    sol_1.Z -= sol_2.Z

    return sol_1

@jit(fastmath=True)#nopython = False)
def matmul(rmatrix, matrix1, matrix2):
    
    lenmatrix1 = len(matrix1)
    lenmatrix2 = len(matrix2)
    
    #print(lenmatrix1, lenmatrix2,len(matrix2[0]) )
    
    for i in range(lenmatrix1):
        for j in range(len(matrix2[0])):
            for k in range(lenmatrix2):
                rmatrix[i][j] += matrix1[i][k] * matrix2[k][j]
    return rmatrix

@njit(fastmath=True)
def variables(centerc, cellidn, haloidn, vertexn, namen, centerg, centerh):
    
    nbnode = len(vertexn)

    I_xx = np.zeros(nbnode)
    I_yy = np.zeros(nbnode)
    I_xy = np.zeros(nbnode)
    R_x = np.zeros(nbnode)
    R_y = np.zeros(nbnode)
    lambda_x = np.zeros(nbnode)
    lambda_y = np.zeros(nbnode)
    number = np.zeros(nbnode)


    for i in range(nbnode):
        for j in range(len(cellidn[i])):
            if cellidn[i][j] != -1:
                center = centerc[cellidn[i][j]]
                Rx = center[0] - vertexn[i][0]
                Ry = center[1] - vertexn[i][1]
                I_xx[i] += (Rx * Rx)
                I_yy[i] += (Ry * Ry)
                I_xy[i] += (Rx * Ry)
                R_x[i] += Rx
                R_y[i] += Ry
                number[i] = number[i] + 1

        if namen[i] != 0 and namen[i] != 10:
            for j in range(2):
                center = centerg[i][j]
                Rx = center[0] - vertexn[i][0]
                Ry = center[1] - vertexn[i][1]
                I_xx[i] += (Rx * Rx)
                I_yy[i] += (Ry * Ry)
                I_xy[i] += (Rx * Ry)
                R_x[i] += Rx
                R_y[i] += Ry
                number[i] = number[i] + 1
        
        if SIZE > 1:
            for j in range(len(haloidn[i])):
                cell = haloidn[i][j]
                if cell != -1:
                    center = centerh[cell]
                    Rx = center[0] - vertexn[i][0]
                    Ry = center[1] - vertexn[i][1]
                    I_xx[i] += (Rx * Rx)
                    I_yy[i] += (Ry * Ry)
                    I_xy[i] += (Rx * Ry)
                    R_x[i] += Rx
                    R_y[i] += Ry
                    number[i] = number[i] + 1

        D = I_xx[i]*I_yy[i] - I_xy[i]*I_xy[i]
        lambda_x[i] = (I_xy[i]*R_y[i] - I_yy[i]*R_x[i]) / D
        lambda_y[i] = (I_xy[i]*R_x[i] - I_xx[i]*R_y[i]) / D
        
    return R_x, R_y, lambda_x,lambda_y, number

@njit(fastmath=True)
def centertovertex(w_c, w_ghost, w_halo, centerc, centerh, cellidn, haloidn, vertexn, namen, centerg,
                   w_node, uexact, unode_exacte, R_x, R_y, lambda_x,lambda_y, number):

    nbnode = len(vertexn)

    for i in range(nbnode):
        for j in range(len(cellidn[i])):
            if cellidn[i][j] != -1:     
                #if namen[i] == 0 or namen[i] == 10:
                center = centerc[cellidn[i][j]]
                xdiff = center[0] - vertexn[i][0]
                ydiff = center[1] - vertexn[i][1]
                alpha = (1. + lambda_x[i]*xdiff + lambda_y[i]*ydiff)/(number[i] + lambda_x[i]*R_x[i] + lambda_y[i]*R_y[i])
                w_node[i].h +=  alpha * w_c[cellidn[i][j]].h
                w_node[i].hu += alpha * w_c[cellidn[i][j]].hu
                w_node[i].hv += alpha * w_c[cellidn[i][j]].hv
                w_node[i].Z += alpha * w_c[cellidn[i][j]].Z
                unode_exacte[i] += alpha * uexact[cellidn[i][j]]
       
        if namen[i] != 0 and namen[i] != 10:
            for j in range(2):
                center = centerg[i][j]
                ghost = np.int(centerg[i][j][2])
                xdiff = center[0] - vertexn[i][0]
                ydiff = center[1] - vertexn[i][1]
                alpha = (1. + lambda_x[i]*xdiff + lambda_y[i]*ydiff)/(number[i] + lambda_x[i]*R_x[i] + lambda_y[i]*R_y[i])
                w_node[i].h +=  alpha * w_ghost[ghost].h
                w_node[i].hu += alpha * w_ghost[ghost].hu
                w_node[i].hv += alpha * w_ghost[ghost].hv
                w_node[i].Z += alpha * w_ghost[ghost].Z
                unode_exacte[i] += alpha * uexact[cellidn[i][j]]

        if SIZE > 1:
            for j in range(len(haloidn[i])):
                cell = haloidn[i][j]
                if namen[i] == 10:
                    if cell != -1:
                        center = centerh[cell]
                        xdiff = center[0] - vertexn[i][0]
                        ydiff = center[1] - vertexn[i][1]
                        alpha = (1. + lambda_x[i]*xdiff + lambda_y[i]*ydiff)/(number[i] + lambda_x[i]*R_x[i] + lambda_y[i]*R_y[i])
                        w_node[i].h +=  alpha * w_halo[cell].h
                        w_node[i].hu += alpha * w_halo[cell].hu
                        w_node[i].hv += alpha * w_halo[cell].hv
                        w_node[i].Z += alpha * w_halo[cell].Z


    return w_node, unode_exacte

@njit(fastmath=True)
def derivxy(w_c, w_ghost, w_halo, centerc, nodeidc, cellnidc, cellidn, haloidn, namen, centerg, centerh,
            w_x, w_y):

    nbelement = len(centerc)
    for i in range(nbelement):
        i_xx = 0.
        i_yy = 0.
        i_xy = 0.
        j_xh = 0.
        j_yh = 0

        j_xhu = 0.
        j_yhu = 0.
        j_xhv = 0.
        j_yhv = 0.
        j_xhc = 0.
        j_yhc = 0.
        j_xz = 0.
        j_yz = 0.

        for j in range(len(cellnidc[i])):
            cell = cellnidc[i][j]
#        for k in range(3):
#            nod = nodeidc[i][k]
#            for j in range(len(cellnid[nod])):
#                cell = cellnid[nod][j]
            if cell != -1:
                j_x = centerc[cell][0] - centerc[i][0]
                j_y = centerc[cell][1] - centerc[i][1]
                i_xx += pow(j_x, 2)
                i_yy += pow(j_y, 2)
                i_xy += (j_x * j_y)

                j_xh += (j_x * (w_c[cell].h - w_c[i].h))
                j_yh += (j_y * (w_c[cell].h - w_c[i].h))

                j_xhu += (j_x * (w_c[cell].hu - w_c[i].hu))
                j_yhu += (j_y * (w_c[cell].hu - w_c[i].hu))

                j_xhv += (j_x * (w_c[cell].hv - w_c[i].hv))
                j_yhv += (j_y * (w_c[cell].hv - w_c[i].hv))

                j_xz += (j_x * (w_c[cell].Z - w_c[i].Z))
                j_yz += (j_y * (w_c[cell].Z - w_c[i].Z))
        
        for k in range(3):
            nod = nodeidc[i][k]
                   
            if namen[nod] != 0 and namen[nod] != 10:
                for j in range(2):
                    center = centerg[nod][j]
                    cell = np.int(centerg[nod][j][2])
                    
                    j_x = center[0] - centerc[i][0]
                    j_y = center[1] - centerc[i][1]
                    i_xx += pow(j_x, 2)
                    i_yy += pow(j_y, 2)
                    i_xy += (j_x * j_y)

                    j_xh += (j_x * (w_ghost[cell].h - w_c[i].h))
                    j_yh += (j_y * (w_ghost[cell].h - w_c[i].h))

                    j_xhu += (j_x * (w_ghost[cell].hu - w_c[i].hu))
                    j_yhu += (j_y * (w_ghost[cell].hu - w_c[i].hu))

                    j_xhv += (j_x * (w_ghost[cell].hv - w_c[i].hv))
                    j_yhv += (j_y * (w_ghost[cell].hv - w_c[i].hv))
                    j_xz += (j_x * (w_ghost[cell].Z - w_c[i].Z))
                    j_yz += (j_y * (w_ghost[cell].Z - w_c[i].Z))


            if SIZE > 1:
                for j in range(len(haloidn[nod])):
                    cell = haloidn[nod][j]
                    if cell != -1:
                        j_x = centerh[cell][0] - centerc[i][0]
                        j_y = centerh[cell][1] - centerc[i][1]
                        i_xx += pow(j_x, 2)
                        i_yy += pow(j_y, 2)
                        i_xy += (j_x * j_y)

                        j_xh += (j_x * (w_halo[cell].h - w_c[i].h))
                        j_yh += (j_y * (w_halo[cell].h - w_c[i].h))

                        j_xhu += (j_x * (w_halo[cell].hu - w_c[i].hu))
                        j_yhu += (j_y * (w_halo[cell].hu - w_c[i].hu))

                        j_xhv += (j_x * (w_halo[cell].hv - w_c[i].hv))
                        j_yhv += (j_y * (w_halo[cell].hv - w_c[i].hv))

                        j_xz += (j_x * (w_halo[cell].Z - w_c[i].Z))
                        j_yz += (j_y * (w_halo[cell].Z - w_c[i].Z))


        dia = i_xx * i_yy - pow(i_xy, 2)

        w_x[i].h = (i_yy * j_xh - i_xy * j_yh) / dia
        w_y[i].h = (i_xx * j_yh - i_xy * j_xh) / dia

        w_x[i].hu = (i_yy * j_xhu - i_xy * j_yhu) / dia
        w_y[i].hu = (i_xx * j_yhu - i_xy * j_xhu) / dia

        w_x[i].hv = (i_yy * j_xhv - i_xy * j_yhv) / dia
        w_y[i].hv = (i_xx * j_yhv - i_xy * j_xhv) / dia

        w_x[i].Z = (i_yy * j_xz - i_xy * j_yz) / dia
        w_y[i].Z = (i_xx * j_yz - i_xy * j_xz) / dia

    return w_x, w_y


@njit(fastmath=True)
def barthlimiter(w_c, w_x, w_y, psi, cellid, faceid, centerc, centerf):
    var = "hu"
    nbelement = len(w_c)
    for i in range(nbelement): psi[i] = 1

    for i in range(nbelement):
        w_max = w_c[var][i]
        w_min = w_c[var][i]

        for j in range(3):
            face = faceid[i][j]
            if cellid[face][1] != -1:
                w_max = max(w_max, w_c[var][cellid[face][0]], w_c[var][cellid[face][1]])
                w_min = min(w_min, w_c[var][cellid[face][0]], w_c[var][cellid[face][1]])
            else:
                w_max = max(w_max, w_c[var][cellid[face][0]])
                w_min = min(w_min, w_c[var][cellid[face][0]])

        for j in range(3):
            face = faceid[i][j]

            if cellid[face][1] != -1:
                r_xy = np.array([centerf[face][0] - centerc[i][0],
                                 centerf[face][1] - centerc[i][1]])
                delta2 = w_x[var][i] * r_xy[0] + w_y[var][i] * r_xy[1]

                if np.fabs(delta2) < 1e-8:
                    psi_ij = 1.
                else:
                    if delta2 > 0.:
                        value = (w_max - w_c[var][i]) / delta2
                        psi_ij = min(1., value)
                    if delta2 < 0.:
                        value = (w_min - w_c[var][i]) / delta2
                        psi_ij = min(1., value)

                psi[i] = min(psi[i], psi_ij)
            else:
                psi[i] = 1

    return psi

@njit(fastmath=True)
def albada(wleft, wright, w_x, w_y, center_left, center_right, lim):

    var_a = 0.
    var_b = 0.
    omega = 1.#2./3
    epsilon = 0.
    limit = 0

    var_t = np.array([(center_right[0] - center_left[0]), (center_right[1] - center_left[1])])

    var_h = np.array([w_x.h, w_y.h])
    var_a = omega * np.dot(var_h, var_t) + (1-omega) * (wright.h - wleft.h)
    var_b = wright.h - wleft.h
    if (var_a*var_b) > 0.:
        limit = ((var_a**2 + epsilon)*var_b + (var_b**2 + epsilon)*var_a) / (var_a**2 + var_b**2)
    else:
        limit = 0.

    lim.h = limit

    var_hu = np.array([w_x.hu, w_y.hu])
    var_a = omega * np.dot(var_hu, var_t) + (1-omega) * (wright.hu - wleft.hu)
    var_b = wright.hu - wleft.hu
    if (var_a*var_b) > 0.:
        limit = ((var_a**2 + epsilon)*var_b + (var_b**2 + epsilon)*var_a) / (var_a**2 + var_b**2)
    else:
        limit = 0.

    lim.hu = limit

    var_hv = np.array([w_x.hv, w_y.hv])
    var_a = omega * np.dot(var_hv, var_t) + (1-omega) * (wright.hv - wleft.hv)
    var_b = wright.hv - wleft.hv
    if (var_a*var_b) > 0.:
        limit = ((var_a**2 + epsilon)*var_b + (var_b**2 + epsilon)*var_a) / (var_a**2 + var_b**2)
    else:
        limit = 0.

    lim.hv = limit

    var_Z = np.array([w_x.Z, w_y.Z])
    var_a = omega * np.dot(var_Z, var_t) + (1-omega) * (wright.Z - wleft.Z)
    var_b = wright.Z - wleft.Z
    if (var_a*var_b) > 0.:
        limit = ((var_a**2 + epsilon)*var_b + (var_b**2 + epsilon)*var_a) / (var_a**2 + var_b**2)
    else:
        limit = 0.
    lim.Z = limit
      
    
    return lim


@njit(fastmath=True)
def minmod(w_c, w_x, w_y, limx, limy, nodeidc, cellidn, mystruct):

    Ux = np.zeros(1, dtype=mystruct)[0]
    Uy = np.zeros(1, dtype=mystruct)[0]
    nbelement = len(w_c)
    
    for i in range(nbelement):

        min_sgnU_x = np.zeros(1, dtype=mystruct)[0]
        min_sgnU_y = np.zeros(1, dtype=mystruct)[0]
        max_sgnU_x = np.zeros(1, dtype=mystruct)[0]
        max_sgnU_y = np.zeros(1, dtype=mystruct)[0]
        min_U_x = np.zeros(1, dtype=mystruct)[0]
        min_U_y = np.zeros(1, dtype=mystruct)[0]

        for j in range(3):
            if j == 0:
                min_sgnU_x.h = 1.
                min_sgnU_x.hu = 1.
                min_sgnU_x.hv = 1.
                min_sgnU_x.Z = 1.
                
                min_sgnU_y.h = 1.
                min_sgnU_y.hu = 1.
                min_sgnU_y.hv = 1.
                min_sgnU_y.Z = 1.
                
                max_sgnU_x.h = -1.
                max_sgnU_x.hu = -1.
                max_sgnU_x.hv = -1.
                max_sgnU_x.Z = -1.
                
                max_sgnU_y.h = -1.
                max_sgnU_y.hu = -1.
                max_sgnU_y.hv = -1.
                max_sgnU_y.Z = -1.
            
            nod = nodeidc[i][j]
            for k in range(len(cellidn[nod])):
                cell = cellidn[nod][k]
                if cell != -1:

                    Ux.h = w_x[cell].h
                    Uy.h = w_y[cell].h
                    Ux.hu = w_x[cell].hu
                    Uy.hu = w_y[cell].hu
                    Ux.hv = w_x[cell].hv
                    Uy.hv = w_y[cell].hv
                    Ux.Z = w_x[cell].Z
                    Uy.Z = w_y[cell].Z
                    
                    if (Ux.h == 0): Ux.h = 0.
                    else: Ux.h /= np.fabs(Ux.h)
                    if (Ux.hu == 0): Ux.hu = 0.
                    else: Ux.hu /= np.fabs(Ux.hu)
                    if (Ux.hv == 0): Ux.hv = 0.;
                    else: Ux.hv /= np.fabs(Ux.hv);
                    if (Ux.Z == 0): Ux.Z = 0.;
                    else: Ux.Z /= np.fabs(Ux.Z);
                    if (Uy.h == 0): Uy.h = 0.
                    else: Uy.h /= np.fabs(Uy.h)
                    if (Uy.hu == 0): Uy.hu = 0.;
                    else: Uy.hu /= np.fabs(Uy.hu)
                    if (Uy.hv == 0): Uy.hv = 0.
                    else: Uy.hv /= np.fabs(Uy.hv)
                    if (Uy.Z == 0): Uy.Z = 0.
                    else: Uy.Z /= np.fabs(Uy.Z)
                    
                    min_sgnU_x.h  = min(min_sgnU_x.h, Ux.h)
                    min_sgnU_x.hu = min(min_sgnU_x.hu, Ux.hu)
                    min_sgnU_x.hv = min(min_sgnU_x.hv, Ux.hv)
                    min_sgnU_x.Z  = min(min_sgnU_x.Z, Ux.Z)
                    min_sgnU_y.h  = min(min_sgnU_y.h, Uy.h)
                    min_sgnU_y.hu = min(min_sgnU_y.hu, Uy.hu)
                    min_sgnU_y.hv = min(min_sgnU_y.hv, Uy.hv)
                    min_sgnU_y.Z  = min(min_sgnU_y.Z, Uy.Z)
                    
                    max_sgnU_x.h  = max(max_sgnU_x.h, Ux.h)
                    max_sgnU_x.hu = max(max_sgnU_x.hu, Ux.hu)
                    max_sgnU_x.hv = max(max_sgnU_x.hv, Ux.hv)
                    max_sgnU_x.Z  = max(max_sgnU_x.Z, Ux.Z)
                    max_sgnU_y.h  = max(max_sgnU_y.h, Uy.h)
                    max_sgnU_y.hu = max(max_sgnU_y.hu, Uy.hu)
                    max_sgnU_y.hv = max(max_sgnU_y.hv, Uy.hv)
                    max_sgnU_y.Z  = max(max_sgnU_y.Z, Uy.Z)

        min_U_y.h = np.fabs(w_y[i].h)
        min_U_y.hu = np.fabs(w_y[i].hu)
        min_U_y.hv = np.fabs(w_y[i].hv)
        min_U_y.Z = np.fabs(w_y[i].Z)
        
        min_U_x.h = np.fabs(w_x[i].h)
        min_U_x.hu = np.fabs(w_x[i].hu)
        min_U_x.hv = np.fabs(w_x[i].hv)
        min_U_x.Z = np.fabs(w_x[i].Z)
        
        for j in range(3):
            nod = nodeidc[i][j]
            for k in range(len(cellidn[nod])):
                cell = cellidn[nod][k]
                if cell != -1:
        
                    Ux = w_x[cell];
                    Uy = w_y[cell];
                                    
                    min_U_x.h  = min(min_U_x.h, np.fabs(Ux.h))
                    min_U_x.hu = min(min_U_x.hu, np.fabs(Ux.hu))
                    min_U_x.hv = min(min_U_x.hv, np.fabs(Ux.hv))
                    min_U_x.Z  = min(min_U_x.Z, np.fabs(Ux.Z))
                    
                    min_U_y.h  = min(min_U_y.h, np.fabs(Uy.h))
                    min_U_y.hu = min(min_U_y.hu, np.fabs(Uy.hu))
                    min_U_y.hv = min(min_U_y.hv, np.fabs(Uy.hv))
                    min_U_y.Z  = min(min_U_y.Z, np.fabs(Uy.Z))

        limx[i].h = 0.5 * (min_sgnU_x.h + max_sgnU_x.h) * min_U_x.h;
        limy[i].h = 0.5 * (min_sgnU_y.h + max_sgnU_y.h) * min_U_y.h;
        
        limx[i].hu = 0.5 * (min_sgnU_x.hu + max_sgnU_x.hu) * min_U_x.hu;
        limy[i].hu = 0.5 * (min_sgnU_y.hu + max_sgnU_y.hu) * min_U_y.hu;
        
        limx[i].hv = 0.5 * (min_sgnU_x.hv + max_sgnU_x.hv) * min_U_x.hv;
        limy[i].hv = 0.5 * (min_sgnU_y.hv + max_sgnU_y.hv) * min_U_y.hv;
        
        limx[i].Z = 0.5 * (min_sgnU_x.Z + max_sgnU_x.Z) * min_U_x.Z
        limy[i].Z = 0.5 * (min_sgnU_y.Z + max_sgnU_y.Z) * min_U_y.Z

    return limx, limy

@njit(fastmath=True)
def update(w_c, wnew, dtime, rez, src, corio, vol):
    nbelement = len(w_c)

    for i in range(nbelement):

        wnew.h[i] = w_c.h[i]  + dtime  * ((rez["h"][i]  + src["h"][i])/vol[i]  + corio["h"][i])
        wnew.hu[i] = w_c.hu[i] + dtime * ((rez["hu"][i] + src["hu"][i])/vol[i] + corio["hu"][i])
        wnew.hv[i] = w_c.hv[i] + dtime * ((rez["hv"][i] + src["hv"][i])/vol[i] + corio["hv"][i])
        wnew.Z[i] = w_c.Z[i]  + dtime  * ((rez["Z"][i]  + src["Z"][i])/vol[i]  + corio["Z"][i])

    return wnew

@njit(fastmath=True)
def time_step(w_c, cfl, normal, mesure, volume, faceid, grav, term_convective):
    nbelement =  len(faceid)
    dt_c = 1e6
    epsilon = 1e-6
    As=1
    p=0.5
    m=1
    zeta=1/(1-p)
    
    
    for i in range(nbelement):
        velson = np.sqrt(grav*w_c[i].h)
        lam = 0
        
        if term_convective == "on":
            for j in range(3):
                norm = normal[faceid[i][j]]
                mesn = mesure[faceid[i][j]]
                d=As*zeta*(3*(w_c.hu[i]/w_c.h[i])**2*norm[0] + (w_c.hv[i]/w_c.h[i])**2*norm[1])
                P=((w_c.hu[i]/w_c.h[i])**2*norm[0]+3*grav*(w_c.h[i]+d))/9
                R=(w_c.hu[i]/w_c.h[i])*norm[0]*(9*grav*(2*w_c.h[i]-d)-2*(w_c.hu[i]/w_c.h[i])**2*norm[0])/54
                teta=np.arccos(R/(np.sqrt(P**3)))
                
                lambda1=2*np.sqrt(P)*np.cos(teta/3)+(2/3)*(w_c.hu[i]/w_c.h[i])*norm[0]
                lambda2=2*np.sqrt(P)*np.cos((teta+2*np.pi)/3)+(2/3)*(w_c.hu[i]/w_c.h[i])*norm[0]
                lambda3=2*np.sqrt(P)*np.cos((teta+4*np.pi)/3)+(2/3)*(w_c.hu[i]/w_c.h[i])*norm[0]
                lambda4=(w_c.hu[i]/w_c.h[i])*norm[0]
                if np.fabs(w_c.h[i]) > epsilon:
                    lam_convect = max(lambda1, lambda2, lambda3, lambda4)/mesn
                    lam += lam_convect * mesn

            dt_c = min(dt_c, cfl * volume[i]/lam)

    dtime = np.asarray(dt_c)#np.min(dt_c))

    return dtime

@njit(fastmath=True)
def exact_solution(uexact, hexact, zexact, center, time, grav):
    nbelement = len(center)
    h_m = 2.534
    u_m = 4.03
    h_1 = 5
    h_2 = 1.
    
    x_0 = 6.
    c_o = np.sqrt(grav*h_1)
    s_s = np.sqrt(0.5*grav*h_m*(1.+(h_m/h_2)))

    x_1 = x_0 - c_o*time
    x_2 = x_0 + 2.*c_o*time - 3.*np.sqrt(grav*h_m)*time
    x_3 = s_s*time + x_0
#    xl = 40.
#    cmax = 1.
#    sigma = 0.25
#    mu = 0.1

    for i in range(nbelement):
        xcent = center[i][0]
        if xcent < x_1:
            hexact[i] = h_1
            uexact[i] = 0

        elif xcent < x_2 and xcent >= x_1:
            hexact[i] = 1/(9*grav) * (2*np.sqrt(grav*h_1) - (xcent-x_0)/time)**2
            uexact[i] = 2/3 * (np.sqrt(grav*h_1) + (xcent-x_0)/time)

        elif xcent < x_3 and xcent >= x_2:
            hexact[i] = h_m
            uexact[i] = u_m

        elif xcent >= x_3:
            hexact[i] = h_2
            uexact[i] = 0

#
#        if np.fabs(xcent - 1500/2) <= 1500/8 :
#            zexact[i] = 8#10 + (40*xcent/14000) + 10*np.sin(np.pi*(4*xcent/14000 - 0.5))
#    
#        hexact[i] = 20 - zexact[i] - 4*np.sin(np.pi*(4*time/86400 + 0.5))
#        #hexact[i] = 64.5 - zexact[i] - 4*np.sin(np.pi*(4*time/86400 + 0.5))
#    
#        uexact[i] = (xcent - 1500)*np.pi/(5400*hexact[i]) * np.cos(np.pi*(4*time/86400 + 0.5))
#        

#        xcent = center[i][0]
#        ycent = center[i][1]
#        hexact[i] = cmax/(1+(4*mu*time)/sigma**2)*np.exp(-(xcent**2+ycent**2)/(sigma**2 + 4*mu*time))
        #np.sin(alpha*xcent)*np.sin(alpha*ycent) + 0.05*np.cos(np.pi*xcent)*np.cos(np.pi*ycent)


    return uexact, hexact

@njit(fastmath=True)
def initialisation(w_c, center):
    Q=10

    nbelements = len(center) 

    for i in range(nbelements):
          xcent = center[i][0]
          ycent = center[i][1]
          if 500 <= xcent <= 700 and 400 <= ycent <= 600 :
              w_c.Z[i] =0*((np.sin((xcent-500)*(np.pi/200))**2)*(np.sin((xcent-400)*(np.pi/200)))**2)
          else :
              w_c.Z[i] =0.
              
          w_c.h[i] = 10 #w_c.Z[i]
          w_c.hu[i] = Q
          w_c.hv[i] = 0.  
    
    return w_c

@njit
def ghost_value(w_c, w_ghost, cellid, name, normal, mesure, time):
    nbface = len(cellid)
    
    for i in range(nbface):
        w_ghost[i] = w_c[cellid[i][0]]
#        aa = 4*time/L + 0.5
#
#        if name[i] == 1:
#            w_ghost[i].h = 20 - 4*np.sin(np.pi * aa)
#            u = -1500 * np.pi/(5400*w_ghost[i].h) * np.cos(np.pi * aa)
#            v = 0
#
#            w_ghost[i].hu = w_ghost[i].h * u
#            w_ghost[i].hv = w_ghost[i].h * v
#
#        elif name[i] == 2:
#            w_ghost[i].hu = 0
#            w_ghost[i].hv = 0

        if (name[i] == 3 or name[i] == 4):
        #slip conditions
            
            if w_c[cellid[i][0]].h > 1e-6:
        
                u_i = w_c[cellid[i][0]].hu/w_c[cellid[i][0]].h
                v_i = w_c[cellid[i][0]].hv/w_c[cellid[i][0]].h
            
                s_n = normal[i] / mesure[i]
            
                u_g = u_i*(s_n[1]*s_n[1] - s_n[0]*s_n[0]) - 2.0*v_i*s_n[0]*s_n[1]
                v_g = v_i*(s_n[0]*s_n[0] - s_n[1]*s_n[1]) - 2.0*u_i*s_n[0]*s_n[1]
            
                w_ghost[i].h = w_c[cellid[i][0]].h
                w_ghost[i].Z = w_c[cellid[i][0]].Z
            
                w_ghost[i].hu = w_c[cellid[i][0]].h * u_g
                w_ghost[i].hv = w_c[cellid[i][0]].h * v_g
#        
            else:
                w_ghost[i].h = w_c[cellid[i][0]].h
                w_ghost[i].Z = w_c[cellid[i][0]].Z
                w_ghost[i].hu = 0.
                w_ghost[i].hv = 0.

#        elif (name[i] == 2):
#            #w_ghost[i].h = 0#w_c[cellid[i][0]]
#            w_ghost[i].hu = 0
#            w_ghost[i].hv = 0.
#        
#        elif (name[i] == 1):
#            w_ghost[i].hu = -w_c[cellid[i][0]].hu
#            w_ghost[i].hv = -w_c[cellid[i][0]].hv

    return w_ghost

def save_paraview_results(w_c, solexact, niter, miter, time, dtime, rank, size, cells, nodes):

    u = np.zeros(len(w_c))
    v = np.zeros(len(w_c))
    epsilon = 1e-6
    nbelement = len(w_c)

    elements = {"triangle": cells}
    points = []
    for i in nodes:
        points.append([i[0], i[1], i[2]])
    
    for i in range(nbelement):
        if np.fabs(w_c["h"][i]) < epsilon:
            u[i] = 0
            v[i] = 0
        else:
            u[i] = w_c["hu"][i]/w_c["h"][i]
            v[i] = w_c["hv"][i]/w_c["h"][i]

    data = {"h" : w_c["h"], "u" : u, "v": v,
            "Z": w_c["Z"], "h+Z": w_c["h"] + w_c["Z"], "exact": solexact}
    if len(w_c) == len(cells):
        data = {"h": data, "u":data, "v": data,  "Z":data, "h+Z":data, "exact":data}

    maxh = np.zeros(1)
    maxh = max(w_c["h"])
    integral_sum = np.zeros(1)

    COMM.Reduce(maxh, integral_sum, MPI.MAX, 0)
    if rank == 0:
        print(" **************************** Computing ****************************")
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$ Saving Results $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print("Iteration = ", niter, "time = ", np.float16(time), "time step = ", np.float16(dtime))
        print("max h =", np.float16(integral_sum[0]))
    
    if len(w_c) == len(cells):
        meshio.write_points_cells("results/visu"+str(rank)+"-"+str(miter)+".vtu",
                                  points, elements, cell_data=data, file_format="vtu")
    else:
        meshio.write_points_cells("results/visu"+str(rank)+"-"+str(miter)+".vtu",
                                  points, elements, point_data=data, file_format="vtu")

    if(rank == 0 and size > 1):
        with open("results/visu"+str(miter)+".pvtu", "a") as text_file:
            text_file.write("<?xml version=\"1.0\"?>\n")
            text_file.write("<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n")
            text_file.write("<PUnstructuredGrid GhostLevel=\"0\">\n")
            text_file.write("<PPoints>\n")
            text_file.write("<PDataArray type=\"Float64\" Name=\"Points\" NumberOfComponents=\"3\" format=\"binary\"/>\n")
            text_file.write("</PPoints>\n")
            text_file.write("<PCells>\n")
            text_file.write("<PDataArray type=\"Int64\" Name=\"connectivity\" format=\"binary\"/>\n")
            text_file.write("<PDataArray type=\"Int64\" Name=\"offsets\" format=\"binary\"/>\n")
            text_file.write("<PDataArray type=\"Int64\" Name=\"types\" format=\"binary\"/>\n")
            text_file.write("</PCells>\n")
            if len(w_c) == len(cells):
                text_file.write("<PCellData Scalars=\"h\">\n")
                text_file.write("<PDataArray type=\"Float64\" Name=\"h\" format=\"binary\"/>\n")
                text_file.write("<PDataArray type=\"Float64\" Name=\"u\" format=\"binary\"/>\n")
                text_file.write("<PDataArray type=\"Float64\" Name=\"v\" format=\"binary\"/>\n")
                text_file.write("<PDataArray type=\"Float64\" Name=\"Z\" format=\"binary\"/>\n")
                text_file.write("<PDataArray type=\"Float64\" Name=\"h+Z\" format=\"binary\"/>\n")
                text_file.write("<PDataArray type=\"Float64\" Name=\"exact\" format=\"binary\"/>\n")
                text_file.write("</PCellData>\n")
            else:
                text_file.write("<PPointData Scalars=\"h\">\n")
                text_file.write("<PDataArray type=\"Float64\" Name=\"h\" format=\"binary\"/>\n")
                text_file.write("<PDataArray type=\"Float64\" Name=\"u\" format=\"binary\"/>\n")
                text_file.write("<PDataArray type=\"Float64\" Name=\"v\" format=\"binary\"/>\n")
                text_file.write("<PDataArray type=\"Float64\" Name=\"Z\" format=\"binary\"/>\n")
                text_file.write("<PDataArray type=\"Float64\" Name=\"h+Z\" format=\"binary\"/>\n")
                text_file.write("<PDataArray type=\"Float64\" Name=\"exact\" format=\"binary\"/>\n")
                text_file.write("</PPointData>\n")
            for i in range(size):
                name1 = "visu"
                bu1 = [10]
                bu1 = str(i)
                name1 += bu1
                name1 += "-"+str(miter)
                name1 += ".vtu"
                text_file.write("<Piece Source=\""+str(name1)+"\"/>\n")
            text_file.write("</PUnstructuredGrid>\n")
            text_file.write("</VTKFile>")
