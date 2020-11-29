#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 00:30:12 2020

@author: cisse
"""

import timeit
import os
import numpy as np
from mpi4py import MPI
from manapy import ddm
from sedima.fvm import *
from sedima.fonction_utils import *
from sedima.localstructure import *
from sedima.meshpartitioning import *




# ... get the mesh directory
try:
    MESH_DIR = os.environ['MESH_DIR']

except:
    BASE_DIR = os.path.dirname(os.path.realpath(__file__))
    BASE_DIR = os.path.join(BASE_DIR, '..', '..')
    MESH_DIR = os.path.join(BASE_DIR, 'mesh')

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

def test_swep():

    if RANK == 0:
        
        #reading gmsh file and partitioning into size subdomains
        filename = os.path.join(MESH_DIR, "sedi.msh")
        ddm.meshpart(SIZE, filename)
        #removing existing vtk files
        mypath = "results"
        if not os.path.exists(mypath):
            os.mkdir(mypath)
        for root, dirs, files in os.walk(mypath):
            for file in files:
                os.remove(os.path.join(root, file))
    COMM.Barrier()

    start = timeit.default_timer()

    #generating local grid for each subdomain
    grid = {}
    grid = ddm.generate_mesh()

    faces = grid["faces"]
    cells = grid["cells"]
    halos = grid["halos"]
    nodes = grid["nodes"]
    
    nbelements = len(cells.center)
    nbfaces = len(faces.name)
    nbnodes = len(nodes.vertex)

    variables = tuple(['h', 'hu', 'hv', 'Z'])
    mystruct = np.dtype([('h', np.float64),
                         ('hu', np.float64),
                         ('hv', np.float64),
                         ('Z', np.float64),])

    w_c = np.recarray(nbelements, dtype=mystruct)
    w_x = np.recarray(nbelements, dtype=mystruct)
    w_y = np.recarray(nbelements, dtype=mystruct)
    src = np.recarray(nbelements, dtype=mystruct)
    rez = np.recarray(nbelements, dtype=mystruct)
    cor = np.recarray(nbelements, dtype=mystruct)
    w_ghost = np.recarray(nbfaces, dtype=mystruct)
    wx_face = np.zeros(nbfaces)
    wy_face = np.zeros(nbfaces)
    hexact = np.zeros(nbelements)
    zexact = np.zeros(nbelements)
    uexact = np.zeros(nbelements)
    
    #compute the arrays needed for the mpi communication
    scount, sdepl, rcount, rdepl, taille, indsend = ddm.prepare_comm(cells, halos)
    #compute the interpolation variables
    R_x, R_y, lambda_x,lambda_y, number =  ddm.variables(cells.center, nodes.cellid, nodes.halonid,
                                                     nodes.vertex, nodes.name, nodes.ghostcenter,
                                                     halos.centvol)

    w_halosend = np.zeros(len(halos.halosint), dtype=mystruct)
    wx_halosend = np.zeros(len(halos.halosint), dtype=mystruct)
    wy_halosend = np.zeros(len(halos.halosint), dtype=mystruct)

    w_n = w_c
    ###Initialisation
    w_c = initialisation(w_c, cells.center)
    
    if RANK == 0: print("Start Computation")
    cfl = 0.8
    time = 0
    tfinal = 0.2
    order = 1#(1:first order, 2:barth jeperson 3:van albada 4:minmod)
    term_convec = "on"
    term_source = "off"
    grav = 9.81
    f_c = 0.
    

    ####calculation of the time step
    d_t =1e-3 # time_step(w_c, cfl, faces.normal, faces.mesure, cells.volume, cells.faceid, grav, 
                        #term_convec)
    # dt_i = np.zeros(1)
    # COMM.Allreduce(d_t, dt_i, MPI.MIN)
    # d_t = np.float64(dt_i)
    #d_t = 0.0001
    #saving 25 vtk file
    tot = int(tfinal/d_t/50) + 1
    miter = 0
    niter = 0

    #loop over time
    while time < tfinal:

        time = time + d_t

        #update the ghost values for the boundary conditions
        w_ghost = ghost_value(w_c, w_ghost, faces.cellid, faces.name, faces.normal, faces.mesure,
                                  time)
        
       #update the halo values
        w_halosend = ddm.define_halosend(w_c, w_halosend, indsend)
        w_halo = ddm.all_to_all(w_halosend, taille, mystruct, variables,
                                scount, sdepl, rcount, rdepl)
        if order != 1:
            #compute derivative
            w_x, w_y = derivxy(w_c, w_ghost, w_halo, cells.center, cells.nodeid, cells.cellnid, 
                                   nodes.cellid, nodes.halonid, nodes.name, nodes.ghostcenter, 
                                   halos.centvol, w_x, w_y)
        
        #update the halo  derivatives values
        wx_halosend = ddm.define_halosend(w_x, wx_halosend, indsend)
        wy_halosend = ddm.define_halosend(w_y, wy_halosend, indsend)

        wx_halo = ddm.all_to_all(wx_halosend, taille, mystruct, variables,
                                 scount, sdepl, rcount, rdepl)
        wy_halo = ddm.all_to_all(wy_halosend, taille, mystruct, variables,
                                 scount, sdepl, rcount, rdepl)
        
        w_node = np.zeros(nbnodes, dtype=mystruct)
        unode_exact = np.zeros(nbnodes)

        if (term_convec == "on"):
            #update the rezidus using explicit scheme
            rez = explicitscheme_convective(w_c, w_x, w_y, w_ghost, w_halo, wx_halo, wy_halo,
                                               faces.cellid, cells.faceid, cells.nodeid, cells.center, 
                                               cells.cellfid, halos.centvol, faces.mesure,
                                               faces.center, faces.normal, faces.halofid, faces.name, 
                                               faces.ghostcenter, nodes.cellid, mystruct, order, grav)
        if (term_source == "on"):
            #update the source using explicit scheme
            src = term_source_srnh(w_c, w_ghost, w_halo, w_x, w_y, wx_halo, wy_halo, cells.nodeid, 
                                      cells.faceid, cells.cellfid, cells.center, cells.volume, cells.nf,
                                      faces.cellid, faces.nodeid, faces.normal, faces.center, faces.name, 
                                      faces.ghostcenter, nodes.vertex, faces.halofid, halos.centvol,
                                      mystruct, order, grav, src)

        #update the new solution
        w_n = update(w_c, w_n, d_t, rez, src, cor, cells.volume)
        w_c = w_n
        
        #save vtk files for the solution
        if niter%tot == 0:
            saving_at_node = 1
            uexact, hexact = exact_solution(uexact, hexact, zexact, cells.center, time, grav)
           
            if saving_at_node:
                w_node, unode_exact = centertovertex(w_c, w_ghost, w_halo, cells.center, halos.centvol, 
                                                     nodes.cellid, nodes.halonid, nodes.vertex, nodes.name,
                                                     nodes.ghostcenter, w_node, uexact, unode_exact,
                                                     R_x, R_y, lambda_x,lambda_y, number)
                
                save_paraview_results(w_node, unode_exact, niter, miter, time, d_t, RANK, SIZE, cells.nodeid, nodes.vertex)
           
            else:
                save_paraview_results(w_c, uexact, niter, miter, time, d_t, RANK, SIZE, cells.nodeid, nodes.vertex)


            miter += 1

        niter += 1
        ####calculation of the time step
        d_t =0.001#time_step(w_c, cfl, faces.normal, faces.mesure, cells.volume, cells.faceid, grav, 
                       #     term_convec)   
        # dt_i = np.zeros(1)
        # COMM.Allreduce(d_t, dt_i, MPI.MIN)
        # d_t = np.float64(dt_i)
        #d_t = 0.0001
        #saving 25 vtk file
        tot = int(tfinal/d_t/50)


    stop = timeit.default_timer()

    if RANK == 0: print(stop - start)

test_swep()
