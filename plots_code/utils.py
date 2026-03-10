import json
import numpy as np
import time
from glob import glob
import os
import ROOT
from root_numpy import hist2array, array2hist
from root_pandas import read_root
import pandas as pd


def merge(runnumber, N):
    t0 = time.time()
    print("MERGING AND REMOVING CHUNK FILES")

    string = "ecdf_map_Run*chunk*_N*.npy"  # select files with same N
    name = glob.glob(string)
    if len(name) == 0:
        print("No file found with the following format: ecdf_map_Run*chunk*_N*.npy")
        return
    if runnumber == '':
        runnumber = name[0].split('Run')[1].split('chunk')[0]
    if N == 0:
        N = int(name[0].split('_N')[1].split('.')[0])
    del name

    # get files of a specific run and specific N
    string = "ecdf_map_Run"+runnumber+"chunk*_N"+str(N)+".npy"
    name = glob.glob(string)
    if len(name) == 0:
        print("No file found with the following format: "+string)
        return
    name.sort()

    arrays = []
    for n in name:
        print(n)
        data = np.load(n, allow_pickle=True)
        arrays.append(data)
        del data

    print(">> Saving file...")
    string = "ecdf_map_Run"+runnumber+"_N"+str(N)
    np.save(string, np.concatenate(arrays))
    print(">> File saved as %s" % (string))

    print(">> Removing chunk files...")
    name.sort()
    for n in name:
        os.remove(n)

    print("Merging elapsed time: %.2f" % (time.time()-t0))


def root_TH2_name(root_file):
    pic = []
    wfm = []
    for i, e in enumerate(root_file.GetListOfKeys()):
        che = e.GetName()
        if ('pic_run' in str(che)):
            pic.append(che)
        elif ('wfm_run' in str(che)):
            wfm.append(che)
    return pic, wfm

# Genereta ECDF


def ecdf(data):
    """ Compute ECDF """
    unique, counts = np.unique(data, return_counts=True)
    y = np.cumsum(counts)
    y = y / float(y[-1])
    y = np.insert(y, 0, 0)
    unique = np.insert(unique, 0, unique[0]-1)
    return(unique, y)


def generate_ecdf(options):
    t1 = time.time()

    filename = options['filename']
    outputfilename = "ecdf_map_Run"+options['runnumber']
    saveMap = 1
    try:
        f = ROOT.TFile(filename)
        pic, wfm = root_TH2_name(f)
        del wfm
    except:
        print("File not found")
        return

    t0 = time.time()
    if options['nimages']:
        # Number of images to be analyzed (usual max number of images)
        N = options['nimages']
    else:
        N = len(pic)
    iTr = 0
    Njump = 5  # ignore first images (usually they are unstable)
    Nx = np.shape(hist2array(f.Get(pic[iTr])))[
        0]  # number of pixes in a column
    Ny = int(options['ny'])  # Ny represeting the number of rows at time

    # Matrix variable to receive N images
    img_real = np.zeros((N-Njump, Ny, Nx), dtype=np.uint16)

    for i in range(0, int(Nx/Ny)):  # (Nx/Ny)
        print("PROCESSING PART %d of %d" % (i+1, int(Nx/Ny)))
        print(">> Loading images...")
        k = -1
        for iTr in range(Njump, N):
            k = k+1
            if k % 500 == 0:
                f.Close()
                f = ROOT.TFile(filename)
                pic, wfm = root_TH2_name(f)
                del wfm
            img_real[k, :, :] = hist2array(f.Get(pic[iTr]))[
                (Ny*i):(Ny*(i+1)), :]

        print("Time elapsed after loading: %.2f" % (time.time()-t0))
        # ECDF
        print(">> ECDF construction...")
        t0 = time.time()
        ecdf_map = []
        for ii in range(0, Ny):
            ecdf_map.append([])
            for jj in range(0, Nx):
                ecdf_map[ii].append([])
                x, y = ecdf(img_real[:, ii, jj])
                ecdf_map[ii][jj].append(x)
                ecdf_map[ii][jj].append(y)
        print("Time elapsed after ECDF: %.2f" % (time.time()-t0))

        if saveMap == 1:
            print(">> Saving chunk file...")
            string = outputfilename+'chunk'+str(i).zfill(2)+'_N'+str(N)
            np.save(string, np.array(ecdf_map))

    print("Total time elapsed: %.2f" % (time.time()-t1))

    return string


def hypot(x, y):
    ax = float(np.abs(x))
    ay = float(np.abs(y))
    amax = 0
    amin = 0
    if ax > ay:
        amax = ax
        amin = ay
    else:
        amin = ax
        amax = ay

    if amin == 0:

        return amax

    f = amin/amax
    return amax*np.sqrt(1.0 + f*f)


def Hypot(x, y):
    return hypot(x, y) + 0.5


def transform(file_name, radius, it, cl_integral, n_hits):
    df_file = read_root(file_name, columns=['run', 'event', 'nCl', 'cl_integral', 'cl_length', 'cl_width', 'cl_nhits', 'cl_iteration',
                        'cl_xmean', 'cl_ymean', 'im_npixels_filter', 'im_npixels_rebin', 'pipe_filter_time', 'pipe_clustering_time', 'pipe_total_time'])
    processed_df = df_file.drop_duplicates(['run', 'event'], keep='first')
    # no cluster remove
    processed_df = processed_df[processed_df['nCl'] > 0]
    columns_of_lists = ['cl_integral', 'cl_length', 'cl_width',
                        'cl_nhits', 'cl_iteration', 'cl_xmean', 'cl_ymean']
    simple_columns = ['run', 'event', 'nCl', 'im_npixels_filter', 'im_npixels_rebin',
                      'pipe_filter_time', 'pipe_clustering_time', 'pipe_total_time']
    flatten_data = []
    # explode list colums
    for c in columns_of_lists:
        flatten_data.append(
            list(np.concatenate(np.array(processed_df[c].values.tolist()))))

    struct_data_values = np.array(flatten_data).T

    struct_data_dataframe = pd.DataFrame(
        struct_data_values, columns=columns_of_lists)

    const_data = processed_df[simple_columns]
    # join data
    const_data = const_data.loc[const_data.index.repeat(
        const_data.nCl)].reset_index(drop=True)
    transformed_data = pd.concat([const_data, struct_data_dataframe], axis=1)
    transformed_data['roi'] = transformed_data.apply(
        lambda x: Hypot(x.cl_xmean - 1024, (x.cl_ymean - 1024)*1.2), axis=1)
    transformed_data['slimness'] = transformed_data['cl_width'] / \
        transformed_data['cl_length']

    transformed_data['filter_name'] = file_name.split(
        '/')[-1].split('.')[0].split('_')[-2]
    transformed_data['n_pts'] = file_name.split(
        '/')[-1].split('.')[0].split('_')[-1]

    return transformed_data[(transformed_data['roi'] < radius) & (transformed_data['cl_nhits'] > n_hits) & (transformed_data['cl_iteration'] == it) & (transformed_data['cl_integral'] < cl_integral)]
