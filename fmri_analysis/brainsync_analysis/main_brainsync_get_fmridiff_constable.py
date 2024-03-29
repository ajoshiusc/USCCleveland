from matplotlib.pyplot import axis
import numpy as np
from brainsync import normalizeData, brainSync
import os
from bfp_utils import load_bfp_data
import scipy.io as spio
import glob
from tqdm import tqdm

# get atlas
atlas_data = np.load('Output_Yale_grp_atlas.npz')[
    'atlas_data']  # , X2=X2, Os=Os)
atlas_data2, _, _ = normalizeData(atlas_data)
print(np.max(atlas_data2-atlas_data))

studydir = '/ImagePTE1/brainsuite/Woojae/Analysis/Epilepsy/Output_Yale'
subids = glob.glob(studydir+'/*')

num_sub = len(subids)

sub_files = list()
sub_ids = list()
for subdir in subids:
    sub = os.path.basename(subdir)
    fname = os.path.join(subdir, 'func', sub + '_rest_bold.BOrd.mat')

    if os.path.isfile(fname):
        sub_files.append(fname)
        sub_ids.append(sub)
    else:
        print('File does not exist: %s' % fname)

num_sub = len(sub_files)


f = spio.loadmat(sub_files[0])['dtseries']

numT = f.shape[1]
fmri_data = np.zeros((numT, f.shape[0], num_sub))

for i, f in enumerate(tqdm(sub_files)):

    d = spio.loadmat(f)['dtseries'].T
    fmri_data[:, :, i] = d[:numT, :]

num_vtx = fmri_data.shape[1]


# calculate vertiexwise mean and variance for trainnon epi data
# fmri diff for epilepsy
fdiff_sub = np.zeros((num_vtx, num_sub))

for subno in tqdm(range(num_sub)):
    d, _ = brainSync(atlas_data, fmri_data[:, :, subno])

    data = np.linalg.norm(atlas_data - d, axis=0)
    fdiff_sub[:, subno] = data

fdiff_mean = np.mean(fdiff_sub, axis=1)
fdiff_std = np.std(fdiff_sub, axis=1)


nsub = fmri_data.shape[2]
# fmri diff for epilepsy
fdiff_sub = np.zeros((num_vtx, nsub))
fdiff_sub_z = np.zeros((num_vtx, nsub))

for subno in range(nsub):
    d, _ = brainSync(atlas_data, fmri_data[:, :, subno])
    data = np.linalg.norm(atlas_data - d, axis=0)
    fdiff_sub[:, subno] = data
    fdiff_sub_z[:, subno] = (data - fdiff_mean)/fdiff_std

np.savez('Constable_fmridiff_BOrd.npz',
         fdiff_sub=fdiff_sub,
         fdiff_sub_z=fdiff_sub_z,
         sub_ids=sub_ids)


print('done')
