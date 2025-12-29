import os, glob, numpy as np
OUT='result_smp'; GT='img/test_mask'
preds = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(os.path.join(OUT,'*.npy'))}
gts = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(os.path.join(GT,'*.npy'))}
common = sorted(set(preds)&set(gts))
print('Common:', len(common))
for k in common:
    pr = np.load(preds[k])
    gt = np.load(gts[k])
    print(k, 'pr', pr.shape, pr.dtype, 'gt', gt.shape, gt.dtype)