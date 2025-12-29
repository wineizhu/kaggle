import os, glob, cv2, numpy as np, torch
import albumentations as A
import segmentation_models_pytorch as smp
from tqdm import tqdm

IMG_SIZE=512
BACKBONE='resnet34'
WEIGHTS='./weights/best.pth'
IN_DIR='img/test_img'
OUT_DIR='result_smp'
GT_DIR='img/test_mask'
THRESH=0.5
os.makedirs(OUT_DIR, exist_ok=True)

transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = smp.Unet(encoder_name=BACKBONE, encoder_weights=None, in_channels=3, classes=1)
state = torch.load(WEIGHTS, map_location=device)
model.load_state_dict(state)
model.to(device).eval()

paths = sorted(glob.glob(os.path.join(IN_DIR,'*.png')))
print('Found images:', len(paths))
for p in tqdm(paths, desc='Infer'):
    img = cv2.imread(p)
    h,w = img.shape[:2]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = transform(image=img)['image']
    t = np.transpose(t, (2,0,1)).astype('float32')
    x = torch.from_numpy(t).unsqueeze(0).to(device)
    with torch.no_grad():
        y = model(x)
        prob = torch.sigmoid(y)[0,0].cpu().numpy()
    prob = cv2.resize(prob, (w,h), interpolation=cv2.INTER_LINEAR)
    base = os.path.splitext(os.path.basename(p))[0]
    np.save(os.path.join(OUT_DIR, base+'.npy'), prob.astype(np.float32))

preds = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(os.path.join(OUT_DIR, '*.npy'))}
gts = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(os.path.join(GT_DIR, '*.npy'))}
common = sorted(set(preds) & set(gts))
print('Common with GT:', len(common))

def iou(a,b):
    a = a.astype(bool); b = b.astype(bool)
    inter = np.logical_and(a,b).sum(); uni = np.logical_or(a,b).sum()
    return (inter/uni) if uni>0 else 1.0

scores = []
# accumulate per-sample metrics and global counts
for k in common:
    pr = (np.load(preds[k])>=THRESH).astype(np.uint8)
    gt = np.load(gts[k])
    # collapse possible channel dimension in GT (e.g., (C,H,W))
    if gt.ndim == 3:
        gt = gt.max(axis=0)
    gt = (gt > 0).astype(np.uint8)
    if pr.shape != gt.shape:
        pr = cv2.resize(pr, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

    a = pr.astype(bool); b = gt.astype(bool)
    inter = np.logical_and(a,b).sum()
    uni = np.logical_or(a,b).sum()
    tp = inter
    fp = np.logical_and(a, ~b).sum()
    fn = np.logical_and(~a, b).sum()

    iou_val = (inter/uni) if uni>0 else 1.0
    dice_den = (2*tp + fp + fn)
    dice = (2*tp/dice_den) if dice_den>0 else 1.0
    precision = (tp/(tp+fp)) if (tp+fp)>0 else 1.0
    recall = (tp/(tp+fn)) if (tp+fn)>0 else 1.0

    scores.append((k, iou_val, dice, precision, recall, int(tp), int(fp), int(fn)))

if scores:
    # sort by IoU desc
    scores.sort(key=lambda x: x[1], reverse=True)
    ious = np.array([s[1] for s in scores])
    dices = np.array([s[2] for s in scores])
    precisions = np.array([s[3] for s in scores])
    recalls = np.array([s[4] for s in scores])
    tps = np.array([s[5] for s in scores], dtype=np.int64)
    fps = np.array([s[6] for s in scores], dtype=np.int64)
    fns = np.array([s[7] for s in scores], dtype=np.int64)

    overall_tp = int(tps.sum())
    overall_fp = int(fps.sum())
    overall_fn = int(fns.sum())
    of1_den = (2*overall_tp + overall_fp + overall_fn)
    oF1 = (2*overall_tp/of1_den) if of1_den>0 else 1.0

    print('Samples:', len(scores))
    print('Mean IoU: %.4f  Median IoU: %.4f' % (np.mean(ious), np.median(ious)))
    print('Mean Dice: %.4f' % (np.mean(dices)))
    print('Precision: %.4f  Recall: %.4f' % (np.mean(precisions), np.mean(recalls)))
    print('oF1: %.4f' % (oF1))

    # per-sample brief listing: name, IoU, Dice, Precision, Recall
    for k,iou_val,dice,prec,rec,_,_,_ in scores:
        print(f'{k}: IoU={iou_val:.4f}  Dice={dice:.4f}  P={prec:.4f}  R={rec:.4f}')
else:
    print('No overlap between predictions and GT masks')
