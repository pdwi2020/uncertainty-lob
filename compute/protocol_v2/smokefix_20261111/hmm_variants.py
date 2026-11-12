import sys, numpy as np
from pathlib import Path
from hmmlearn import hmm
sys.path.insert(0, str(Path(__file__).resolve().parents[3]/"experiments"))
from build_macc_features import build_hmm_features
from fold9_features import FI, load, SEQ
feat, lab = load(FI/"NoAuction_Zscore_Training"/"Train_Dst_NoAuction_ZScore_CF_9.txt")
split = int(len(feat)*0.9); val = feat[split:]; n_win = len(val)-SEQ+1; sl = slice(SEQ-1, SEQ-1+n_win)
def fit(x):
    m = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=100, random_state=42); m.fit(build_hmm_features(x)); return m, int(np.argmax(m.means_[:,0]))
def report(name, r):
    r = r[sl] if len(r) != n_win else r; h = len(r)//2
    print(f"{name:40s} eval low/high {np.bincount(r[h:],minlength=2)}  calib {np.bincount(r[:h],minlength=2)}")
m, hi = fit(feat[:split])
report("fit train, predict full series", (m.predict(build_hmm_features(feat))==hi).astype(int)[split:])
m2, hi2 = fit(val)
report("fit+predict on val block", (m2.predict(build_hmm_features(val))==hi2).astype(int))
cf = [load(FI/"NoAuction_Zscore_Training"/f"Train_Dst_NoAuction_ZScore_CF_{i}.txt")[0] for i in range(1,10)]
cat = np.concatenate(cf); s80 = int(len(cat)*0.8)
m3, hi3 = fit(cat[:s80])
report("fit concat-80% (old builder), predict val", (m3.predict(build_hmm_features(val))==hi3).astype(int))
m4, hi4 = fit(feat)
report("fit full CF_9, predict val", (m4.predict(build_hmm_features(val))==hi4).astype(int))
