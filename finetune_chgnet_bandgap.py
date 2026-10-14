"""
CHGNet Band Gap Fine-tuning Pipeline
=====================================
使用 CHGNet 作为冻结特征提取器 -> 训练 MLP 回归头预测 bandgap
"""

import json, os, sys, time, warnings, argparse
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(os.environ.get('CRYSTALAGENT_ROOT', Path(__file__).resolve().parent)).resolve()
DATA_DIR = str(PROJECT_ROOT / "data")
CIF_DIR    = os.path.join(DATA_DIR, "structures")
TRAIN_META = os.path.join(DATA_DIR, "routing_table", "train_meta.json")
EVAL_META  = os.path.join(DATA_DIR, "routing_table", "eval_meta.json")
CACHE_DIR  = os.path.join(DATA_DIR, "chgnet_finetune")
os.makedirs(CACHE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MLP_BATCH, FEAT_DIM, PATIENCE = 256, 64, 15
HIDDEN_DIMS = [128, 64, 32]

def graph_to_device(graph, device):
    for attr in dir(graph):
        if attr.startswith("_"): continue
        try:
            val = getattr(graph, attr)
            if isinstance(val, torch.Tensor):
                setattr(graph, attr, val.to(device))
        except: pass
    return graph

def load_metadata(meta_path, cif_dir, max_samples=None):
    with open(meta_path) as f: records = json.load(f)
    valid, missing, no_label = [], 0, 0
    for r in records:
        bg = r.get("band_gap")
        if bg is None or (isinstance(bg, float) and np.isnan(bg)):
            no_label += 1; continue
        cid = r["id"]
        cif_path = os.path.join(cif_dir, f"{cid}.cif")
        if not os.path.exists(cif_path):
            missing += 1; continue
        valid.append((cid, cif_path, float(bg), r.get("crystal_system", "unknown")))
    if max_samples: valid = valid[:max_samples]
    print(f"  加载 {len(records)} 条 -> 有效 {len(valid)} 条 (无标签 {no_label}, CIF缺失 {missing})")
    return valid

def extract_features(records, cache_file, batch_size=32, device=DEVICE):
    if os.path.exists(cache_file):
        print(f"  [跳过] 缓存已存在: {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
        return data["features"], data["labels"], data["ids"]
    from chgnet.graph.converter import CrystalGraphConverter
    from chgnet.model.model import CHGNet
    from pymatgen.io.cif import CifParser
    print("  加载 CHGNet 模型...")
    model = CHGNet.load(); model.eval(); model.to(device)
    converter = CrystalGraphConverter()
    all_ids, all_feats, all_labels = [], [], []
    pbar = tqdm(range(0, len(records), batch_size), desc="特征提取")
    for start in pbar:
        batch_records = records[start:start+batch_size]
        graphs, batch_ids, batch_labels = [], [], []
        for cid, cif_path, bg, _ in batch_records:
            try:
                s = CifParser(cif_path).parse_structures()[0]
                graph = converter(s)
                graph_to_device(graph, device)
                graphs.append(graph); batch_ids.append(cid); batch_labels.append(bg)
            except Exception as e:
                print(f"    WARN: {cid} 解析失败: {e}")
        if not graphs: continue
        with torch.no_grad():
            result = model(graphs, task="e", return_crystal_feas=True)
        all_feats.append(result["crystal_fea"].cpu().numpy())
        all_labels.extend(batch_labels); all_ids.extend(batch_ids)
        pbar.set_postfix({"ok": len(all_ids)})
    features = np.concatenate(all_feats, axis=0)
    labels = np.array(all_labels, dtype=np.float32)
    ids = np.array(all_ids)
    np.savez(cache_file, features=features, labels=labels, ids=ids)
    print(f"  保存缓存: {cache_file} ({len(ids)} 样本)")
    return features, labels, ids

class BandGapHead(nn.Module):
    def __init__(self, input_dim=64, hidden_dims=None, dropout=0.1):
        super().__init__()
        if hidden_dims is None: hidden_dims = [128, 64, 32]
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev,h), nn.BatchNorm1d(h), nn.SiLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x).squeeze(-1)

def train_mlp(train_feats, train_labels, eval_feats, eval_labels,
              model_path, epochs=100, lr=1e-3, device=DEVICE):
    mean, std = train_labels.mean(), train_labels.std()
    tn = (train_labels - mean) / std
    en = (eval_labels - mean) / std
    tl = DataLoader(TensorDataset(torch.FloatTensor(train_feats), torch.FloatTensor(tn)), batch_size=MLP_BATCH, shuffle=True)
    el = DataLoader(TensorDataset(torch.FloatTensor(eval_feats), torch.FloatTensor(en)), batch_size=MLP_BATCH)
    model = BandGapHead(FEAT_DIM, HIDDEN_DIMS).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=8)
    crit = nn.MSELoss()
    best_loss, pc, history = float("inf"), 0, {"train_loss":[],"eval_loss":[],"eval_mae":[]}
    print()
    print("  {:>6s} {:>10s} {:>10s} {:>10s} {:>10s}".format("Epoch","TrainLoss","EvalLoss","EvalMAE","LR"))
    print("  " + "-"*50)
    for epoch in range(1, epochs+1):
        model.train(); tl_sum = 0.0
        for x,y in tl:
            x,y=x.to(device),y.to(device); opt.zero_grad()
            loss=crit(model(x),y); loss.backward(); opt.step()
            tl_sum+=loss.item()*x.size(0)
        tl_sum/=len(tn); history["train_loss"].append(tl_sum)
        model.eval(); el_sum=0.0; ap,ay=[],[]
        with torch.no_grad():
            for x,y in el:
                x,y=x.to(device),y.to(device); p=model(x)
                el_sum+=crit(p,y).item()*x.size(0); ap.append(p.cpu()); ay.append(y.cpu())
        el_sum/=len(en); history["eval_loss"].append(el_sum)
        preds=torch.cat(ap).numpy()*std+mean; ys=torch.cat(ay).numpy()*std+mean
        emae=np.mean(np.abs(preds-ys)); history["eval_mae"].append(emae)
        sch.step(el_sum); clr=opt.param_groups[0]["lr"]
        print(f"  {epoch:6d} {tl_sum:10.6f} {el_sum:10.6f} {emae:10.4f} {clr:10.2e}")
        if el_sum < best_loss-1e-6:
            best_loss=el_sum; pc=0
            torch.save({"model_state_dict":model.state_dict(),"mean":mean,"std":std,"history":history}, model_path)
        else:
            pc+=1
            if pc>=PATIENCE:
                print(); print(f"  Early stopping at epoch {epoch}"); break
    print()
    print(f"  最佳模型保存至: {model_path}")
    return history, mean, std

def final_evaluate(model_path, eval_feats, eval_labels, device=DEVICE):
    ckpt=torch.load(model_path,map_location=device,weights_only=False)
    mean,std=ckpt["mean"],ckpt["std"]
    model=BandGapHead(FEAT_DIM,HIDDEN_DIMS).to(device)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    with torch.no_grad():
        preds=model(torch.FloatTensor(eval_feats).to(device)).cpu().numpy()*std+mean
    mae=np.mean(np.abs(preds-eval_labels))
    rmse=np.sqrt(np.mean((preds-eval_labels)**2))
    ssr=np.sum((eval_labels-preds)**2); sst=np.sum((eval_labels-np.mean(eval_labels))**2)
    r2=1-ssr/sst if sst>0 else 0
    print()
    print("  ╔══════════════════════════════╗")
    print("  ║  最终评估结果                ║")
    print("  ╠══════════════════════════════╣")
    print(f"  ║  MAE  = {mae:8.4f} eV        ║")
    print(f"  ║  RMSE = {rmse:8.4f} eV        ║")
    print(f"  ║  R2   = {r2:8.4f}            ║")
    print("  ╚══════════════════════════════╝")
    return {"mae":float(mae),"rmse":float(rmse),"r2":float(r2)}

def main():
    p=argparse.ArgumentParser(description="CHGNet Band Gap Fine-tuning")
    p.add_argument("--max-train",type=int,default=None)
    p.add_argument("--max-eval",type=int,default=None)
    p.add_argument("--epochs",type=int,default=100)
    p.add_argument("--lr",type=float,default=1e-3)
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--device",type=str,default=None)
    a=p.parse_args()
    device=torch.device(a.device) if a.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("="*60)
    print(f"  CHGNet -> Band Gap Fine-tuning")
    print(f"  Device: {device}, Feature Batch: {a.batch_size}")
    print("="*60)
    print(); print("[1/4] 加载训练数据...")
    tr=load_metadata(TRAIN_META,CIF_DIR,a.max_train)
    print(); print("[1/4] 加载评估数据...")
    er=load_metadata(EVAL_META,CIF_DIR,a.max_eval)
    tc=os.path.join(CACHE_DIR,f"train_{len(tr)}.npz")
    ec=os.path.join(CACHE_DIR,f"eval_{len(er)}.npz")
    mp=os.path.join(CACHE_DIR,f"model_{len(tr)}.pt")
    print(); print("[2/4] 提取训练特征..."); t0=time.time()
    tf,tl,ti=extract_features(tr,tc,batch_size=a.batch_size,device=device)
    print(f"  耗时: {time.time()-t0:.1f}s")
    print(); print("[2/4] 提取评估特征..."); t0=time.time()
    ef,el,ei=extract_features(er,ec,batch_size=a.batch_size,device=device)
    print(f"  耗时: {time.time()-t0:.1f}s")
    print(); print(f"[3/4] 训练 MLP 回归头 (训练 {len(ti)}, 评估 {len(ei)})...")
    h,mean,std=train_mlp(tf,tl,ef,el,mp,epochs=a.epochs,lr=a.lr,device=device)
    print(); print("[4/4] 最终评估...")
    m=final_evaluate(mp,ef,el,device=device)
    r={"train_samples":len(ti),"eval_samples":len(ei),"metrics":m,"history":h,
       "config":{"hidden_dims":HIDDEN_DIMS,"lr":a.lr,"epochs":a.epochs,"batch_size":a.batch_size,"device":str(device)}}
    rp=os.path.join(CACHE_DIR,f"result_{len(ti)}.json")
    with open(rp,"w") as f: json.dump(r,f,indent=2,default=float)
    print(f"\n  结果已保存至: {rp}")

if __name__=="__main__": main()
