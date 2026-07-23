"""从 Materials Project 批量下载 CIF 结构文件

输入: data/routing_table/train.json + eval.json (所有 mp-xxx ID)
输出: data/structures/ 目录下的 CIF 文件

用法: python scripts/download_structures.py --api_key YOUR_MP_API_KEY
"""
import json
import os
import argparse
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))


def main():
    parser = argparse.ArgumentParser(description="下载 Materials Project CIF 结构")
    parser.add_argument("--api_key", required=True, help="Materials Project API key")
    parser.add_argument("--data_dir", default="data/routing_table")
    parser.add_argument("--output_dir", default="data/structures")
    parser.add_argument("--batch_size", type=int, default=500,
                       help="每批下载数量 (MP API 限制)")
    args = parser.parse_args()

    BASE_DIR = Path(__file__).resolve().parent.parent
    data_dir = BASE_DIR / args.data_dir
    output_dir = BASE_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 收集所有唯一 ID
    all_ids = set()
    for fname in ["train_meta.json", "eval_meta.json"]:
        path = data_dir / fname
        if not path.exists():
            print(f"跳过: {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            items = json.load(f)
        ids = {item["id"] for item in items}
        all_ids.update(ids)
        print(f"{fname}: {len(ids)} 个唯一 ID")

    all_ids = sorted(all_ids)
    print(f"\n总计唯一 mp-id: {len(all_ids)}")

    # 检查已下载的
    existing = set()
    for f in output_dir.glob("*.cif"):
        existing.add(f.stem)
    todo = [mid for mid in all_ids if mid not in existing]
    print(f"已下载: {len(existing)} | 待下载: {len(todo)}")

    if not todo:
        print("全部已下载完成！")
        return

    # 分批下载
    total_batches = (len(todo) + args.batch_size - 1) // args.batch_size
    for batch_idx in range(total_batches):
        start = batch_idx * args.batch_size
        end = min(start + args.batch_size, len(todo))
        batch = todo[start:end]

        print(f"\n批次 {batch_idx+1}/{total_batches}: {len(batch)} 个 ({start+1}-{end}/{len(todo)})")

        # 使用 MPRester 批量下载
        try:
            from mp_api.client import MPRester
            with MPRester(args.api_key) as mpr:
                # 按 material_ids 查询
                docs = mpr.materials.summary.search(
                    material_ids=batch,
                    fields=["material_id", "structure"],
                )
                for doc in docs:
                    mid = doc.material_id
                    structure = doc.structure
                    cif_path = output_dir / f"{mid}.cif"
                    structure.to(filename=str(cif_path), fmt="cif")
                print(f"  成功: {len(docs)}/{len(batch)}")
        except Exception as e:
            print(f"  API 错误: {e}")
            print(f"  等待 30 秒后重试...")
            time.sleep(30)
            try:
                from mp_api.client import MPRester
                with MPRester(args.api_key) as mpr:
                    docs = mpr.materials.summary.search(
                        material_ids=batch,
                        fields=["material_id", "structure"],
                    )
                    for doc in docs:
                        mid = doc.material_id
                        structure = doc.structure
                        cif_path = output_dir / f"{mid}.cif"
                        structure.to(filename=str(cif_path), fmt="cif")
                    print(f"  重试成功: {len(docs)}/{len(batch)}")
            except Exception as e2:
                print(f"  重试也失败: {e2}")

        # API 限速
        if batch_idx < total_batches - 1:
            time.sleep(2)

    # 最终统计
    final = set(f.stem for f in output_dir.glob("*.cif"))
    success = len(final.intersection(all_ids))
    missing = len(all_ids) - success
    print(f"\n下载完成: {success}/{len(all_ids)} 成功, {missing} 缺失")


if __name__ == "__main__":
    main()
