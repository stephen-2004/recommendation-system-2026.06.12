# -*- coding: utf-8 -*-
"""
构建 Amazon 数据集的物品 ID → 名称映射文件
输出：data/Beauty_item_id_to_meta.json
      data/Sports_and_Outdoors_item_id_to_meta.json

映射逻辑与 S3Rec/data/data_process.py 完全一致：
  1. 读取评论文件，筛选 overall > 0 的评论
  2. 按用户分组，按时间戳排序
  3. 5-core 过滤（用户和物品均至少出现 5 次）
  4. 按首次出现顺序分配连续整数 ID（从 1 开始）
  5. 与 meta 文件关联，提取 title/brand/categories

运行：python data/build_amazon_mapping.py
"""

import os
import sys
import json
import gzip
import ast
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AMAZON_DIR = os.path.join(os.path.dirname(BASE_DIR), "amazon_data")


# ── 工具函数（与 S3Rec data_process.py 相同逻辑）─────────────────────────

def get_interaction(datas):
    """(user, item, time) 列表 → {user: [item1, item2, ...]}（按时间排序）"""
    user_seq = {}
    for user, item, time in datas:
        user_seq.setdefault(user, []).append((item, time))
    for user in user_seq:
        user_seq[user].sort(key=lambda x: x[1])
        user_seq[user] = [t[0] for t in user_seq[user]]
    return user_seq


def check_Kcore(user_items, user_core, item_core):
    user_count = defaultdict(int)
    item_count = defaultdict(int)
    for user, items in user_items.items():
        for item in items:
            user_count[user] += 1
            item_count[item] += 1
    for num in user_count.values():
        if num < user_core:
            return user_count, item_count, False
    for num in item_count.values():
        if num < item_core:
            return user_count, item_count, False
    return user_count, item_count, True


def filter_Kcore(user_items, user_core=5, item_core=5):
    user_count, item_count, is_ok = check_Kcore(user_items, user_core, item_core)
    while not is_ok:
        for user in list(user_count):
            if user_count[user] < user_core:
                user_items.pop(user)
            else:
                user_items[user] = [
                    it for it in user_items[user] if item_count[it] >= item_core
                ]
        user_count, item_count, is_ok = check_Kcore(user_items, user_core, item_core)
    return user_items


def id_map(user_items):
    """与 S3Rec 完全相同：按用户首次出现顺序分配整数 ID"""
    item2id = {}   # asin → int_id
    id2item = {}   # int_id → asin
    item_id = 1
    for items in user_items.values():
        for asin in items:
            if asin not in item2id:
                item2id[asin] = str(item_id)
                id2item[str(item_id)] = asin
                item_id += 1
    return id2item, item_id - 1


# ── 读取评论文件（支持 .json.gz 和 .json）────────────────────────────────

def read_reviews(filepath, rating_score=0.0):
    """读取评论文件，返回 (user, asin, timestamp) 列表"""
    datas = []
    open_fn = gzip.open if filepath.endswith(".gz") else open
    mode = "rb" if filepath.endswith(".gz") else "r"
    print(f"  读取评论文件: {os.path.basename(filepath)}")
    count = 0
    with open_fn(filepath, mode) as f:
        for raw in f:
            count += 1
            if count % 500000 == 0:
                print(f"    已读 {count:,} 行，有效记录 {len(datas):,}...")
            try:
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", errors="replace")
                # 尝试标准 JSON（双引号）
                try:
                    obj = json.loads(raw.strip())
                except json.JSONDecodeError:
                    obj = ast.literal_eval(raw.strip())
                overall = float(obj.get("overall", 0))
                if overall <= rating_score:
                    continue
                user = obj["reviewerID"]
                asin = obj["asin"]
                ts = int(obj.get("unixReviewTime", 0))
                datas.append((user, asin, ts))
            except Exception:
                pass
    print(f"  共 {count:,} 行，过滤后有效 {len(datas):,} 条")
    return datas


# ── 读取 meta 文件（Python dict 字面量格式）──────────────────────────────

def read_meta(filepath):
    """返回 {asin: {"title": ..., "brand": ..., "categories": ...}}"""
    meta = {}
    print(f"  读取 meta 文件: {os.path.basename(filepath)}")
    count = 0
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            count += 1
            if count % 200000 == 0:
                print(f"    已读 {count:,} 行...")
            raw = raw.strip()
            if not raw:
                continue
            try:
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    obj = ast.literal_eval(raw)
                asin = obj.get("asin", "")
                if not asin:
                    continue
                title = obj.get("title", "")
                brand = obj.get("brand", "")
                cats = obj.get("categories", [])
                # categories 是嵌套列表 [[主类, 子类, ...], ...]
                cat_flat = []
                for cat_list in cats:
                    cat_flat.extend(cat_list[1:] if len(cat_list) > 1 else cat_list)
                meta[asin] = {
                    "title": title,
                    "brand": brand,
                    "categories": cat_flat,
                }
            except Exception:
                pass
    print(f"  共读取 {len(meta):,} 条 meta 记录")
    return meta


# ── 主流程 ──────────────────────────────────────────────────────────────

def build_mapping(dataset_name, review_file, meta_file, expected_item_num=None):
    """
    为一个数据集构建 item_id → meta 映射，输出到 data/{dataset_name}_item_id_to_meta.json
    """
    print(f"\n{'='*60}")
    print(f"处理数据集: {dataset_name}")
    print(f"{'='*60}")

    # 1. 读取评论数据
    datas = read_reviews(review_file)

    # 2. 按用户分组并排序
    print("  构建用户序列...")
    user_items = get_interaction(datas)
    print(f"  初始用户数: {len(user_items):,}")

    # 3. 5-core 过滤
    print("  执行 5-core 过滤...")
    user_items = filter_Kcore(user_items, user_core=5, item_core=5)
    # 去除空序列
    user_items = {u: items for u, items in user_items.items() if items}
    print(f"  过滤后用户数: {len(user_items):,}")

    # 4. 分配整数 ID
    print("  分配整数 ID...")
    id2item, item_num = id_map(user_items)
    print(f"  物品总数: {item_num:,}")

    if expected_item_num is not None:
        if item_num == expected_item_num:
            print(f"  [OK] 物品数匹配预期 ({expected_item_num})")
        else:
            print(f"  [WARN] 警告：物品数 {item_num} 与预期 {expected_item_num} 不符！")
            print(f"    映射文件仍会生成，但 item_id 可能不准确。")

    # 5. 读取 meta 数据
    meta = read_meta(meta_file)

    # 6. 构建最终映射：int_id → {title, brand, categories}
    print("  构建最终映射...")
    result = {}
    missing = 0
    for int_id, asin in id2item.items():
        m = meta.get(asin)
        if m and m.get("title"):
            result[int_id] = {
                "title": m["title"],
                "brand": m["brand"],
                "categories": m["categories"],
                "asin": asin,
            }
        else:
            # meta 中没有该 ASIN 的信息，使用 ASIN 作为兜底
            result[int_id] = {
                "title": f"[{asin}]",
                "brand": "",
                "categories": [],
                "asin": asin,
            }
            missing += 1

    print(f"  有 title 的物品: {item_num - missing:,}，缺失 meta: {missing:,}")

    # 7. 保存
    out_file = os.path.join(BASE_DIR, f"{dataset_name}_item_id_to_meta.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  [SAVED] 已保存到: {out_file}")
    return out_file


if __name__ == "__main__":
    # ── Beauty ──────────────────────────────────────────────────────────
    # S3Rec 使用的是 5-core 预过滤版本 reviews_Beauty_5.json.gz
    beauty_review = os.path.join(AMAZON_DIR, "reviews_Beauty_5.json.gz")
    if not os.path.exists(beauty_review):
        # 若没有 5-core 版，使用完整版（结果相同，但速度较慢）
        beauty_review = os.path.join(AMAZON_DIR, "reviews_Beauty.json")

    beauty_meta = os.path.join(AMAZON_DIR, "meta_Beauty.json")

    # item_size=12103，其中 item 0 为 padding，mask_id=12102
    # 有效物品 ID 为 1..12101，即 item_num = 12101
    build_mapping(
        dataset_name="Beauty",
        review_file=beauty_review,
        meta_file=beauty_meta,
        expected_item_num=12101,
    )

    # ── Sports_and_Outdoors ─────────────────────────────────────────────
    # !! 重要说明 !!
    # amazon_data/Sports_and_Outdoors.json 是 2019 年完整版数据集（1300 万条）
    # 而 data/Sports_and_Outdoors.txt 是用 2014 年 5-core 预过滤版（29.6 万条）生成的
    # 直接使用完整版会得到错误的物品 ID 映射（103911 个而非 18357 个）
    #
    # 要生成正确的 Sports 映射，请：
    # 1. 下载 reviews_Sports_and_Outdoors_5.json.gz（2014 年版 5-core，约 28MB）
    #    来源：http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/
    #           reviews_Sports_and_Outdoors_5.json.gz
    # 2. 将下载的文件放到 amazon_data/ 目录
    # 3. 取消下面代码块的注释并重新运行本脚本

    sports_5core = os.path.join(AMAZON_DIR, "reviews_Sports_and_Outdoors_5.json.gz")
    sports_meta  = os.path.join(AMAZON_DIR, "meta_Sports_and_Outdoors.json")

    if os.path.exists(sports_5core):
        build_mapping(
            dataset_name="Sports_and_Outdoors",
            review_file=sports_5core,
            meta_file=sports_meta,
            expected_item_num=18357,
        )
    else:
        print("\n[SKIP] Sports_and_Outdoors:")
        print(f"  找不到 {sports_5core}")
        print("  请下载 reviews_Sports_and_Outdoors_5.json.gz（2014版5-core）并放入 amazon_data/")
        print("  下载地址: http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/")

    print("\n全部完成！")
