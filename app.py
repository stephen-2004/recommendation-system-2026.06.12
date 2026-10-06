# -*- coding: utf-8 -*-
"""
CoSeRec 推荐系统后端
使用 Python 内置 http.server，无需安装额外依赖
运行：python app.py
访问：http://localhost:8000
"""

import sys
import os
import json
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from types import SimpleNamespace

# 将 src 目录加入 Python 路径
SRC_DIR = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, SRC_DIR)

import torch
from models import SASRecModel

# ── 数据集配置（从各自的 .txt 日志文件中提取）─────────────────────────
BASE_DIR = os.path.dirname(__file__)

DATASETS = {
    "Beauty": {
        "label": "Amazon Beauty",
        "item_size": 12103,
        "pt_file": os.path.join(BASE_DIR, "src/output/CoSeRec-Beauty-0.pt"),
        "data_file": os.path.join(BASE_DIR, "data/Beauty.txt"),
        # 物品名称映射文件（由 data/build_amazon_mapping.py 生成）
        "meta_file": os.path.join(BASE_DIR, "data/Beauty_item_id_to_meta.json"),
    },
    "Sports_and_Outdoors": {
        "label": "Amazon Sports & Outdoors",
        "item_size": 18359,
        "pt_file": os.path.join(BASE_DIR, "src/output/CoSeRec-Sports_and_Outdoors-0.pt"),
        "data_file": os.path.join(BASE_DIR, "data/Sports_and_Outdoors.txt"),
        # 需要 reviews_Sports_and_Outdoors_5.json.gz（2014版5-core）才能生成正确映射
        # 生成后放入 data/ 目录并命名为 Sports_and_Outdoors_item_id_to_meta.json
        "meta_file": os.path.join(BASE_DIR, "data/Sports_and_Outdoors_item_id_to_meta.json"),
    },
    "Yelp": {
        "label": "Yelp",
        "item_size": 20035,
        "pt_file": os.path.join(BASE_DIR, "src/output/CoSeRec-Yelp-0.pt"),
        "data_file": os.path.join(BASE_DIR, "data/Yelp.txt"),
        # Yelp 已有现成映射文件
        "meta_file": os.path.join(BASE_DIR, "data/yelp_item_id_to_meta.json"),
    },
}

# 公共模型参数（三个数据集相同）
COMMON_ARGS = dict(
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=2,
    hidden_act="gelu",
    attention_probs_dropout_prob=0.5,
    hidden_dropout_prob=0.5,
    initializer_range=0.02,
    max_seq_length=50,
    cuda_condition=False,
    no_cuda=True,
)

# ── 模型 & 数据缓存（懒加载，首次请求时加载）──────────────────────────
_model_cache = {}      # dataset_name -> SASRecModel
_seq_cache   = {}      # dataset_name -> list of item lists
_meta_cache  = {}      # dataset_name -> {str(item_id): {"title":..., ...}}


def load_item_meta(dataset_name):
    """懒加载物品名称映射表，不存在时返回空字典（前端降级为仅显示 ID）"""
    if dataset_name in _meta_cache:
        return _meta_cache[dataset_name]

    meta_file = DATASETS[dataset_name].get("meta_file", "")
    if not meta_file or not os.path.exists(meta_file):
        _meta_cache[dataset_name] = {}
        return {}

    with open(meta_file, encoding="utf-8") as f:
        raw = json.load(f)
    # 统一格式：确保 key 为 str(item_id)，取 title 或 name 字段
    meta = {}
    for k, v in raw.items():
        title = v.get("title") or v.get("name") or ""
        brand = v.get("brand", "")
        cats  = v.get("categories", [])
        if isinstance(cats, list):
            cat_str = ", ".join(cats[:3]) if cats else ""
        else:
            cat_str = str(cats)
        meta[str(k)] = {"title": title, "brand": brand, "categories": cat_str}
    _meta_cache[dataset_name] = meta
    print(f"[元数据] {dataset_name}: 已加载 {len(meta)} 个物品名称")
    return meta


def item_info(dataset_name, item_id):
    """返回 {"title": ..., "brand": ..., "categories": ...}，不存在时返回空字段"""
    meta = load_item_meta(dataset_name)
    return meta.get(str(item_id), {"title": "", "brand": "", "categories": ""})


def get_user_seqs(data_file):
    """读取 .txt 文件，返回 user_seq 列表（每项为该用户的物品 ID 列表）"""
    user_seq = []
    with open(data_file, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ')
            items = [int(x) for x in parts[1:]]  # parts[0] 是用户 ID（原始），忽略
            user_seq.append(items)
    return user_seq


def load_dataset(name):
    """懒加载：首次调用时加载数据和模型，之后从缓存读取"""
    if name in _model_cache:
        return _model_cache[name], _seq_cache[name]

    cfg = DATASETS[name]
    print(f"[加载] {name} 数据集...")

    # 1. 读取用户序列
    user_seq = get_user_seqs(cfg["data_file"])
    _seq_cache[name] = user_seq

    # 2. 构造 args
    args = SimpleNamespace(
        item_size=cfg["item_size"],
        mask_id=cfg["item_size"] - 1,
        **COMMON_ARGS
    )

    # 3. 实例化模型并加载权重
    model = SASRecModel(args=args)
    state = torch.load(cfg["pt_file"], map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    _model_cache[name] = model

    print(f"[加载完成] {name}: {len(user_seq)} 用户, item_size={cfg['item_size']}")
    return model, user_seq


def compute_hit_users(dataset_name, topk=20, batch_size=512):
    """
    批量推理所有用户，返回测试集物品命中 Top-K 的用户索引列表。
    结果缓存到 data/{dataset}_hit_users.json，下次直接读取。
    """
    cache_file = os.path.join(BASE_DIR, "data", f"{dataset_name}_hit_users.json")
    if os.path.exists(cache_file):
        print(f"[缓存] 读取 {cache_file}")
        with open(cache_file, encoding='utf-8') as f:
            return json.load(f)

    print(f"[批量推理] {dataset_name}，计算命中用户（topk={topk}）…")
    model, user_seq = load_dataset(dataset_name)
    max_len = COMMON_ARGS["max_seq_length"]
    n_users = len(user_seq)
    hit_users = []

    # 预取物品嵌入矩阵（只取一次，节省时间）
    with torch.no_grad():
        item_emb = model.item_embeddings.weight.detach()  # [item_size, 64]

    for start in range(0, n_users, batch_size):
        end = min(start + batch_size, n_users)
        batch_seqs = user_seq[start:end]

        # 构造 padding 后的输入
        padded_batch = []
        for items in batch_seqs:
            inp = items[:-1][-max_len:]
            padded = [0] * (max_len - len(inp)) + inp
            padded_batch.append(padded)

        inp_tensor = torch.tensor(padded_batch, dtype=torch.long)

        with torch.no_grad():
            seq_out = model.transformer_encoder(inp_tensor)   # [B, 50, 64]
            seq_out = seq_out[:, -1, :]                        # [B, 64]
            scores_batch = torch.matmul(seq_out, item_emb.T).numpy()  # [B, item_size]

        for i, items in enumerate(batch_seqs):
            scores = scores_batch[i].copy()
            # 屏蔽训练/验证中见过的物品
            seen = set(items[:-1])
            for sid in seen:
                if 0 < sid < len(scores):
                    scores[sid] = -1e9
            scores[0] = -1e9  # padding token

            answer = items[-1]
            top_idx = np.argpartition(scores, -topk)[-topk:]
            if answer in top_idx:
                hit_users.append(start + i)

        print(f"  {end}/{n_users} ({100*end/n_users:.1f}%)  命中: {len(hit_users)}", end='\r')

    print(f"\n[完成] {dataset_name}: 共 {n_users} 用户，命中 {len(hit_users)} 个"
          f"（HIT@{topk}={len(hit_users)/n_users:.4f}）")

    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(hit_users, f)
    print(f"[缓存] 已保存到 {cache_file}")
    return hit_users


def recommend(dataset_name, user_idx, topk=20):
    """
    对指定用户生成 Top-K 推荐。
    返回 [(item_id, score), ...] 按 score 降序排列
    """
    model, user_seq = load_dataset(dataset_name)
    items = user_seq[user_idx]

    # 测试模式输入：items[:-1]，答案是 items[-1]
    input_ids = items[:-1]
    answer = items[-1]

    # 截取最后 max_seq_length 项，并在前面补 0（padding）
    max_len = COMMON_ARGS["max_seq_length"]
    input_ids = input_ids[-max_len:]
    padded = [0] * (max_len - len(input_ids)) + input_ids

    with torch.no_grad():
        inp = torch.tensor([padded], dtype=torch.long)    # [1, 50]
        seq_out = model.transformer_encoder(inp)           # [1, 50, 64]
        seq_out = seq_out[:, -1, :]                        # [1, 64]
        item_emb = model.item_embeddings.weight            # [item_size, 64]
        scores = torch.matmul(seq_out, item_emb.T)        # [1, item_size]
        scores = scores[0].numpy()                         # [item_size]

    # 屏蔽训练中见过的物品（items[:-1] 全部屏蔽）
    seen = set(items[:-1])
    for sid in seen:
        if 0 < sid < len(scores):
            scores[sid] = -1e9
    scores[0] = -1e9  # 屏蔽 padding token

    # 取 Top-K
    top_idx = np.argsort(scores)[::-1][:topk]
    results = [(int(i), float(scores[i])) for i in top_idx]

    return results, answer, items


# ── HTTP 请求处理器 ────────────────────────────────────────────────────
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")


class Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        print(f"[{self.command}] {self.path}")

    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path, mime="text/html; charset=utf-8"):
        with open(path, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path   = parsed.path.rstrip("/")
        qs     = parse_qs(parsed.query)

        try:
            # ── 静态前端 ──────────────────────────────────────────────
            if path in ("", "/", "/index.html"):
                self.send_file(os.path.join(FRONTEND_DIR, "index.html"))
                return

            # ── API: 数据集列表 ──────────────────────────────────────
            if path == "/api/datasets":
                result = []
                for name, cfg in DATASETS.items():
                    pt_exists = os.path.exists(cfg["pt_file"])
                    result.append({
                        "name": name,
                        "label": cfg["label"],
                        "item_size": cfg["item_size"],
                        "model_ready": pt_exists,
                    })
                self.send_json(result)
                return

            # ── API: 用户列表（分页）─────────────────────────────────
            if path.startswith("/api/users/"):
                name = path.split("/api/users/")[1]
                if name not in DATASETS:
                    self.send_json({"error": "dataset not found"}, 404); return
                _, user_seq = load_dataset(name)
                page      = int(qs.get("page", ["1"])[0])
                page_size = int(qs.get("page_size", ["50"])[0])
                total     = len(user_seq)
                start     = (page - 1) * page_size
                end       = min(start + page_size, total)
                users = []
                for i in range(start, end):
                    items = user_seq[i]
                    users.append({
                        "user_idx": i,
                        "seq_len": len(items),
                        "train_len": len(items) - 2,
                    })
                self.send_json({"total": total, "page": page,
                                "page_size": page_size, "users": users})
                return

            # ── API: 用户序列详情 ─────────────────────────────────────
            if path.startswith("/api/sequence/"):
                parts = path.split("/api/sequence/")[1].split("/")
                if len(parts) < 2:
                    self.send_json({"error": "bad request"}, 400); return
                name, user_idx = parts[0], int(parts[1])
                if name not in DATASETS:
                    self.send_json({"error": "dataset not found"}, 404); return
                _, user_seq = load_dataset(name)
                if user_idx >= len(user_seq):
                    self.send_json({"error": "user not found"}, 404); return
                items = user_seq[user_idx]
                # 为每个 item_id 附加名称信息
                def enrich(item_id):
                    info = item_info(name, item_id)
                    return {"item_id": item_id, **info}
                self.send_json({
                    "user_idx":    user_idx,
                    "total_len":   len(items),
                    "train_items": [enrich(i) for i in items[:-2]],
                    "valid_item":  enrich(items[-2]),
                    "test_item":   enrich(items[-1]),
                })
                return

            # ── API: 推荐结果 ─────────────────────────────────────────
            if path.startswith("/api/recommend/"):
                parts = path.split("/api/recommend/")[1].split("/")
                if len(parts) < 2:
                    self.send_json({"error": "bad request"}, 400); return
                name, user_idx = parts[0], int(parts[1])
                if name not in DATASETS:
                    self.send_json({"error": "dataset not found"}, 404); return
                _, user_seq = load_dataset(name)
                if user_idx >= len(user_seq):
                    self.send_json({"error": "user not found"}, 404); return

                recs, answer, items = recommend(name, user_idx, topk=20)

                # 归一化分数到 0~1（softmax 风格，只对 Top-20 做）
                raw_scores = [s for _, s in recs]
                min_s, max_s = min(raw_scores), max(raw_scores)
                span = max_s - min_s if max_s != min_s else 1.0

                answer_info = item_info(name, answer)
                self.send_json({
                    "user_idx": user_idx,
                    "answer":   answer,
                    "answer_title": answer_info.get("title", ""),
                    "recommendations": [
                        {
                            "rank":       idx + 1,
                            "item_id":    item_id,
                            "title":      item_info(name, item_id).get("title", ""),
                            "brand":      item_info(name, item_id).get("brand", ""),
                            "categories": item_info(name, item_id).get("categories", ""),
                            "score":      score,
                            "score_norm": round((score - min_s) / span, 4),
                            "is_answer":  item_id == answer,
                        }
                        for idx, (item_id, score) in enumerate(recs)
                    ],
                })
                return

            # ── API: 命中用户列表（批量推理，带缓存）────────────────────
            if path.startswith("/api/hit_users/"):
                name = path.split("/api/hit_users/")[1]
                if name not in DATASETS:
                    self.send_json({"error": "dataset not found"}, 404); return
                _, user_seq = load_dataset(name)
                hit = compute_hit_users(name, topk=20)
                # 返回命中用户的详细信息（含序列长度）
                users_info = []
                for idx in hit:
                    items = user_seq[idx]
                    users_info.append({
                        "user_idx": idx,
                        "seq_len": len(items),
                        "train_len": len(items) - 2,
                    })
                self.send_json({
                    "total_users": len(user_seq),
                    "hit_count": len(hit),
                    "hit_rate": round(len(hit) / len(user_seq), 4),
                    "users": users_info,
                })
                return

            self.send_json({"error": "not found"}, 404)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.send_json({"error": str(e)}, 500)


# ── 启动服务器 ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    PORT = 8000
    os.makedirs(FRONTEND_DIR, exist_ok=True)
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"CoSeRec 后端已启动：http://localhost:{PORT}")
    print("按 Ctrl+C 停止")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务器已停止")
