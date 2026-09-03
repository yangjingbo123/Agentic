"""v2 训练启动前自检（CPU，10 秒内完成）。

检查会静默毁掉训练的配置陷阱：
1. SFT checkpoint 目录冲突：load_trainable_models 优先读 {sft}/{role}/{role}/，
   若旧 v1 目录残留，新训的扁平 SFT 会被静默忽略（同 Fix 9 类失败）
2. SFT 数据格式：必须是 v2（decision/request），v1 格式会让 controller 永不 stop
3. max_len 截断率：截断砍掉的正是「最终答案：」/「分数:」——唯一计入 loss 的字段
4. 数据 GT 对账、评测集泄漏

用法：
    python preflight_v2.py                        # 检查 SFT 数据与配置
    python preflight_v2.py --sft-ckpt checkpoints/sft_v2   # 额外检查 checkpoint 结构
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OK, WARN, FAIL = "  [ok]", "  [warn]", "  [FAIL]"
_fails, _warns = [], []


def check(cond, msg, fatal=True):
    if cond:
        print(f"{OK} {msg}")
    elif fatal:
        print(f"{FAIL} {msg}")
        _fails.append(msg)
    else:
        print(f"{WARN} {msg}")
        _warns.append(msg)
    return cond


def check_sft_data(path="data/sft_train_v2.jsonl", max_len=1024, char_per_tok=2.2):
    print(f"\n[1] SFT 数据格式与长度 ({path})")
    if not os.path.isfile(path):
        check(False, f"{path} 不存在 —— 先跑 python data/convert_sft_v2.py")
        return
    from agents.parsing import parse_decision, parse_interaction

    rows = [json.loads(l) for l in open(path)]
    n_ctrl = n_ctrl_ok = n_work = n_work_ok = 0
    n_stop = 0
    v1_markers = 0
    lens = []
    for ep in rows:
        for t in ep.get("turns", []):
            body = t["system"] + t["user"] + t["response"]
            lens.append(len(body))
            if "strategy:" in t["response"] or "request_critic" in t["response"] \
               or "support:" in t["response"]:
                v1_markers += 1
            if t.get("role_name") == "controller":
                n_ctrl += 1
                d = parse_decision(t["response"])
                n_ctrl_ok += d in ("continue", "stop")
                n_stop += d == "stop"
            else:
                n_work += 1
                n_work_ok += parse_interaction(t["response"])[0] in \
                    ("none", "request", "challenge")

    check(v1_markers == 0,
          f"无 v1 残留标记（strategy:/request_critic/support:），实测 {v1_markers} 处")
    check(n_ctrl_ok == n_ctrl, f"controller decision 可解析 {n_ctrl_ok}/{n_ctrl}")
    check(n_work_ok == n_work, f"worker interaction 可解析 {n_work_ok}/{n_work}")
    check(n_stop > 0, f"含 stop 样本 {n_stop} 条（否则 controller 学不会终止 → stop_rate≈0）")

    over = sum(1 for l in lens if l / char_per_tok > max_len)
    rate = over / max(len(lens), 1)
    check(rate < 0.02,
          f"max_len={max_len} 预计截断率 {rate:.1%}（>2% 会伤 parse_rate，"
          f"截断砍掉的正是「最终答案：」）", fatal=rate >= 0.02)
    print(f"       样本数 {len(rows)} episodes / {len(lens)} turns")


def check_sft_ckpt(path):
    print(f"\n[2] SFT checkpoint 结构 ({path})")
    if not os.path.isdir(path):
        check(False, f"{path} 不存在（SFT 还没跑完就先跳过本项）", fatal=False)
        return
    roles = ("proposer", "controller", "critic", "verifier")
    # 复刻 load_trainable_models 的查找优先级
    stale = []
    for r in roles:
        nested = os.path.join(path, r, r, "adapter_model.safetensors")
        flat = os.path.join(path, r, "adapter_model.safetensors")
        if os.path.isfile(nested) or os.path.isfile(flat):
            stale.append(r)
    top = os.path.join(path, "adapter_model.safetensors")

    if stale and os.path.isfile(top):
        check(False,
              f"目录冲突！存在 role 子目录 {stale} 又存在顶层 adapter_model.safetensors；"
              f"load 会优先读 role 子目录 → 新训的 SFT 被静默忽略。"
              f"请把 SFT 输出到干净的新目录（+sft.save_dir=...）")
    elif stale:
        check(True, f"按角色分开的 SFT（role 子目录：{stale}）—— 确认这是你想要的版本")
    elif os.path.isfile(top):
        check(True, "扁平单 adapter SFT（四个角色共享同一起点，train_sft.py 的产出）")
    else:
        check(False, f"{path} 下找不到任何 adapter_model.safetensors")


def check_rl_data():
    print("\n[3] RL 数据与评测集")
    from agents.grader import math_equal
    train_p, test_p = "data/math_train_rl.jsonl", "data/math_test.jsonl"
    for p in (train_p, test_p):
        if not check(os.path.isfile(p), f"{p} 存在"):
            return
    train = [json.loads(l) for l in open(train_p)]
    test = [json.loads(l) for l in open(test_p)]
    check(all(r.get("answer") for r in train), "训练集无空答案")
    lvl5 = [r for r in test if r.get("level") == "Level 5"]
    check(len(lvl5) >= 300, f"测试集 Level 5 样本 {len(lvl5)} 条（eval 取前 300）")
    # 泄漏检查
    tr_q = {r["question"] for r in train}
    overlap = sum(1 for r in test if r["question"] in tr_q)
    check(overlap == 0, f"train/test 零重叠（实测重叠 {overlap} 条）")
    # SFT 泄漏
    if os.path.isfile("data/sft_train_v2.jsonl"):
        sft_q = {json.loads(l)["question"] for l in open("data/sft_train_v2.jsonl")}
        leak_rl = len(sft_q & tr_q)
        leak_test = sum(1 for r in test if r["question"] in sft_q)
        check(leak_test == 0, f"SFT/test 零重叠（实测 {leak_test} 条）")
        check(leak_rl == 0, f"SFT/RL-train 零重叠（实测 {leak_rl} 条）", fatal=False)
    # grader 自反性抽样
    bad = [r["answer"] for r in test[:500] if not math_equal(r["answer"], r["answer"])]
    check(not bad, f"grader 自反性抽样通过（异常 {len(bad)} 条）")


def _load_flat_yaml(path: str) -> dict:
    """极简平链 yaml 解析（只处理 key: value），避免强依赖 PyYAML。"""
    out = {}
    for line in open(path):
        line = line.split("#")[0].strip()
        if not line or ":" not in line or line.startswith("-"):
            continue
        k, v = line.split(":", 1)
        v = v.strip().strip('"').strip("'")
        if v in ("true", "True"):
            out[k.strip()] = True
        elif v in ("false", "False"):
            out[k.strip()] = False
        elif v in ("null", ""):
            out[k.strip()] = None
        else:
            try:
                out[k.strip()] = int(v)
            except ValueError:
                try:
                    out[k.strip()] = float(v)
                except ValueError:
                    out[k.strip()] = v
    return out


def check_config():
    print("\n[4] v2 配置一致性")
    cfg = _load_flat_yaml("configs/agentic/default.yaml")
    dat = _load_flat_yaml("configs/data/math.yaml")
    check(str(dat.get("sft_path", "")).endswith("sft_train_v2.jsonl"),
          f"data.sft_path 指向 v2 数据（当前 {dat.get('sft_path')}）")
    for k in ("c_int", "int_miss", "lambda_int", "token_credit",
              "max_hops", "stop_gate",
              "eps_force_init", "eps_force_min"):
        check(k in cfg, f"v2 超参 {k} = {cfg.get(k)}")
    check(isinstance(cfg.get("token_credit"), bool),
          f"token_credit 必须是布尔值（当前 {cfg.get('token_credit')!r}）")
    try:
        _lambda_int = float(cfg.get("lambda_int"))
        check(_lambda_int >= 0.0,
              f"lambda_int 必须 >= 0（当前 {_lambda_int}）")
    except (TypeError, ValueError):
        check(False, f"lambda_int 必须是数值（当前 {cfg.get('lambda_int')!r}）")
    # 消融组合合法性
    if cfg.get("max_hops", 0) == 0 and cfg.get("stop_gate"):
        check(False, "max_hops=0（禁交互）时 stop_gate 必须为 false，"
                     "否则永远拿不到 verifier 分数 → stop 闸门死锁", fatal=True)
    # 资源配置
    nw = cfg.get("vllm_num_workers", 1) or 1
    slot = cfg.get("vllm_gpu_slot", 1) or 1
    print(f"       vllm_num_workers={nw}（训练模型占 cuda:0，vLLM 占 "
          f"cuda:{slot}..{slot + nw - 1}） → 需要 {1 + nw} 张卡；"
          f"不足时命令行覆盖 agentic.vllm_num_workers=N")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-ckpt", default="checkpoints/sft_v2")
    ap.add_argument("--max-len", type=int, default=1024)
    args = ap.parse_args()

    print("=" * 68)
    print("RACA v2 启动前自检")
    print("=" * 68)
    check_sft_data(max_len=args.max_len)
    check_sft_ckpt(args.sft_ckpt)
    check_rl_data()
    check_config()

    print("\n" + "=" * 68)
    if _fails:
        print(f"✗ {len(_fails)} 项致命问题，修复后再启动：")
        for m in _fails:
            print(f"   - {m}")
        sys.exit(1)
    print(f"✓ 全部通过（{len(_warns)} 项提醒）")
    for m in _warns:
        print(f"   ! {m}")


if __name__ == "__main__":
    main()
