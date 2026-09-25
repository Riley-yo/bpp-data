"""BPP 5种formulation的COPT求解脚本。

对每个BPP实例，用COPT跑5种建模（Compact/DP-flow/Arc-flow/Set-cover/Set-partition），
记录求解时间、是否最优、目标值，生成 performance.csv 和 good.csv。

依据：你给的5种BPP formulation数学公式 + 建模与ISA说明.md第5节。

用法（服务器上）：
  python solve_bpp_formulations.py --instances BPP_原始实例.jsonl --output-dir bpp_perf --time-limit 60
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any


# === 实例加载 ===

def load_instances(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# === Formulation 1: Compact（物品到箱指派） ===
# x_ij=物品i是否装箱j, y_j=是否开箱j
# min sum(y_j)
# s.t. sum_j(x_ij)=1 ∀i; sum_i(wi*xi_j) <= W*y_j ∀j

def solve_compact(sizes, capacity, coptpy, time_limit):
    n = len(sizes)
    env = coptpy.Envr()
    model = env.createModel("bpp_compact")
    model.param.TimeLimit = float(time_limit)

    x, y = {}, {}
    for i in range(n):
        for j in range(n):
            x[i, j] = model.addVar(vtype=coptpy.COPT.BINARY, name=f"x_{i}_{j}")
    for j in range(n):
        y[j] = model.addVar(vtype=coptpy.COPT.BINARY, obj=1.0, name=f"y_{j}")

    for i in range(n):
        expr = coptpy.LinExpr([(x[i, j], 1.0) for j in range(n)])
        model.addConstr(expr, coptpy.COPT.EQUAL, 1.0, name=f"assign_{i}")
    for j in range(n):
        expr = coptpy.LinExpr([(x[i, j], float(sizes[i])) for i in range(n)])
        model.addConstr(expr, coptpy.COPT.LESS_EQUAL, float(capacity) * y[j], name=f"cap_{j}")

    model.solve()
    status = model.status
    is_optimal = (status == coptpy.COPT.OPTIMAL)
    obj = float(model.ObjVal) if is_optimal else -1
    return is_optimal, obj, status


# === Formulation 2: DP-flow（物品层与容量状态） ===
# 网络流：节点 (j, d)，j=物品层，d=容量状态
# 弧：不装物品 j 的弧 ((j-1,d)->(j,d)) 和装物品 j 的弧 ((j-1,d)->(j,d+w_j))
# 源点 o=(0,0)，汇点 t
# min z (z=使用的箱数)
# 流守恒 + 每物品恰走一条label=j的弧

def solve_dp_flow(sizes, capacity, coptpy, time_limit):
    n = len(sizes)
    # 构建节点和弧
    # 节点编号: (j, d) -> j*(capacity+1) + d
    def node_id(j, d):
        return j * (capacity + 1) + d

    # 源点 = (0, 0), 汇点 = n*(capacity+1)
    source = node_id(0, 0)
    sink = n * (capacity + 1)

    # 收集所有弧
    arcs = []  # (from_node, to_node, label)  label=j 表示装了物品j, label=0表示不装
    for j in range(1, n + 1):
        w = sizes[j - 1]
        for d in range(capacity + 1):
            # 不装物品j
            arcs.append((node_id(j - 1, d), node_id(j, d), 0))
            # 装物品j
            if d + w <= capacity:
                arcs.append((node_id(j - 1, d), node_id(j, d + w), j))

    # 汇点弧：(n, d) -> sink
    for d in range(capacity + 1):
        arcs.append((node_id(n, d), sink, 0))

    num_nodes = (n + 1) * (capacity + 1) + 1

    env = coptpy.Envr()
    model = env.createModel("bpp_dp_flow")
    model.param.TimeLimit = float(time_limit)

    # 变量：每条弧的整数流 f_a，以及箱数 z
    f = {}
    for idx, (u, v, label) in enumerate(arcs):
        f[idx] = model.addVar(vtype=coptpy.COPT.INTEGER, lb=0, name=f"f_{idx}")
    z = model.addVar(vtype=coptpy.COPT.INTEGER, lb=0, obj=1.0, name="z")

    # 出入弧索引
    out_arcs = {i: [] for i in range(num_nodes)}
    in_arcs = {i: [] for i in range(num_nodes)}
    for idx, (u, v, label) in enumerate(arcs):
        out_arcs[u].append(idx)
        in_arcs[v].append(idx)

    # 流守恒约束
    for v_node in range(num_nodes):
        if v_node == source:
            model.addConstr(
                coptpy.LinExpr([(f[i], 1.0) for i in out_arcs[v_node]])
                - coptpy.LinExpr([(f[i], 1.0) for i in in_arcs[v_node]]),
                coptpy.COPT.EQUAL, z, name=f"flow_{v_node}")
        elif v_node == sink:
            model.addConstr(
                coptpy.LinExpr([(f[i], 1.0) for i in in_arcs[v_node]])
                - coptpy.LinExpr([(f[i], 1.0) for i in out_arcs[v_node]]),
                coptpy.COPT.EQUAL, z, name=f"flow_{v_node}")
        else:
            model.addConstr(
                coptpy.LinExpr([(f[i], 1.0) for i in out_arcs[v_node]])
                - coptpy.LinExpr([(f[i], 1.0) for i in in_arcs[v_node]]),
                coptpy.COPT.EQUAL, 0.0, name=f"flow_{v_node}")

    # 每个物品恰走一条label=j的弧
    for j in range(1, n + 1):
        expr = coptpy.LinExpr([(f[idx], 1.0) for idx, (u, v, label) in enumerate(arcs) if label == j])
        model.addConstr(expr, coptpy.COPT.EQUAL, 1.0, name=f"item_{j}")

    model.solve()
    status = model.status
    is_optimal = (status == coptpy.COPT.OPTIMAL)
    obj = float(model.ObjVal) if is_optimal else -1
    return is_optimal, obj, status


# === Formulation 3: Arc-flow（容量状态） ===
# 简化版：基于容量的弧流网络
# 节点 0..capacity，弧 (d1 -> d1+w_i) 表示装物品i
# 超源汇，每条路径=一个箱的装箱方案
# min z (路径数=箱数)

def solve_arc_flow(sizes, capacity, coptpy, time_limit):
    n = len(sizes)
    # 节点 0..capacity
    # 弧：(d -> d+w) 表示在容量d处装入大小w的物品
    # 但需要区分不同物品，用多组弧

    # 收集弧：对每个物品i，在容量d处可以装（d+wi<=capacity）
    arcs = []  # (from, to, item_idx)
    for i in range(n):
        w = sizes[i]
        for d in range(capacity - w + 1):
            arcs.append((d, d + w, i))
    # 添加从capacity回到0的弧（表示一个箱装完了，开始新箱）
    # 以及从0到sink的弧

    num_nodes = capacity + 2  # 0..capacity + sink
    sink = capacity + 1

    env = coptpy.Envr()
    model = env.createModel("bpp_arc_flow")
    model.param.TimeLimit = float(time_limit)

    # 变量
    f = {}
    for idx, (u, v, item) in enumerate(arcs):
        f[idx] = model.addVar(vtype=coptpy.COPT.INTEGER, lb=0, name=f"f_{idx}")
    # 回环弧：capacity -> 0
    loop_var = model.addVar(vtype=coptpy.COPT.INTEGER, lb=0, obj=1.0, name="loop")
    # 汇弧：0 -> sink
    sink_var = model.addVar(vtype=coptpy.COPT.INTEGER, lb=0, name="sink_flow")

    # 出入弧索引
    out_arcs = {i: [] for i in range(num_nodes)}
    in_arcs = {i: [] for i in range(num_nodes)}
    for idx, (u, v, item) in enumerate(arcs):
        out_arcs[u].append(idx)
        in_arcs[v].append(idx)
    # 回环弧
    out_arcs[capacity].append(-1)  # -1表示loop_var
    in_arcs[0].append(-1)
    # 汇弧
    out_arcs[0].append(-2)  # -2表示sink_var
    in_arcs[sink].append(-2)

    def get_var(idx):
        if idx == -1:
            return loop_var
        elif idx == -2:
            return sink_var
        return f[idx]

    # 流守恒
    for v_node in range(num_nodes):
        out_expr = coptpy.LinExpr([(get_var(i), 1.0) for i in out_arcs[v_node]]) if out_arcs[v_node] else None
        in_expr = coptpy.LinExpr([(get_var(i), 1.0) for i in in_arcs[v_node]]) if in_arcs[v_node] else None
        if out_expr and in_expr:
            model.addConstr(out_expr - in_expr, coptpy.COPT.EQUAL, 0.0, name=f"flow_{v_node}")
        elif out_expr:
            model.addConstr(out_expr, coptpy.COPT.EQUAL, 0.0, name=f"flow_{v_node}")
        elif in_expr:
            model.addConstr(in_expr, coptpy.COPT.EQUAL, 0.0, name=f"flow_{v_node}")

    # 每个物品恰装一次
    for i in range(n):
        expr = coptpy.LinExpr([(f[idx], 1.0) for idx, (u, v, item) in enumerate(arcs) if item == i])
        model.addConstr(expr, coptpy.COPT.EQUAL, 1.0, name=f"item_{i}")

    model.solve()
    status = model.status
    is_optimal = (status == coptpy.COPT.OPTIMAL)
    obj = float(model.ObjVal) if is_optimal else -1
    return is_optimal, obj, status


# === Formulation 4: Set-cover（完整装箱模式） ===
# 枚举所有合法装箱模式（子集和<=capacity），用集合覆盖
# min sum(x_p) (选的模式数=箱数)
# s.t. 每个物品至少被覆盖一次

def solve_set_cover(sizes, capacity, coptpy, time_limit):
    n = len(sizes)
    # 枚举所有合法模式（子集和<=capacity）
    # 对n>20的实例，模式数会爆炸，限制最多枚举10000个模式
    patterns = []
    max_patterns = 10000
    # 用DFS枚举
    def dfs(start, current_pattern, current_load):
        if len(patterns) >= max_patterns:
            return
        if current_pattern:
            patterns.append(list(current_pattern))
        for i in range(start, n):
            if current_load + sizes[i] <= capacity:
                current_pattern.append(i)
                dfs(i + 1, current_pattern, current_load + sizes[i])
                current_pattern.pop()
    dfs(0, [], 0)

    if not patterns:
        return False, -1, -1

    env = coptpy.Envr()
    model = env.createModel("bpp_set_cover")
    model.param.TimeLimit = float(time_limit)

    # 变量：每个模式是否选用
    x = {}
    for p_idx in range(len(patterns)):
        x[p_idx] = model.addVar(vtype=coptpy.COPT.BINARY, obj=1.0, name=f"x_{p_idx}")

    # 约束：每个物品至少被覆盖一次
    for i in range(n):
        covering = [p_idx for p_idx, pattern in enumerate(patterns) if i in pattern]
        if covering:
            expr = coptpy.LinExpr([(x[p_idx], 1.0) for p_idx in covering])
            model.addConstr(expr, coptpy.COPT.GREATER_EQUAL, 1.0, name=f"cover_{i}")

    model.solve()
    status = model.status
    is_optimal = (status == coptpy.COPT.OPTIMAL)
    obj = float(model.ObjVal) if is_optimal else -1
    return is_optimal, obj, status


# === Formulation 5: Set-partition（规格模式，恰好覆盖一次） ===
# 类似set-cover但用等号（每个物品恰好被覆盖一次）

def solve_set_partition(sizes, capacity, coptpy, time_limit):
    n = len(sizes)
    patterns = []
    max_patterns = 10000
    def dfs(start, current_pattern, current_load):
        if len(patterns) >= max_patterns:
            return
        if current_pattern:
            patterns.append(list(current_pattern))
        for i in range(start, n):
            if current_load + sizes[i] <= capacity:
                current_pattern.append(i)
                dfs(i + 1, current_pattern, current_load + sizes[i])
                current_pattern.pop()
    dfs(0, [], 0)

    if not patterns:
        return False, -1, -1

    env = coptpy.Envr()
    model = env.createModel("bpp_set_partition")
    model.param.TimeLimit = float(time_limit)

    x = {}
    for p_idx in range(len(patterns)):
        x[p_idx] = model.addVar(vtype=coptpy.COPT.BINARY, obj=1.0, name=f"x_{p_idx}")

    # 每个物品恰好被覆盖一次（等号）
    for i in range(n):
        covering = [p_idx for p_idx, pattern in enumerate(patterns) if i in pattern]
        if covering:
            expr = coptpy.LinExpr([(x[p_idx], 1.0) for p_idx in covering])
            model.addConstr(expr, coptpy.COPT.EQUAL, 1.0, name=f"part_{i}")

    model.solve()
    status = model.status
    is_optimal = (status == coptpy.COPT.OPTIMAL)
    obj = float(model.ObjVal) if is_optimal else -1
    return is_optimal, obj, status


# === 主函数 ===

def run_all_formulations(instances, output_dir, time_limit):
    try:
        import coptpy
    except ImportError:
        raise RuntimeError("COPT Python API (coptpy) is unavailable. Run: pip install coptpy")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    formulations = ["compact", "dp_flow", "arc_flow", "set_cover", "set_partition"]

    # performance.csv: 每行一个实例，列为5种formulation的求解时间(秒)
    perf_rows = []
    # good.csv: 每行一个实例，列为5种formulation的0/1标签(1=最优且在时限内)
    good_rows = []
    # objectives.csv: 每行的目标值
    obj_rows = []

    total = len(instances)
    for idx, inst in enumerate(instances):
        params = inst["parameters"]
        sizes = params["sizes"]
        capacity = params["capacity"]
        inst_id = inst["instance_id"]

        # 跳过实例太大无法枚举模式的
        n = len(sizes)

        perf_row = {"instance_id": inst_id}
        good_row = {"instance_id": inst_id}
        obj_row = {"instance_id": inst_id}

        for form_name in formulations:
            try:
                start_time = time.time()
                if form_name == "compact":
                    is_opt, obj, status = solve_compact(sizes, capacity, coptpy, time_limit)
                elif form_name == "dp_flow":
                    # DP-flow 对大实例节点数爆炸，跳过n>20
                    if n > 20:
                        perf_row[form_name] = ""
                        good_row[form_name] = ""
                        obj_row[form_name] = ""
                        continue
                    is_opt, obj, status = solve_dp_flow(sizes, capacity, coptpy, time_limit)
                elif form_name == "arc_flow":
                    is_opt, obj, status = solve_arc_flow(sizes, capacity, coptpy, time_limit)
                elif form_name == "set_cover":
                    is_opt, obj, status = solve_set_cover(sizes, capacity, coptpy, time_limit)
                elif form_name == "set_partition":
                    is_opt, obj, status = solve_set_partition(sizes, capacity, coptpy, time_limit)
                elapsed = time.time() - start_time

                perf_row[form_name] = f"{elapsed:.6f}"
                good_row[form_name] = "1" if is_opt else "0"
                obj_row[form_name] = str(obj) if obj > 0 else ""
            except Exception as e:
                perf_row[form_name] = ""
                good_row[form_name] = ""
                obj_row[form_name] = ""
                print(f"  {inst_id} / {form_name}: ERROR {e}")

        perf_rows.append(perf_row)
        good_rows.append(good_row)
        obj_rows.append(obj_row)

        if (idx + 1) % 50 == 0 or idx == 0 or idx == total - 1:
            print(f"  [{idx+1}/{total}] {inst_id}: compact={perf_row.get('compact','')}, dp_flow={perf_row.get('dp_flow','')}, arc_flow={perf_row.get('arc_flow','')}, set_cover={perf_row.get('set_cover','')}, set_partition={perf_row.get('set_partition','')}")

    # 写CSV
    perf_path = output_dir / "performance.csv"
    with perf_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["instance_id"] + formulations)
        writer.writeheader()
        writer.writerows(perf_rows)

    good_path = output_dir / "good.csv"
    with good_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["instance_id"] + formulations)
        writer.writeheader()
        writer.writerows(good_rows)

    obj_path = output_dir / "objectives.csv"
    with obj_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["instance_id"] + formulations)
        writer.writeheader()
        writer.writerows(obj_rows)

    print(f"\n保存 {len(perf_rows)} 个实例的求解结果到 {output_dir}")
    print(f"  performance.csv: 求解时间(秒)")
    print(f"  good.csv: 最优性标签(0/1)")
    print(f"  objectives.csv: 目标值(最少箱数)")

    # 统计
    for form_name in formulations:
        opt_count = sum(1 for row in good_rows if row.get(form_name) == "1")
        has_data = sum(1 for row in good_rows if row.get(form_name) != "")
        print(f"  {form_name}: {opt_count}/{has_data} 最优 ({100*opt_count/max(has_data,1):.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BPP 5种formulation COPT求解")
    parser.add_argument("--instances", type=str, required=True, help="BPP原始实例JSONL")
    parser.add_argument("--output-dir", type=str, required=True, help="输出目录")
    parser.add_argument("--time-limit", type=float, default=60.0, help="每个formulation求解时限(秒)")
    parser.add_argument("--max-instances", type=int, default=None, help="最多处理多少个实例(调试用)")
    args = parser.parse_args()

    instances = load_instances(args.instances)
    if args.max_instances:
        instances = instances[:args.max_instances]
    print(f"加载 {len(instances)} 个BPP实例, 时限={args.time_limit}秒")
    run_all_formulations(instances, args.output_dir, args.time_limit)
