# 4×4 Mesh 上 2-ary, 4-flat Flattened Butterfly 的 Color 子拓扑分解

> **着色策略**：`ilp_min_C`（[`color_planners.py`](../wsesim/network/color_planners.py) · OR-Tools CP-SAT）  
> **目标**：在 4×4 物理 2D Mesh 上，通过 **2 个 Color** 时分复用实现完整 FB 逻辑拓扑。

---

## 1. 基本定义

### 1.1 节点编号（行优先 4×4）

```
 0  1  2  3   ← row 0
 4  5  6  7   ← row 1
 8  9 10 11   ← row 2
12 13 14 15   ← row 3
```

`node = row × 4 + col`

### 1.2 逻辑拓扑：2-ary, 4-flat (k=2, n=4)

- 有向逻辑链路：**64**
- 逻辑直径：**≤ 4 跳**
- 每维邻居数：**1**

---

## 2. Color 分解（ILP min C）

| 指标 | 值 |
|------|-----|
| TDM Color 数 C | **2** |
| 物理边负载下界 | 2 |
| 负载均衡 max/min | 1.00 |
| 每 Color 链路数范围 | 32 – 32 |

### 各 Color 链路分配

| Color | 有向链路数 |
|-------|----------|
| C0 | 32 |
| C1 | 32 |
| **合计** | **64** |

### 物理跳数分布

| 物理跳数 | 链路数 |
|---------|--------|
| 1 | 32 |
| 2 | 32 |

---

## 3. 时分调度

```
周期 C = 2：时隙 t 激活 Color (t mod 2)
```

---

## 4. 路由示例

| 通信 | 维度序路由 | Color |
|------|-----------|-------|
| `0 → 5` | 0→1(C1) → 1→5(C1) | 单 C1 |
| `0 → 6` | 0→2(C0) → 2→6(C1) | 跨 2 Color |
| `0 → 15` | 0→1(C1) → 1→3(C1) → 3→7(C1) → 7→15(C1) | 单 C1 |
| `3 → 12` | 3→2(C0) → 2→0(C0) → 0→4(C1) → 4→12(C1) | 跨 2 Color |

---

## 5. 说明

4×4 mesh · kⁿ=16 · C=lb=2

---

## 6. 可视化

交互式 Color 时分图：`tdm_fb_k2_n4_color_subtopology.html`

---

*生成：`scripts/generate_tdm_fb_hypercube_color_docs.py` · planner=ilp_min_C*
