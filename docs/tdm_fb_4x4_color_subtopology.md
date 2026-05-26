# 4×4 Mesh 上 4-ary, 2-flat Flattened Butterfly 的 Color 子拓扑分解

> **着色策略**：`ilp_min_C`（[`color_planners.py`](../wsesim/network/color_planners.py) · OR-Tools CP-SAT）  
> **目标**：在 4×4 物理 2D Mesh 上，通过 **4 个 Color** 时分复用实现完整 FB 逻辑拓扑。

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

### 1.2 逻辑拓扑：4-ary, 2-flat (k=4, n=2)

- 有向逻辑链路：**96**
- 逻辑直径：**≤ 2 跳**
- 每维邻居数：**3**

---

## 2. Color 分解（ILP min C）

| 指标 | 值 |
|------|-----|
| TDM Color 数 C | **4** |
| 物理边负载下界 | 4 |
| 负载均衡 max/min | 2.75 |
| 每 Color 链路数范围 | 16 – 44 |

### 各 Color 链路分配

| Color | 有向链路数 |
|-------|----------|
| C0 | 44 |
| C1 | 17 |
| C2 | 16 |
| C3 | 19 |
| **合计** | **96** |

### 物理跳数分布

| 物理跳数 | 链路数 |
|---------|--------|
| 1 | 48 |
| 2 | 32 |
| 3 | 16 |

---

## 3. 时分调度

```
周期 C = 4：时隙 t 激活 Color (t mod 4)
```

---

## 4. 路由示例

| 通信 | 维度序路由 | Color |
|------|-----------|-------|
| `0 → 9` | 0→1(C1) → 1→9(C3) | 跨 2 Color |
| `0 → 15` | 0→3(C2) → 3→15(C2) | 单 C2 |
| `0 → 8` | 0→8(C3) | 单 C3 |
| `7 → 8` | 7→4(C2) → 4→8(C0) | 跨 2 Color |

---

## 5. 说明

4×4 mesh · kⁿ=16 · C=lb=4

---

## 6. 可视化

交互式 Color 时分图：`tdm_fb_4x4_color_subtopology.html`

---

*生成：`scripts/generate_tdm_fb_hypercube_color_docs.py` · planner=ilp_min_C*
