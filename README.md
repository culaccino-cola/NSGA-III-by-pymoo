# 多目标作业车间调度问题（JSSP）求解方案
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)

## 项目简介
本项目基于 **NSGA-Ⅲ（非支配排序遗传算法Ⅲ）** 实现多目标作业车间调度问题（JSSP）的优化求解，针对调整后的数据文件 `machines_jobs.xls` 中的作业与机器配置，同时优化 **最大完工时间（Makespan）** 和 **总流程时间（Total Flow Time）** 两大核心目标，并满足每台机器至少分配 5 个作业的约束条件。


## 核心特性
✅ **多目标优化**：同时优化 Makespan 和 Total Flow Time，输出 Pareto 最优解集  
✅ **多种改进策略**：支持编码优化、遗传算子改进、模拟退火局部搜索等组合方案  
✅ **可视化输出**：自动生成甘特图（作业调度时间线）、Pareto 前沿对比图  
✅ **约束满足**：初始化阶段确保每台机器至少分配 5 个作业，避免无效解  
✅ **实验对比**：内置 8 组对比实验，一键运行并输出量化结果  

## 环境依赖
```bash
# 推荐使用 Anaconda 或虚拟环境安装
pip install numpy pandas matplotlib pymoo
```
numpy==1.26.4
pandas==2.2.2
matplotlib==3.8.4
pymoo==0.6.1.5

## 项目结构
```
JSSP-Multi-Objective-Optimization/
├── GA.py                # 核心算法代码（NSGA-Ⅲ实现+改进策略）
├── machines_jobs.xls    # 调整后调度数据（作业-机器处理时间矩阵）
├── baseline_gantt.png  # baseline Gantt
├── G_gantt.png  # 策略G Gantt
├── LICENSE              # MIT许可证
└── README.md            # 项目说明文档
```

## 调整后数据说明
### 数据结构（基于 `machines_jobs.xls`）
文件包含作业与机器的对应处理时间，经解析确认数据格式如下：
- **行维度**：作业编号（如 J001-J050，保持原数量规模）
- **列维度**：机器编号（如 M1-M3，保持原机器数量）
- **核心数据**：每个单元格代表对应作业在对应机器上的标准处理时间（单位：分钟/小时，与原数据一致）

### 数据示例（部分）
| 作业   | 机器1（M1） | 机器2（M2） | 机器3（M3） |
|--------|-------------|-------------|-------------|
| J001   | 57          | 75          | 10          |
| J002   | 37          | 78          | 52          |
| ...    | ...         | ...         | ...         |
| J050   | 92          | 86          | 62          |

### 约束条件（保持不变）
- 每台机器至少分配 5 个作业
- 每个作业仅在一台机器上执行，不可拆分
- 机器同一时间只能处理一个作业（无并行执行）

## 快速开始（适配新数据文件）
1. **克隆仓库**
```bash
git clone https://github.com/culaccino-cola/NSGA-III-by-pymoo.git
cd NSGA-III-by-pymoo
```

2. **修改文件路径（关键步骤）**
打开 `GA.py`将路径替换为新数据文件 `machines_jobs.xls` 的本地路径：
```python
# 替换为本地路径
path = r"你的文件路径/machines_jobs.xls"
# 示例：Linux/Mac 路径
# path = r"/home/username/Documents/machines_jobs.xls"
# 示例：Windows 路径
# path = r"D:\JSSP\machines_jobs.xls"
```

3. **运行算法**
```bash
python GA.py
```

4. **查看结果**
- 控制台输出：每组实验的最优 Makespan、Total Flow Time、运行时间，以及详细调度方案
- 可视化图表：自动弹出每组实验的甘特图和Pareto 前沿图

## 实验设计（保持不变）
项目内置 8 组对比实验，覆盖不同改进策略的组合，可直接对比效果：

| 实验名称               | 编码策略 | 改进遗传算子 | 局部搜索（模拟退火） |
|------------------------|----------|--------------|----------------------|
| 基线算法               | 默认编码 | ❌            | ❌                    |
| A-编码策略             | 反向编码 | ❌            | ❌                    |
| B-遗传算子             | 默认编码 | ✅            | ❌                    |
| C-局部搜索             | 默认编码 | ❌            | ✅                    |
| D-编码策略+遗传算子    | 反向编码 | ✅            | ❌                    |
| E-编码策略+局部搜索    | 反向编码 | ❌            | ✅                    |
| F-遗传算子+局部搜索    | 默认编码 | ✅            | ✅                    |
| G-组合策略             | 反向编码 | ✅            | ✅                    |

### 关键参数可配置
在 `GA.py` 中修改以下参数，适配不同场景：
```python
# 问题参数（与新数据文件规模匹配）
num_jobs = 50          # 作业数量（与 machines_jobs.xls 一致）
num_machines = 3       # 机器数量（与 machines_jobs.xls 一致）
min_jobs_per_machine = 5  # 每台机器最少作业数

# 算法参数
pop_size = 200         # 种群规模
n_offsprings = 100     # 每代后代数量
n_gen = 500            # 迭代次数

# 模拟退火参数（局部搜索）
sa_max_iter = 100      # 最大迭代次数
sa_T0 = 100            # 初始温度
sa_alpha = 0.95        # 降温系数
```

## 结果解读
### 1. 甘特图
- 纵轴：机器编号（M1-M3）
- 横轴：时间轴
- 色块：单个作业的执行区间，标注作业编号（J1-J50）
- 用途：直观查看机器负载均衡、作业执行顺序、空闲时间

### 2. Pareto 前沿图
- 横轴：Makespan（越小越好，代表生产效率）
- 纵轴：Total Flow Time（越小越好，代表资源利用率）
- 红色点：最终 Pareto 最优解集（非支配解，无法在不牺牲一个目标的前提下改进另一个目标）
- 用途：对比不同算法的优化效果，选择符合实际需求的调度方案

### 3. 量化指标
控制台输出示例：
```
[组合策略] Makespan = 1256.0, Total Flow Time = 48923.0, 用时 = 15.32s
```
- Makespan：所有作业完成的最晚时间（生产周期）
- Total Flow Time：所有作业从开始到完成的时间总和（资源占用总时长）
- 用时：算法运行时间（可评估算法效率）
