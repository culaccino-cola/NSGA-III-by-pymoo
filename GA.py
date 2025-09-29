import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymoo.core.problem import Problem
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.core.sampling import Sampling
from pymoo.operators.crossover.pntx import PointCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
import time

# ------------------------------
# 参数初始化
# ------------------------------
num_jobs = 50
num_machines = 3
min_jobs_per_machine = 5

# 每个作业在每台机器上的处理时间（只需选择其中一个机器）
path = "machines_jobs.xls"
data = pd.read_excel(path,sheet_name="Sheet1",index_col=0,header=3).T
processing_times = np.array(data)
print("每台机器的平均处理时间：", np.mean(processing_times, axis=0))

# ------------------------------
# 可行性初始化采样器：用于生成满足每台机器至少5个任务的初始解
# ------------------------------
class FeasibleInit(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        X = []
        while len(X) < n_samples:
            job_seq = np.random.rand(num_jobs)
            machine_choices = np.random.randint(0, num_machines, size=num_jobs)

            # 确保每台机器至少 min_jobs_per_machine 个任务
            for m in range(num_machines):
                count = np.sum(machine_choices == m)
                if count < min_jobs_per_machine:
                    replace_indices = np.random.choice(np.where(machine_choices != m)[0],
                                                       size=min_jobs_per_machine - count,
                                                       replace=False)
                    machine_choices[replace_indices] = m

            individual = np.concatenate([job_seq, machine_choices])
            X.append(individual)
        return np.array(X)

# ------------------------------
# 编码方案：每个作业对应两个变量（1）作业顺序（2）机器选择
# 加入约束：每台机器最少被分配 5 个作业
# ------------------------------
class SimpleJSSP(Problem):
    def __init__(self,encoding_type="default"):
        super().__init__(n_var=2*num_jobs,
                         n_obj=2,
                         n_constr=num_machines,  # 每台机器一个约束
                         xl=np.concatenate([np.zeros(num_jobs), np.zeros(num_jobs)]),
                         xu=np.concatenate([np.ones(num_jobs)*(num_jobs - 1), np.ones(num_jobs)*(num_machines - 1)]))
        self.encoding_type = encoding_type

    def _evaluate(self, X, out, *args, **kwargs):
        f1, f2 = [], []
        g = []

        for row in X:
            if self.encoding_type == "reverse":
                job_seq_keys = -row[:num_jobs]#改变顺序编码方式
            else:
                job_seq_keys = row[:num_jobs]
            machine_choices = row[num_jobs:].astype(int)
            job_order = np.argsort(job_seq_keys)

            # 用于记录每台机器上的调度时间线
            machine_schedules = [[] for _ in range(num_machines)]
            job_start_time = [0] * num_jobs
            job_end_time = [0] * num_jobs

            for job in job_order:
                m = machine_choices[job]
                pt = processing_times[job][m]

                if machine_schedules[m]:
                    last_end = machine_schedules[m][-1][1]
                else:
                    last_end = 0
                start = last_end
                end = start + pt

                machine_schedules[m].append((start, end))
                job_start_time[job] = start
                job_end_time[job] = end

            makespan = max(job_end_time)
            total_flow_time = sum(job_end_time)

            f1.append(makespan)
            f2.append(total_flow_time)

            # 计算每台机器作业数量是否满足下限约束
            machine_counts = [np.sum(machine_choices == m) for m in range(num_machines)]
            g_row = [min_jobs_per_machine - c for c in machine_counts]  # g(x) <= 0
            g.append(g_row)

        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.array(g)


# ------------------------------
# 甘特图绘制
# ------------------------------
def plot_gantt(job_order, machine_choices,title):
    machine_schedules = [[] for _ in range(num_machines)]
    schedule = []

    for job in job_order:
        m = machine_choices[job]
        pt = processing_times[job][m]

        if machine_schedules[m]:
            last_end = machine_schedules[m][-1][1]
        else:
            last_end = 0
        start = last_end
        end = start + pt
        machine_schedules[m].append((start, end))

        schedule.append((job, m, start, pt))

    print(f"\n调度方案如下-{title}：")
    for job, m, start, pt in schedule:
        print(f"J{job+1} 在M{m+1} 上执行，开始时间: {start}，处理时间: {pt}，结束时间: {start + pt}")

    colors = plt.cm.tab20(np.linspace(0, 1, num_jobs))
    fig, ax = plt.subplots(figsize=(12, 6))
    for job, m, start, pt in schedule:
        ax.barh(y=m, left=start, width=pt, color=colors[job % 20], edgecolor='black')
        ax.text(start + pt / 2, m, f"J{job+1}", ha='center', va='center', color='white', fontsize=6)

    ax.set_yticks(range(num_machines))
    ax.set_yticklabels([f"M{i+1}" for i in range(num_machines)])
    ax.set_xlabel("Time")
    ax.set_ylabel("Machine")
    ax.set_title(f"Gantt Chart-{title}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ------------------------------
# 局部搜索方法：模拟退火
# ------------------------------
def simulated_annealing(x, problem, max_iter=100, T0=100, alpha=0.95):
    current = x.copy()
    current_f = evaluate_individual(current, problem)
    best = current.copy()
    best_f = current_f.copy()
    T = T0
    for _ in range(max_iter):
        neighbor = current.copy()
        if np.random.rand() < 0.5:
            i, j = np.random.choice(num_jobs, 2, replace=False)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        else:
            idx = np.random.randint(num_jobs, 2*num_jobs)
            neighbor[idx] = np.random.randint(0, num_machines)
        f_neighbor = evaluate_individual(neighbor, problem)
        delta = sum(f_neighbor) - sum(current_f)
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            current = neighbor.copy()
            current_f = f_neighbor.copy()
            if sum(current_f) < sum(best_f):
                best = current.copy()
                best_f = current_f.copy()
        T *= alpha
    return best

def evaluate_individual(x, problem):
    out = {}
    problem._evaluate(np.array([x]), out)
    return out["F"][0]

# ------------------------------
# 主运行函数
# ------------------------------
def run_ga(name,encoding_type="default",use_sa=False,use_alt_operator=False):
    print(f"\n=== 实验：{name} ===")
    problem = SimpleJSSP(encoding_type=encoding_type)
    ref_dirs = get_reference_directions("energy", 2,100, n_partitions=20)
    # 是否使用改进遗传算子
    if use_alt_operator:
        algorithm = NSGA3(ref_dirs=ref_dirs, sampling=FeasibleInit(),
                          crossover=PointCrossover(n_points=5,prob=1.0), mutation=PolynomialMutation(prob=1.0),
                          pop_size=200,n_offsprings=100,
                          ellipsoid_sampling=True,eliminate_duplicates=True,save_history=True)
    else:
        algorithm = NSGA3(ref_dirs=ref_dirs, sampling=FeasibleInit(),
                          pop_size=200, n_offsprings=100,
                          ellipsoid_sampling=True, eliminate_duplicates=True,save_history=True)

    termination = get_termination("n_gen", 500)

    start_time = time.time()
    res = minimize(problem, algorithm, termination, seed=1, verbose=True, save_history=True)
    end_time = time.time()

    best_X = res.X[0]
    if use_sa:
        best_X = simulated_annealing(best_X, problem)

    job_order = np.argsort(best_X[:num_jobs])
    machine_choices = best_X[num_jobs:].astype(int)
    best_f = evaluate_individual(best_X, problem)
    print(f"[{name}] Makespan = {best_f[0]}, Total Flow Time = {best_f[1]}, 用时 = {end_time - start_time:.2f}s")
    plot_gantt(job_order, machine_choices,name)
    # ------------------------------
    # # 可视化 Pareto 前沿
    # ------------------------------
    F = res.F
    plt.figure(figsize=(12,10))

    # 尝试绘制历史前沿点
    if hasattr(algorithm, 'history'):
        for gen in algorithm.history:
            if gen.opt is not None and gen.opt.get('F') is not None:
                front = gen.opt.get('F')
                plt.scatter(front[:, 0], front[:, 1], alpha=0.7, s=10)

    # 绘制最终解集
    plt.scatter(res.F[:, 0], res.F[:, 1], c='red', s=20,alpha=0.7)
    plt.xlabel("Makespan")
    plt.ylabel("Total Flow Time")
    plt.title(f"Pareto Fronts-{name}")
    plt.grid(True)
    plt.legend()
    plt.show()
    return best_f,F,name

# ------------------------------
# 执行 5 类实验
# ------------------------------
results = []
pareto_data = []
plt.rcParams['font.sans-serif']=['SimHei']#中文显示
plt.rcParams['axes.unicode_minus'] = False#负号
results.append(run_ga("基线算法"))
results.append(run_ga("A-编码策略", encoding_type="reverse"))
results.append(run_ga("B-遗传算子", use_alt_operator=True))
results.append(run_ga("C-局部搜索", use_sa=True))
results.append(run_ga("D-编码策略+遗传算子", encoding_type="reverse",use_alt_operator=True))
results.append(run_ga("E-编码策略+局部搜索", encoding_type="reverse", use_sa=True))
results.append(run_ga("F-遗传算子+局部搜索", use_alt_operator=True, use_sa=True))
results.append(run_ga("G-组合策略", encoding_type="reverse", use_sa=True, use_alt_operator=True))

# 解包 Pareto 前沿数据
for res in results:
    if isinstance(res, tuple):
        fval, fset, label = res
        pareto_data.append((fset, label))

# 绘制所有实验的综合 Pareto 图
plt.figure(figsize=(8,6))
for front, label in pareto_data:
    plt.scatter(front[:,0], front[:,1], s=15, label=label)
plt.xlabel("Makespan")
plt.ylabel("Total Flow Time")
plt.title("Comparison of Pareto Fronts")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("pareto_comparison.png", dpi=300)
plt.show()
