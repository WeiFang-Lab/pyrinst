# PyRInst

## 安装指南

### 推荐环境配置 (针对 MACE )
下面是一个验证过的环境配置步骤，用于支持 MACE 势能面计算：

```bash
conda create -n your_name python=3.13
pip install cuequivariance==0.7.0 cuequivariance-torch==0.7.0 cuequivariance-ops-torch-cu12==0.7.0
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install mace_torch-0.3.14-py3-none-any.whl #这里的 .whl 是我们单独提供的一个包，基于官方 mace_torch-0.3.14 修改
```

需要注意的点有：
- 如果使用其他镜像请仔细检查版本号是否一致
- 目前仅建议在 x86 平台进行计算任务
- 待补充。。。

### i-pi nvt-cc bug 修复
目前 i-pi 的 nvt-cc 实现存在 bug 需要修复，位于 `ipi/engine/motion/dynamics.py` 的 `NVTCCIntegrator` 类的 `step` 方法，需要在 

```python
# self.qcstep() # for the moment I just avoid doing the centroid step.
self.nm.free_qstep()
```
下再加一行

```python
self.nm.free_qstep()
```

在虚拟环境下，这个文件通常位于目录 `path/to/you/venv/lib/python3.13/site-packages/ipi/engine/motion/dynamics.py` 下，其中 `path/to/you/venv/` 是 conda 虚拟环境存放的目录，`python3.13/` 应当与实际安装的python 版本保持一致。

## 使用工作流

## Harmonic FEP 
Harm FEP 用于对给定的质心构型在特定温度下进行基于参考谐振子态的热力学修正，以获得更为精确的自由能估算。该模块包含构象参考态生成、基于正态模的构象采样、由于势能面差异的能量修正计算。为满足简谐近似合理性，尽量避免有过大虚频的构型。

整体工作流分为以下四个主要步骤：

### 1. 生成参考结构
首先需要通过的质心构型计算其 Hessian 矩阵及频率矩阵，构建参考谐振子

**相关脚本：** `pyrinst-gen-ref` (等效于 `bin/gen_ref.py`)

**使用示例：**
```bash
pyrinst-gen-ref geom.xyz -o ref_out -P MACE --model_path /path/to/model.model --device cuda
```

**参数：**
- `input`: 输入的质心几何结构 XYZ 文件
- `-o, --output`: 输出的参考状态 Pkl 文件名前缀（默认：`ref`）
- `-P, --PES`: 所选取的势能面（例如 `MACE`）
- **针对MACE PES 特别参数：**
  - `--model_path`: MACE 模型的路径位置。
  - `--dtype`: 精度要求（ `float64` 或 `float32`）
  - `--device`: 运行设备（ `cuda` 或 `cpu`）

脚本运行完毕后，将输出相应的频率状态信息并在当前目录生成包含频率和简正模等关键数据的 `.pkl` 文件。

---

### 2. 生成采样构象

在 normal modes 空间中，对简谐势下的系统进行采样，生成各个珠子的构型

**相关脚本：** `pyrinst-sampling` (等效于 `bin/sampling.py`)

**使用示例：**
```bash
pyrinst-sampling ref_out.pkl -T 300 -N 4096 -n 24 -o simulation.pos
```

**参数：**
- `input`: 第一步生成的 `pkl` 文件路径。
- `-T`: 目标采样温度，单位为 K（默认: `300`）。
- `-N`: 总采样数目（默认: `4096`）。
- `-n, --nbeads`: beads 个数（默认: `24`）。
- `-o, --output`: 输出文件名的前缀（默认: `simulation.pos` ，这会生成如同 `simulation.pos_00.xyz` 到 `simulation.pos_23.xyz` 等多个文件）

> **注意：** 该脚本在执行过程中不仅会输出 XYZ 格式的构象文件，还会**更新 `pkl` 文件**并存入珠子本身的 Harmonic reference 能量，**请保存好更新后的 `pkl` 副本。**

---

### 3. 能量估计 

> 这一步通常是最耗时间的步骤，在正式投入计算前应当仔细规划好如何分配资源使得计算效率最高


#### 使用 MACE 计算能量示例

一个示例脚本如下：

```bash
for d in {00..23}; do
    mace_eval_configs \
    --configs="simulation.pos_${d}.xyz" \
    --model="/path/to/model.model" \
    --output="simulation.pos_eval_${d}.xyz" \
    --device="cuda" \
    --enable_cueq \
    --no_forces \
    --batch_size=1024
done

```
基于效率考虑，需要注意的参数有：
- `--device` 尽量使用 `cuda`，cpu 计算速度极慢
- `--enable_cueq` 可以在大幅加快计算速度的同时减少显存占用，建议参考环境配置建议中步骤创建环境
- `--no_forces` 这是 `mace_beta` 中专门针对能量估计设置的选项，开启后会跳过力的计算，由于削减了大量耗时的 `auto_gradient()` 操作，同样可以在大幅加快计算速度的同时减少显存占用
- `--batch_size` 尽量占满单卡的显存，当然要预留部分余量防止偶现的显存未及时回收导致的 `OOM` 问题（如果出现输出文件为空或者一直阻塞在某个文件的情况大概率是由于显存不够）

---

### 4. 自由能微扰校正计算
从已经计算好能量的所有珠子构象及 `pkl` 文件中，计算得出真实的自由能估计值。

**相关脚本：** `pyrinst-fep-eval` (等效于 `bin/fep_eval.py`)

**使用示例：**
```bash
pyrinst-fep-eval ref_out.pkl --prefix simulation.pos_eval -n 24
```

**参数：**
- `input`: 此处为第一步生成的 `pkl` 文件
- `--prefix`: 与采样过程中生成的文件前缀保持一致
- `-n, --nbeads`: beads 数量（默认: `24`）

**脚本输出：**
- `reference`: 参考自由能估计值
- `correction`: FEP 求得的自由能修正量
- `Delta F`: 目标温度下的质心自由能与参考势能面**差值**，即训练有效势时的能量
- `uncertainty`: 校正结果的不确定度误差（基于自相关等因素给出）


## Instanton FEP 
在转变温度 Tc 之下，基于简谐近似的 Harm FEP 会带来较大误差。
Instanton FEP 用于这种低温下进行基于瞬子 (Instanton) 参考构型的热力学修正估算。其大致流程与 Harm FEP 类似，只是在寻找参考态的环节有所不同。

由于基于瞬子的修正计算，整体工作流分为以下五个主要步骤：

### 1. 生成初始结构
首先通过质心构型计算其参考谐振子态用于后续优化初猜。

**相关脚本：** `pyrinst-gen-ref`

**使用示例：**
```bash
pyrinst-gen-ref water.xyz -P MACE --model_path MACE-OFF23_medium_water_train3_run-1020_stagetwo.model
```
默认输出参考文件为 `ref.pkl`。

---

### 2. 优化至瞬子结构
在生成初始参考结构后，进行基于特定温度下的固定质心瞬子的几何优化。

**相关脚本：** `pyrinst-optimize`

**使用示例：**
```bash
pyrinst-optimize ref.pkl -o inst.pkl -T 300 --mode centroid -P MACE -F MACE-OFF23_medium_water_train3_run-1020_stagetwo.model -N 24 -s 0.189
```

**参数：**
- `input`: 第一步生成的 `ref.pkl` 路径
- `-o, --output`: 优化完成输出的 Pkl 名称（例如：`inst.pkl`）
- `-T`: 目标温度
- `--mode`: 此处使用 `centroid` 作为固定质心的瞬子优化模式
- `-P, --PES`: 所选体系势能面
- `-F`: 对应势能面的额外参数路径模型
- `-N`: beads 分布数目（如：`24`）
- `-s, --spread`: 瞬子初猜构型的长度

> **贴士：** 如果瞬子优化不能顺利收敛，请改变优化参数多次尝试，比如缩小优化步长`maxstep`、更换优化算法`opt`、调整初猜长度`spread`、启动`no-update`从而在优化中使用真实Hessian而非更新的近似Hessian等方法。

---

### 3. 生成瞬子采样构象
在获得优化的 Instanton 结构后采样其构型。

**相关脚本：** `pyrinst-sampling`

**使用示例：**
```bash
pyrinst-sampling inst.pkl -T 300 -N 2048
```
这一步骤与 `Harm-FEP` 完全一致，将产生 `simulation.pos` 系列珠子的构文件及更新 `inst.pkl` 文件里的谐振势能量。

---

### 4. 能量估计
参照上述 `Harm-FEP` 中说明使用。

---

### 5. 自由能微扰校正计算
最后通过生成的各珠子的能量进行 FEP 的最后求解并进行输出。

**相关脚本：** `pyrinst-fep-eval`

**使用示例：**
```bash
pyrinst-fep-eval inst.pkl --prefix simulation.pos
```


# For developers

## Architecture

```text
pyrinst/
├── bin/
├── docs/
├── src/
│   └── pyrinst/
│       ├── config/
│       ├── core/
│       ├── io/
│       ├── opt/
│       ├── potentials/
│       └── utils/
├── tests/
├── .gitignore
├── pyproject.toml
└── README.md
```
