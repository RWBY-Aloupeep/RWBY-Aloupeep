# Profile 重设计：三个样稿

点进每个文件夹，GitHub 会直接渲染那一版的 README。浅色、深色主题都适配，切换 GitHub 主题就能看到两种效果。

| 样稿 | 风格 | 主视觉 |
| --- | --- | --- |
| [`paper/`](./paper) | 论文首页：Computer Modern 字体、Abstract、Keywords、Figure 1 | 三个小仿真：势流翼型 / 混沌双摆 / 布料 |
| [`minimal/`](./minimal) | 极简学术主页：名字 + 一行方向 + 一张大图 | 七个初始角度相差 0.0002 rad 的双摆，先重合后发散 |
| [`log/`](./log) | 仿真日志：终端窗口里逐行输出自我介绍 | 右侧视口里的布料仿真，日志里的数字是真实运行结果 |

所有动画都由代码真实计算，不是手绘的：

- `sims.py`：双摆（RK4，能量误差约 1e-9）、布料（位置动力学 PBD，周期解，循环接缝误差约 3e-14）；流场复用 `../assets/banner.py`
- `gen.py`：排版并生成 SVG；字体（Computer Modern / Inter / JetBrains Mono，均为 OFL 许可）按需子集化后嵌入，任何系统上显示都一致

重新生成：`python drafts/gen.py`（需要 numpy、fonttools、brotli 和 npm）

选定一版之后，会把它整理成根目录的 README，并删除 `drafts/` 文件夹。
