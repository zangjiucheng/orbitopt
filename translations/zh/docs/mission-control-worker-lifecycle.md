# Mission Control-计算工人生命周期

Mission Control runs mission loaders in one persistent worker **process**
(`orbitopt.viz.compute_pool.ComputePool`, started with `spawn` on every OS)
so a GIL-holding compute cannot freeze the UI. This page lists what must hold
for that process and how it is tested.

## 担保

| 假设 | 预期行为 |
|---|---|
| 工人崩溃中程计算( native segfault, OOM kill) | *"计算过程意外退出"*, 警徽显示FAILED, 下一个负载开始 一个新生工人。 |
| 工人在闲置时死亡(在负载之间死亡) | 下一个负载是受到影响的**没有**:每次提交前的一次健康调查透明地取代了死去的工人。 |
| 计算运行时退出 | 10 s 下的窗口关闭;工人终止*并加入*(无僵尸);关闭后不应用 Stale results. |
| 闲暇时退出 | 工人离职后加入. |
| `QApplication.quit()` 不关闭窗口 | `aboutToQuit`终止工人资格,因此`concurrent.futures`的退出钩不会等待计算. |
| 装入竞速退出 | 封闭的游泳池拒绝新工作,因此退出后没有工人产卵. |
| App 崩溃/ 被强制杀死 | 工人通过母监(EOF在POSIX上的管子,在Windows上的流程控件)和在~1 s.Windows内部的退出通知,Windows从未再生育孤儿,因此旧的`getppid()`单检查从未在那里发射过. |
| 重复任务切换 | 整个会话的一个工人过程;加载时重新激活被忽略. |
| UI 计算时的反应 | a 16 ms 主线计时器持续勾选;无摊位 > 0.5 s. |

## 自动化测试

```bash
# Headless, Qt-free except QtCore -- runs anywhere, including Windows/macOS:
pytest tests/test_compute_pool.py -v

# Real window (needs a display; CI runs it under xvfb):
pytest tests/test_app_worker_lifecycle.py -v
```

`test_compute_pool.py`覆盖了池层上方的每一行,包括一个子进程"假应用程序",即SIGKILLed / TerminationProcess'd检查其工人退出. `test_app_worker_lifecycle.py`检查窗口线程:退出中程计算,退出闲置,崩溃 → FAILED → 下一个负载工作,重复切换.

CI只运行于Linux. 在发布前在macOS和Windows上运行`pytest tests/test_compute_pool.py`(写成OS-独立,但CI只在Linux上练习).

## 手动核对表(macOS和Windows)

以 `orbitopt-app`(或`python -m orbitopt.viz.app`)启动应用程序. 首先删除场景缓存, 所以负载真的计算 。 寻找活动监视器 / `ps -ef | grep spawn_main` (macOS) 或任务管理器 → 细节 / `Get-CimInstance Win32_Process -Filter "Name='python.exe'" | ft ProcessId,ParentProcessId,CommandLine` (Windows) 的工人. 这是孩子`python`,他的指挥线包含`multiprocessing.spawn`.

1. **One worker per session.** Switch Solar System → Artemis II → GOES →
   Solar System. Exactly one worker process exists, with the same PID throughout.
2. **UI stays live during compute.** While Artemis II computes, the spinner
   动画,窗口会平稳地移动和调整大小, Windows 从未标记过它
   "不回应"。
3. **工人杀了中计算。** 开始阿尔特米斯 II,然后杀死工人
   (`kill -9 <pid>` / 结束任务). *"没有上膛"* 对话框出现,
   警徽显示FAILED,任务列表可用. 选择 GOES:它装入一个
   新工人 PID.
4. **Worker killed while idle.** After a load finishes, kill the worker, then
   选择另一个未完成的任务。 它装入时没有错误对话框 。
5. **戒中计.** 启动Artemis II 立即退出(⌘Q/).
   窗口关闭 / Alt+ F4 。 应用程序最多在几秒钟后退出,没有
   工人的过程仍然存在。 在macOS上,`ps -o pid,stat,command`显示没有
   `Z`(僵尸)条目.
6. **应用武力杀死。** 启动Artemis II,然后杀死*应用*进程.
   (`kill -9` / 结束对母的任务). 工人在~1 – 2 s 内消失.
   这是Windows的特定案例.
7. **快速点击。** 快速点击数个任务,同时有一个在计算.
   列表在装入时被禁用; 无队列上方, 没有额外
   工人出现了。
8. **In-process fallback.** `ORBITOPT_INPROCESS_COMPUTE=1 orbitopt-app`: no
   工人流程已经开始;负荷仍在工作(UI可能会结巴-预期)。
