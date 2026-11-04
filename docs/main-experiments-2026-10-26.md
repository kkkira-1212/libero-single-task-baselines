# 单任务主实验进度与结果

生成时间（UTC）：2026-10-27T03:16:37.374348+00:00

任务：alphabet soup → basket；40 条训练示范、10 条验证示范。结果来自各自保存的评估报告。

| 方法／协议 | 种子 | 更新次数×batch | 成功／回合 | 状态 |
|---|---:|---:|---:|---|
| ACT／chunk20/exec5 | 0 | 10000×16 | 13/20 | completed |
| ACT／chunk20/exec5 | 1 | 10000×16 | 9/20 | completed |
| DP／DDPM100/exec5 | 0 | 3000×16 | 10/20 | completed |
| DP／DDPM100/exec5 | 1 | 3000×16 | 9/20 | completed |
| DP／DDPM100/exec1 | 0 | 3000×16 | 16/20 | completed |
| DP／DDPM100/exec1 | 1 | 3000×16 | 11/20 | completed |
| BAKU／chunk10/temporalagg | 0 | 10000×16 | 19/20 | completed |
| BAKU／chunk10/temporalagg | 1 | 10000×16 | 1/20 | completed |
| A2A／flow6/exec1 | 0 | 40000×4 | 0/20 | completed |
| A2A／flow6/exec1 | 1 | 40000×4 | 0/20 | completed |
| A2A／flow1/exec1 | 0 | 40000×4 | 0/20 | completed |
| A2A／flow1/exec1 | 1 | 40000×4 | 0/20 | completed |
| A2A／flow6/10k_pilot | 0 | 10000×4 | 0/20 | completed |
| A2A／flow1/10k_pilot | 0 | 10000×4 | 0/20 | completed |
| BAKU／3k_pilot | 0 | 3000×16 | 0/20 | completed |
| BAKU／frozen/newIDs40..49 | 0 | 10000×16 | 9/10 | completed |
| A2A／40k_seed1_hardware_failed_attempt | 1 | 40000×4 | 待完成 | FAILED |

解释与限制：

- 这是一个任务的适配实验，不是论文完整 suite 复现；相同初始状态被反复用于开发评估。
- 不同方法的历史长度、优化器、初始化、归一化和执行方式不同，不能将差异仅归因于架构。
- BAKU 两个种子差异很大，必须同时报告，不能只选择 95% 那次。
- A2A 的 6 步为预定主设置，1 步为同一最终模型上的诊断对照；没有独立的论文延迟复现。
- 表中包含早期失败试验和新状态确认，role 字段在 CSV/JSON 中区分；不混合计算独立测试成功率。
- 缺失／失败任务保留为未完成，不能计作已完成的 0% 成功率。
- 全部预定比较是否完成：True
