# AGENTS.md — score_predictor 项目协作说明

> 本文件是 `score_predictor` 项目的 Codex / AI 协作说明。新开 Codex 窗口时，请先阅读本文件，再执行任何修改。  
> 当前项目正在从旧的 “A 源建模 + B 源 Value Analysis” 架构，升级为新的 **双赔率通道共同服务预测模型** 架构。

---

## 1. 项目定位

本项目是一个足球 **90 分钟比分概率预测与多盘口联合校准系统**，当前核心包为 `score_predictor`。

项目目标不是给出投注建议，也不是保证命中比分，而是提供：

- 赛前 90 分钟比分概率分布。
- 胜平负、大小球、BTTS、总进球等概率输出。
- 多盘口联合校准后的 `lambda_home / lambda_away / rho`。
- Poisson / Dixon-Coles 比分矩阵。
- 国际赔率通道 + 体彩赔率通道的联合建模。
- 盘口一致性审计、市场质量评分、数据完整度和模型置信度。
- 预测历史与赛后复盘。

本项目只做概率建模、盘口数据校准、风险审计、数据展示和赛后复盘。

禁止输出或暗示：

```text
推荐下注
稳赚
稳赢
稳胆
必买
包中
```

必须保留免责声明：

```text
本系统仅用于概率建模、数据分析和赛后复盘，不构成投注建议。
```

---

## 2. 当前核心架构：双赔率通道联合校准

当前主线是 V3：

```text
国际 API 赔率 + 体彩 YAML 赔率
  ↓
赔率去水
  ↓
市场质量评分 / 盘口一致性评分
  ↓
自适应权重
  ↓
多市场联合校准
  ↓
反推 market_lambda_home / market_lambda_away / rho
  ↓
可选 Dixon-Coles 低比分修正
  ↓
生成比分矩阵
  ↓
输出 1X2 / Top scores / total goals / OU / BTTS / diagnostics
```

V3 不是训练好的机器学习模型，而是：

```text
多盘口联合校准 + Poisson / Dixon-Coles 比分概率模型
```

不要把 V3 改成黑箱机器学习模型。  
不要用赛后结果、复盘结果反向修改赛前预测。  
不要让任何 EV / Value Analysis 逻辑影响 lambda。  
当前新架构中，Value Analysis 已不再是主功能。

---

## 3. 新旧架构变更说明

旧架构：

```text
A 源：国际盘，用于 V3 calibration
B 源：体彩，用于 Value Analysis / EV / Edge
```

新架构：

```text
国际赔率通道：primary calibration channel
体彩赔率通道：supplemental soft calibration channel
```

新架构中：

- 中国体彩 YAML 不再是 Value Analysis 源。
- 体彩固定奖金不再用于 EV / Edge / breakeven probability。
- 体彩固定奖金是 **补充校准源**，用于补充国际 API 缺失或不完整的市场。
- 国际 API 与体彩 YAML 都可以进入 V3 calibration，但权重不同。
- 体彩赔率必须作为 **soft calibration**，不能单独支配最终比分。

UI、报告、文案中不要再出现以下主流程概念：

```text
Value Analysis
EV
Edge
breakeven probability
independent_comparison
reference_only
B源价值比较
体彩价值分析
```

如果历史文件 `value_analysis.py` 暂时保留，只能作为 legacy code，不应出现在主 UI 流程。

---

## 4. 推荐数据结构

新的数据源角色建议使用：

```yaml
odds_channels:
  international:
    role: primary_calibration
    source: the_odds_api
    provider: pinnacle
    weight: 1.0

  sporttery:
    role: supplemental_calibration
    source: yaml
    provider: sporttery
    weight: 0.35
```

推荐统一市场结构：

```yaml
markets:
  international:
    source: the_odds_api
    provider: pinnacle
    weight: 1.0
    h2h: ...
    h2h_3_way: ...
    spreads: ...
    totals: ...
    alternate_totals: ...
    btts: ...
    alternate_spreads: ...
    draw_no_bet: ...
    team_totals: ...

  sporttery:
    source: yaml
    provider: sporttery
    weight: 0.35
    sporttery_1x2: ...
    sporttery_handicap_3way: ...
    sporttery_correct_score: ...
    sporttery_total_goals: ...
    sporttery_half_full: ...
```

V3 calibration 应通过类似函数合并约束：

```python
calibration_constraints = build_constraints_from_channels(
    international_channel,
    sporttery_channel,
    settings,
)
```

---

## 5. 当前主要模块

常见目录：

```text
src/score_predictor/
  predictor.py
  schemas.py
  report.py

src/score_predictor/v3/
  ensemble.py
  market_calibration.py
  dixon_coles.py
  score_calibration.py
  handicap_consistency.py
  sensitivity.py
  value_analysis.py        # legacy only; 不应再作为主流程 UI 功能

src/score_predictor/ui/
  streamlit_app.py
  charts.py
  form_helpers.py
  yaml_io.py
  theme.py

src/score_predictor/connectors/
  the_odds_api.py
  odds_api_normalizer.py
  cache.py

src/score_predictor/history/
  store.py
  models.py
  evaluator.py

config/
  provider_the_odds_api.example.yaml
  worldcup_2026_groups.yaml
  team_aliases_worldcup_2026.yaml

examples/
tests/
scripts/
```

如果某些文件不存在，先检查当前工作区，不要凭空假设。  
新增功能时尽量放在对应模块，不要把大量逻辑堆进 `streamlit_app.py`。

---

## 6. 国际赔率通道规则

国际赔率通道是主建模通道，典型来源：

```text
The Odds API
Pinnacle
Betfair Exchange
Bet365
Matchbook
其他国际主流 bookmaker
```

优先 bookmaker：

```text
pinnacle
betfair_ex_eu / betfair_ex_uk
bet365
matchbook
```

The Odds API 常用 markets：

### 一级市场：直接参与 V3 calibration loss

```text
h2h
h2h_3_way
spreads
totals
alternate_totals
btts
```

用途：

- `h2h / h2h_3_way`：胜平负，约束胜平负概率。
- `spreads`：让球，约束强弱差。
- `totals`：主大小球，约束总进球。
- `alternate_totals`：多条大小球线，更稳地拟合总进球分布。
- `btts`：双方进球，约束 1-1、2-1、1-0、2-0 等相邻比分结构。

### 二级市场：第一阶段 audit / diagnostics，不强行影响 lambda

```text
alternate_spreads
draw_no_bet
team_totals
```

说明：

- `alternate_spreads` 可用于净胜球结构审计。
- `draw_no_bet` 可用于胜负方向审计。
- `team_totals` 对单队进球很有价值，但第一阶段可只解析展示和审计。

### 暂时忽略

```text
player markets
cards markets
corners markets
first goal scorer
anytime goal scorer
half-time markets
```

除非后续实现半场子模型、球员模型、角球或牌类模型，否则不要把这些盘口接入 V3 全场比分主模型。

---

## 7. 体彩赔率通道规则

中国体彩 YAML 是 **supplemental soft calibration channel**。

它不是 Value Analysis，不计算 EV，不计算 Edge，不输出 breakeven probability。

体彩市场处理规则如下。

### 7.1 胜平负固定奖金

示例：

```yaml
sporttery_1x2:
  home: 1.62
  draw: 3.32
  away: 4.75
```

用途：

- 若国际 API 已有 `h2h / h2h_3_way`，体彩 1X2 低权重参与。
- 若国际 API 缺少 1X2，体彩 1X2 可作为候补输入。

默认低权重，不应覆盖国际主盘口。

### 7.2 让球胜平负固定奖金

示例：

```yaml
sporttery_handicap_3way:
  line: -1
  home_win: 3.11
  draw: 3.20
  away_win: 2.02
```

重要：体彩让球胜平负是三项市场，不是亚洲让球二项盘口。

用途：

- 按三项让球胜平负处理。
- 可低权重约束净胜球结构。
- 可用于强弱差一致性审计。
- 不得误当 Asian Handicap 二项盘。

### 7.3 比分固定奖金 / Correct Score

这是体彩最重要的补充市场。国际 API 当前通常没有 correct score，但体彩有。

示例：

```yaml
sporttery_correct_score:
  scores:
    "0-0": 9.50
    "1-0": 5.40
    "1-1": 5.00
    "2-0": 6.50
    "2-1": 6.00
```

用途：

- 作为 soft constraint 参与比分矩阵校准。
- 帮助稳定 Top score。
- 降低“最可能比分对 lambda/rho 敏感”的风险。

必须注意：

- 必须先去水。
- 权重必须较低。
- 不能让 correct score 单独决定最大概率比分。
- `home_other / draw_other / away_other` 第一阶段可以只 audit，不参与 loss。
- 如果存在 other 项但暂不使用，输出 warning：`correct_score_other_not_used`。

建议 soft loss：

```python
loss_correct_score = weight * sum(
    (model_score_prob[score] - market_score_prob[score]) ** 2
    for score in listed_scores
)
```

### 7.4 总进球固定奖金

示例：

```yaml
sporttery_total_goals:
  odds:
    "0": 9.50
    "1": 4.30
    "2": 3.10
    "3": 3.60
    "4": 6.20
    "5": 12.50
    "6": 22.00
    "7+": 35.00
```

用途：

- 作为 soft constraint 参与总进球分布校准。
- 与国际 API 的 `totals / alternate_totals` 共同拟合 `lambda_home + lambda_away`。
- `7+` 应映射为 `P(total_goals >= 7)`。

建议 soft loss：

```python
loss_total_goals = weight * sum(
    (model_total_goals_prob[k] - market_total_goals_prob[k]) ** 2
    for k in ["0", "1", "2", "3", "4", "5", "6", "7+"]
)
```

### 7.5 半全场

示例：

```yaml
sporttery_half_full:
  HH: 2.51
  HD: 16.00
  HA: 36.00
  DH: 4.25
  DD: 4.80
  DA: 10.00
  AH: 25.00
  AD: 16.00
  AA: 8.40
```

第一阶段只进入 audit / diagnostics，不参与全场 lambda 校准。

原因：

- 当前 V3 是全场 90 分钟比分模型。
- 半全场需要半场 lambda 或状态转移模型。
- 强行用半全场影响全场 lambda 可能造成错误约束。

默认：

```yaml
sporttery_half_full_weight: 0.00
```

---

## 8. 默认权重与自适应权重

默认权重建议：

```yaml
odds_channel_weights:
  international:
    h2h_3_way: 1.00
    h2h: 1.00
    spreads: 0.50
    totals: 1.00
    alternate_totals: 0.80
    btts: 0.60
    alternate_spreads: 0.30
    team_totals: 0.40
    draw_no_bet: 0.20

  sporttery:
    sporttery_1x2: 0.15
    sporttery_handicap_3way: 0.15
    sporttery_total_goals: 0.30
    sporttery_correct_score: 0.20
    sporttery_half_full: 0.00
```

UI 中允许用户调整体彩权重，但要设置合理上下限：

```yaml
sporttery_1x2: [0.05, 0.35]
sporttery_handicap_3way: [0.05, 0.30]
sporttery_total_goals: [0.10, 0.45]
sporttery_correct_score: [0.10, 0.35]
sporttery_half_full: [0.00, 0.00]
```

最终权重不应直接使用固定值，而应按市场质量和一致性修正：

```python
final_weight = base_weight * market_quality_score * consistency_score
```

其中：

```python
market_quality_score in [0.25, 1.00]
consistency_score in [0.25, 1.00]
```

如果第一阶段实现复杂，可以先实现评分函数、返回结构和 UI 展示，再逐步接入更细的 loss 权重。

---

## 9. 市场质量评分

建议新增函数：

```python
score_market_quality(market: MarketInput) -> dict
```

输出：

```python
{
    "score": float,
    "level": "high" | "medium" | "low",
    "drivers": list[str],
    "warnings": list[str],
}
```

评分维度：

1. 市场是否完整。
2. 是否经过人工确认。
3. 数据是否临场或足够新。
4. 是否包含“其他项”。
5. 返还率是否合理。
6. 是否存在明显缺失或异常赔率。
7. 是否与同类市场重复冲突。

规则：

- correct score 如果只录入少数比分，质量不得超过 medium。
- correct score 如果缺少 `home_other / draw_other / away_other`，质量最多 medium。
- total goals 如果包含 `0/1/2/3/4/5/6/7+`，质量可以 high。
- half_full 可以评分，但第一阶段不参与 lambda。

---

## 10. 返还率 / overround

所有赔率市场进入模型前必须去水。

对每个市场计算：

```python
raw_prob_sum = sum(1 / odds)
payout_rate = 1 / raw_prob_sum
overround = raw_prob_sum - 1
```

如果某个市场 `payout_rate` 明显偏低，说明水位厚，应降低 `market_quality_score`。

不要用单个赔率高低决定权重，应使用市场整体返还率、完整度和一致性决定权重。

需要支持：

- 2 项市场去水。
- 3 项市场去水。
- 多结果市场去水。
- Correct Score 去水。
- Total Goals `0/1/2/3/4/5/6/7+` 去水。

---

## 11. 盘口一致性评分

建议新增函数：

```python
score_channel_consistency(international_constraints, sporttery_constraints) -> dict
```

输出：

```python
{
    "score": float,
    "level": "aligned" | "mild_conflict" | "strong_conflict",
    "conflicts": list[str],
    "warnings": list[str],
}
```

检测：

1. 国际 totals 偏大，但体彩 total_goals 明显偏小。
2. 国际 h2h 主队强，但体彩让球胜平负不支持。
3. 国际 BTTS Yes 高，但体彩 correct score 集中在零封比分。
4. 体彩 correct_score 与体彩 1X2 自身不一致。
5. 体彩 total_goals 与体彩 correct_score 自身不一致。

如果出现 strong conflict：

- 不要报错。
- 降低对应市场权重。
- 降低模型置信度。
- UI 显示“盘口源分歧”。
- 预测仍应可以继续运行。

---

## 12. The Odds API 接入规则

API key 只能从环境变量读取：

```text
THE_ODDS_API_KEY
```

不要把 API key 写进：

```text
代码
README
YAML
日志
metadata
测试 fixture
报错详情
Streamlit 页面
```

PowerShell 设置方式：

```powershell
[Environment]::SetEnvironmentVariable("THE_ODDS_API_KEY", "your_new_key", "User")
```

Python 检查方式：

```powershell
python -c "import os; print('set' if os.getenv('THE_ODDS_API_KEY') else 'missing')"
```

The Odds API 只作为数据源，不要在 V3 核心模型文件中直接联网。

推荐流程：

```text
UI / script
  ↓
connectors/the_odds_api.py
  ↓
raw response cache
  ↓
odds_api_normalizer.py
  ↓
V3-compatible payload / YAML
  ↓
predictor
```

UI 默认建议：

```text
regions = eu
bookmaker = pinnacle
```

盘口模式：

```text
省额度模式：h2h, spreads, totals
完整建模模式：h2h, h2h_3_way, spreads, totals, alternate_totals, btts
扩展审计模式：完整建模模式 + alternate_spreads, draw_no_bet, team_totals
```

避免自动频繁刷新 API，防止消耗额度。UI 中应由用户点击“查找赛事”“查询可用盘口”“拉取赔率”后再请求。

单元测试必须 mock 网络请求，不要真实消耗 API quota。

---

## 13. 世界杯 2026 分组和队名

UI 不要让用户手敲队名。优先使用：

```text
config/worldcup_2026_groups.yaml
config/team_aliases_worldcup_2026.yaml
```

UI 可显示中文队名，API 匹配使用英文 canonical name。

需要支持常见别名：

```text
Korea Republic / South Korea
Czech Republic / Czechia
USA / United States
Türkiye / Turkey / Turkiye
Côte d’Ivoire / Cote d'Ivoire / Ivory Coast
Curaçao / Curacao
IR Iran / Iran
Cabo Verde / Cape Verde
DR Congo / Congo DR / Democratic Republic of the Congo
Bosnia and Herzegovina / Bosnia & Herzegovina / Bosnia-Herzegovina / Bosnia
Mexico / México
```

不要把 48 队和分组硬编码在 `streamlit_app.py` 里。  
如果分组、队名或 API 名称后续变化，只改配置文件。

---

## 14. Streamlit UI 设计原则

当前 UI 是深色中文金融终端风格。

保持：

```text
深蓝黑背景
中文 tab
中文按钮
中文图表标题
KPI 卡片
预测看板
国际赔率通道
体彩赔率通道
多市场联合校准
盘口一致性审计
预测历史
审计与风险
```

全局字体优先：

```text
Microsoft YaHei
Noto Sans SC
PingFang SC
Hiragino Sans GB
SimHei
sans-serif
```

不要重新变回默认 Streamlit 实验风格。

UI 必须保留：

```text
YAML 上传
手动输入
国际赔率通道
体彩赔率通道
预测历史
审计与风险
下载按钮
market_only_mode
```

UI 不应再展示：

```text
Value Analysis
EV
Edge
breakeven probability
独立价值比较
体彩价值分析
```

按钮和文案用中文，例如：

```text
开始预测
查找赛事
查询可用盘口
拉取赔率
应用到国际赔率通道
上传体彩 YAML
保存赛果并复盘
导出预测历史 CSV
```

体彩赔率通道页面应展示识别到的市场：

```text
胜平负
让球胜平负
比分固定奖金
总进球固定奖金
半全场
```

并展示每个市场状态：

```text
参与 soft calibration
audit-only
ignored
```

同时展示：

```text
base_weight
market_quality_score
consistency_score
final_weight
payout_rate
warnings
```

---

## 15. 风险提示文案规则

旧提示：

```text
V3 缺少比分固定奖金输入
```

应改得更准确。

推荐提示：

1. 如果国际 API 没有 correct score，但体彩 YAML 有：

```text
国际赔率通道未返回正确比分盘口；已使用体彩比分固定奖金作为补充校准源。
```

2. 如果两个通道都没有 correct score：

```text
缺少正确比分盘口，具体比分排序可能对 lambda/rho 较敏感。
```

3. 如果体彩比分固定奖金参与 soft calibration：

```text
体彩比分固定奖金已作为补充 soft calibration，不会单独决定最大概率比分。
```

4. 如果体彩总进球固定奖金参与校准：

```text
体彩总进球固定奖金已参与总进球分布校准。
```

5. 如果有 BTTS：不要再提示缺少双方进球市场。

6. 如果没有 BTTS：

```text
缺少双方进球盘口，1-1、2-1、1-0 等相邻比分区分稳定性较弱。
```

7. 如果 Top1 与 Top2 分差很小：

```text
最可能比分对 lambda 或 rho 的小幅变化较敏感，请同时参考 Top5 比分簇。
```

8. 如果市场质量低：

```text
体彩该市场返还率偏低或录入不完整，已自动降低校准权重。
```

9. 如果盘口源冲突：

```text
国际赔率通道与体彩赔率通道存在方向分歧，模型置信度已下调。
```

---

## 16. 赛前情报规则

赛前情报保留，但定位为事实修正，不是赔率源。

它可以影响：

```text
lambda_home
lambda_away
total_goals tendency
confidence_score
risk_score
```

但必须限制幅度。

允许输入事实：

```text
官方首发
伤停
停赛
主力轮换
赛程休息天数
天气
场地
中立场
比赛性质
小组出线形势
```

禁止输入观点：

```text
专家推荐
媒体预测
投注建议
盘口解读
网友观点
```

修正规则：

```text
单个事实通常 1%-5%
重大事实最多 8%-12%
总修正不超过 ±15%
```

赛前情报不应推翻市场盘口，只能轻量修正。

---

## 17. 预测历史与赛后复盘规则

预测历史用于保存赛前预测，不用于改写模型结果。

默认存储：

```text
data/prediction_history/predictions.sqlite
```

如果使用 JSONL，也要封装统一接口，方便以后替换。

每次预测成功后保存：

```text
prediction_id
match_id
created_at
home_team
away_team
match_date
competition
stage
input_yaml_text
input_hash
settings
odds_channels
lambda_summary
rho
probabilities_1x2
top_scores
total_goals_distribution
over_under_probabilities
btts_probabilities
warnings
confidence_scores
market_quality_summary
channel_consistency_summary
api_source_metadata
app_version
```

赛后只回填 90 分钟比分：

```text
actual_home_goals_90
actual_away_goals_90
actual_result_90
actual_total_goals_90
actual_btts
settled_at
notes
```

不要把加时和点球混入 90 分钟模型。

复盘指标包括：

```text
胜平负是否命中
Top 1 / Top 3 / Top 5 比分是否命中
大小球 1.5 / 2.5 / 3.5 是否命中
BTTS 是否命中
Brier score for 1X2
Log loss for 1X2
实际比分在比分矩阵中的概率排名
中文复盘摘要
```

---

## 18. Historical Goal Models 规划

未来可以加入历史进球弱模型，但不能覆盖市场盘口。

可选模块：

```text
historical_goal_models/
  mle_poisson.py
  ewma.py
  ar_goal.py
  ensemble.py
```

建议权重：

```text
市场 lambda：80%-90%
历史 MLE / EWMA：10%-15%
AR1 / AR2：0%-5%
```

如果没有可靠历史数据，默认不要启用。  
AR2 是弱信号，不能主导结果。

---

## 19. 默认模型参数建议

一般默认：

```text
market_only_mode = true
dc_enabled = true
max_goals = 8
market_weight = 1.0
x1x2_weight = 1.0
ou_weight = 1.0
btts_weight = 0.6
correct_score_weight = 0.20 - 0.35
team_adjustment_strength = 1.0
```

新双通道默认：

```text
国际赔率通道：primary calibration
体彩赔率通道：supplemental soft calibration
```

体彩比分固定奖金和总进球固定奖金可以参与预测，但必须是低权重 soft constraint。

---

## 20. 安全和合规规则

禁止：

```text
硬编码 API key
输出 API key
把 API key 写入 YAML / JSON / metadata
把 EV 解释为下注建议
生成“稳赢”“稳赚”“必买”表达
用赛后比分影响赛前预测
把加时/点球当成 90 分钟结果
重写 V3 核心逻辑为黑箱模型
无故删除 YAML 上传和手动输入功能
把体彩让球胜平负误解析为亚洲让球二项盘
让半全场强行影响全场 lambda
引入球员、角球、牌类盘口到全场比分主模型
因为盘口源分歧就报错中断预测
让 correct score 单独决定最大概率比分
```

必须：

```text
保留免责声明
保留审计与风险
保留 market_only_mode
保留 YAML 上传
保留测试
赔率进入模型前去水
体彩作为 soft calibration 时低权重处理
```

---

## 21. 测试要求

每次修改后至少运行：

```powershell
python -m pytest -q
```

如果改 UI，至少确认：

```powershell
python -m streamlit run src\score_predictor\ui\streamlit_app.py
```

如果改 The Odds API：

```text
单元测试必须 mock 网络请求，不要真实消耗 API quota。
```

如果改预测历史：

```text
测试保存预测记录、回填赛果、Top 5 命中、Brier score、导出 CSV。
```

如果改双赔率通道或 calibration，必须覆盖：

```text
国际通道 h2h/spreads/totals 可以独立预测
国际通道加入 btts 后不再提示缺少 BTTS
国际通道加入 alternate_totals 后总进球拟合使用多条大小球
体彩 YAML correct_score 可以作为补充 soft calibration 参与比分矩阵拟合
体彩 YAML total_goals 可以作为补充 soft calibration 参与总进球分布拟合
体彩 YAML 1X2 可以低权重参与校准
体彩让球胜平负按三项市场处理，不得误当 Asian Handicap 两项市场
体彩半全场第一阶段只进入 audit，不影响 lambda
UI 不再出现 EV / Edge / breakeven
如果国际无 correct_score 但体彩有，应提示已使用体彩比分固定奖金作为补充校准源
如果两个通道都没有 correct_score，才提示缺少正确比分盘口
correct score 缺少 other 项时，质量最多 medium
strong_conflict 不报错，只降低权重和模型置信度
sporttery_half_full_weight 必须保持 0，不影响 lambda
```

---

## 22. 常用运行命令

安装 UI 依赖：

```powershell
python -m pip install -e ".[ui]"
```

启动 UI：

```powershell
python -m streamlit run src\score_predictor\ui\streamlit_app.py
```

运行测试：

```powershell
python -m pytest -q
```

检查 API key：

```powershell
python -c "import os; print('set' if os.getenv('THE_ODDS_API_KEY') else 'missing')"
```

旧 CLI 示例：

```powershell
python -m score_predictor.cli predict examples\match_v3_multi_market.yaml --dc-enabled true
```

The Odds API 查询可用盘口示例：

```powershell
python scripts\fetch_the_odds_api_match.py --sport-key soccer_fifa_world_cup --event-id d1f4f946c70a0b4e81f5d43e9d32361c --regions eu --print-markets
```

The Odds API 拉取完整建模盘口示例：

```powershell
python scripts\fetch_the_odds_api_match.py --sport-key soccer_fifa_world_cup --event-id d1f4f946c70a0b4e81f5d43e9d32361c --regions eu --markets h2h,h2h_3_way,spreads,totals,alternate_totals,btts --bookmaker pinnacle --output data\input\generated\canada_bosnia_the_odds_api_rich.yaml
```

---

## 23. 开发方式

做新功能时优先拆小步：

```text
1. 数据结构 / schema
2. connector / parser
3. calibration constraints
4. predictor 集成
5. UI 展示
6. tests
7. README / AGENTS.md
```

不要一次性无边界大改多个核心模块。  
大改前建议先保存当前版本：

```powershell
git status
git add .
git commit -m "baseline before dual odds channel calibration"
```

每次完成后回复：

```text
新增/修改文件
核心实现点
如何运行
测试结果
当前限制
下一步建议
```

---

## 24. 项目当前优先级

当前优先级：

```text
P0：稳定 V3 + 深色中文 UI + 双赔率通道校准
P1：The Odds API 国际赔率通道，支持完整建模模式
P2：体彩 YAML 作为 supplemental soft calibration，重点接入比分固定奖金和总进球固定奖金
P3：市场质量评分、返还率、盘口一致性评分、模型置信度
P4：预测历史与赛后复盘
P5：赛前情报事实修正
P6：Historical Goal Models 弱信号
P7：更多 API provider / TheStatsAPI
```

任何任务都不要破坏 P0。

---

## 25. 新 Codex 接手时的第一句话建议

新开 Codex 窗口时可以直接说：

```text
请先阅读项目根目录的 AGENTS.md。当前项目已升级为双赔率通道架构：国际赔率通道用于 primary calibration，体彩 YAML 用于 supplemental soft calibration。请不要再使用旧的 Value Analysis / EV / Edge 逻辑。接下来我要继续实现体彩比分固定奖金和总进球固定奖金进入 V3 soft calibration。
```
