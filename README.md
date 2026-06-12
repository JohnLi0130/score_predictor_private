# 世界杯 90 分钟比分预测引擎 V0

这是一个赛前 90 分钟足球比分概率引擎，用于个人娱乐和概率建模练习，不构成投注建议，也不提供任何投注推荐。

## 功能

- 读取单场比赛 YAML 输入
- 1X2 与大小球赔率去水
- 根据 1X2 与可选大小球概率反推市场 `lambda_home` / `lambda_away`
- 手动输入内部 `lambda_home` / `lambda_away`
- 使用 log-space blending 融合市场 lambda 与内部 lambda
- 应用赛前乘法修正
- 生成 Poisson 比分概率矩阵
- 输出最大概率比分、Top 5 比分、胜平负、大小球、BTTS、置信度和风险提示
- V1 新增 facts-only 赛前情报层：官方名单/首发、伤停、赛程、天气、战意、阵型、叙事热度
- V1 新增 LSI、MII、Narrative Heat、Data Quality 和 Audit Trail
- V1 会过滤预测文章、竞彩推荐、盘口解读、投注建议等噪声来源

V0 会读取亚洲让球字段，但尚未把亚洲让球纳入 lambda 反推；报告中会明确提示这一点。Dixon-Coles 模块已预留，但默认不启用。

## 安装

```bash
python -m pip install -e .[test]
```

如果只是直接从源码运行，也可以在项目根目录设置 `PYTHONPATH=src` 后运行。

## 使用

```bash
python -m score_predictor.cli predict examples/match_input_example.yaml
```

V1 完整输入：

```bash
python -m score_predictor.cli predict examples/match_full_example.yaml
```

V0 比赛输入 + 单独情报文件：

```bash
python -m score_predictor.cli predict examples/match_input_example.yaml --intel examples/match_intel_example.yaml
```

仅输出 JSON：

```bash
python -m score_predictor.cli predict examples/match_input_example.yaml --json-only
```

## 输入示例

见 [examples/match_input_example.yaml](examples/match_input_example.yaml)。

首版最低建议输入：

- 比赛名称与开球时间
- 主客/中立场
- 1X2 十进制赔率
- 大小球盘口与赔率
- 手动内部 lambda
- 预测时间点，例如 `T-24h`、`T-6h`、`T-1h`

## V1 情报层原则

V1 只允许事实信息进入模型：

- 官方大名单、官方首发、球队/赛事/足协公告
- 比赛地点、开球时间、天气、裁判、赛程、休息天数
- 伤停/停赛事实、历史正式比赛结果、球队排名

禁止作为模型输入：

- 赛前预测文章
- 专家推荐
- 竞彩推荐
- 盘口分析
- 比分预测
- 投注建议
- “红单”“稳胆”“爆冷”“串关”等内容

官方首发和官方大名单越完整，`data_quality` 和 `confidence` 越高。友谊赛、轮换、未确认首发、强叙事热度会自动降低置信度或调整 lambda。

## 测试

```bash
pytest
```

## V2: official market feature layer

V2 adds a controlled facts-and-market layer around the existing 90-minute score
engine. It is still for entertainment probability modeling only and is not
betting advice.

V2 principles:

- Do not use prediction articles, betting tips, red-sheet content, score-pick
  articles, or subjective handicap analysis as model input.
- Official Sporttery odds should be manually transcribed first, or entered after
  a screenshot has been checked by a human. The project intentionally does not
  add a brittle automatic Sporttery page crawler.
- Sporttery `rqspf` is treated as official Sporttery handicap win/draw/loss,
  not as an Asian handicap market.
- International odds are optional market references only. They are not FIFA
  official odds; FIFA does not publish official betting odds.
- Every fetched or entered source is represented in the audit trail.

Market-only Sporttery analysis:

```bash
python -m score_predictor.cli market examples/match_v2_sporttery_manual.yaml
```

The market command reports raw implied probabilities, de-vig market implied
probabilities, payout rate, bookmaker margin, hidden return multiplier,
Sporttery total-goals features, correct-score award features, and snapshot
movement when `market_snapshots` are present.

Build a V2 research bundle:

```bash
python -m score_predictor.cli research examples/match_research_config.yaml --write-yaml data/processed/research_bundle.yaml
```

Attach a research bundle to the original predictor without breaking V0/V1
inputs:

```bash
python -m score_predictor.cli predict examples/match_full_example.yaml --research examples/match_research_config.yaml
```

All network-facing tests are mocked. Open-Meteo, official page fetching, and the
optional odds API are designed as controlled connectors with source whitelisting,
prediction/betting-content rejection, warnings, and audit metadata.

## Streamlit UI

安装 UI 依赖：

```bash
pip install -e .[ui]
```

启动：

```bash
streamlit run src/score_predictor/ui/streamlit_app.py
```

UI 支持上传 YAML、手动录入 1X2 / Over-Under / BTTS / Correct Score /
Asian Handicap / rqspf 和 facts-only Team Context，可打开或关闭 Dixon-Coles，
运行 V3 多盘口联合校准预测，并下载输入 YAML、预测 JSON、比分矩阵 CSV 和报告。

## The Odds API 国际盘

The Odds API 只作为 A 源国际盘数据连接器使用，不在 V3 核心模型里直接联网。API key 必须从环境变量读取，不要写入代码、YAML、README、日志或报告。

Windows PowerShell 设置环境变量：

```powershell
[Environment]::SetEnvironmentVariable("THE_ODDS_API_KEY", "your_new_key", "User")
```

检查是否已设置：

```bash
python -c "import os; print('set' if os.getenv('THE_ODDS_API_KEY') else 'missing')"
```

启动 UI：

```bash
python -m streamlit run src\score_predictor\ui\streamlit_app.py
```

UI 流程：

1. 打开 `A源国际盘`。
2. 选择世界杯小组、主队、客队。
3. 确认 `sport_key`、`regions`、`markets`、`bookmaker`。
4. 点击 `查找赛事`。
5. 点击 `拉取赔率`。
6. 点击 `应用到 A源国际盘` 或直接点击 `开始预测`。

命令行生成 V3 兼容 YAML：

```bash
python scripts/fetch_the_odds_api_match.py --home "Korea Republic" --away "Czech Republic" --regions eu,uk --markets h2h,spreads,totals --bookmaker auto --output data/input/generated/korea_czech_the_odds_api.yaml
```

支持的第一版 markets：`h2h`、`spreads`、`totals`。B 源体彩盘仍保留，用于体彩胜平负、让球胜平负、比分固定奖金、总进球和半全场的价值分析。

本工具只做概率建模和娱乐分析，不构成投注建议，不提供投注推荐。最大概率比分不是确定结果。

## 免责声明

本项目只做概率建模和娱乐预测。模型输出不是“正确比分答案”，最大概率比分通常也只有较低的单点概率。请不要把任何输出当作投注建议。
