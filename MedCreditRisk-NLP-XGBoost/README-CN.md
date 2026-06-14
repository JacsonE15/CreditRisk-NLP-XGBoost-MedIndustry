# 医药生物行业信用风险预测模型

基于财务数据与年报 NLP 情感分析，预测 A 股医药生物上市公司未来 3 年内是否被 ST 的机器学习模型。

---

## 项目背景

ST（Special Treatment）制度是中国 A 股市场对财务异常或合规违规公司的风险警示机制。本项目尝试通过上市公司历年财务报表数据与年报管理层讨论与分析（MDA）文本的情感特征，提前预警潜在的 ST 风险。

---

## 整体流程

```
财务数据爬取 → 年报PDF下载 → MDA文本提取 → FinBERT情感分析 → 特征工程 → 模型训练 → 评估
```

---

## 文件结构

```
credit_risk_project/
│
├── 1_Download_3_finance_table.ipynb     # 步骤一：爬取财务三张报表 + 行业分类
├── 2_Download_annual_report_and_NLP.ipynb  # 步骤二：下载年报PDF + MDA提取 + NLP分析
├── 3_Training_and_result.ipynb          # 步骤三：特征工程 + 模型训练 + 结果可视化
│
├── master2.csv                          # 最终宽表（财务 + NLP特征合并）
└── BDT_CO.csv                          # 医药生物行业ST事件数据
```

---

## 步骤说明

### Notebook 1：财务数据爬取

**数据来源：** AKShare（东方财富接口）

**爬取内容：**
- 利润表（`stock_lrb_em`）
- 资产负债表（`stock_zcfz_em`）
- 现金流量表（`stock_xjll_em`）
- 申万一级行业分类

**覆盖范围：** 2020–2025 年，全 A 股上市公司年报数据

**筛选条件：** 申万一级行业 = 医药生物，共 451 家公司

**输出：** `master_table.csv`（财务宽表）

---

### Notebook 2：年报下载与 NLP 分析

**年报下载：**
- 通过 AKShare 获取年报 PDF 链接，批量下载（每批约 100 家公司）
- 覆盖 2020–2025 年度年报

**MDA 文本提取：**
- 使用 `pdfplumber` 解析 PDF
- 定位"管理层讨论与分析"章节起止位置
- 自动跳过目录页、扫描版 PDF

**NLP 情感分析：**
- 模型：`yiyanghkust/finbert-tone`（金融领域 FinBERT）
- 输出特征：
  - `negative_score`：负面情绪平均分
  - `negative_score_max`：负面情绪最大值
  - `risk_word_ratio`：风险词频率
  - `uncertainty_ratio`：不确定词频率
  - `mda_length`：MDA 文本长度

**输出：** `master2.csv`（财务 + NLP 特征合并宽表）

---

### Notebook 3：模型训练与评估

#### 标签定义（Y）

```
Y = 1：该年年报报告期结束后 3 年内，公司首次被 ST
Y = 0：3 年内未发生 ST 事件
```

> 排除已被 ST 年份及之后的样本，防止信息泄露。

**样本统计：**
- 总样本：2,168 条
- 正样本（Y=1）：16 条，占比 0.74%
- ST 原因分布：审计否定（8）、两年亏损（3）、信息披露违规（3）、重大诉讼（2）

#### 特征工程

| 类别 | 特征 |
|------|------|
| 规模 | `log_营业总收入`、`log_资产-总资产` |
| 盈利 | `净利润率`、`营业利润率`、`ROE`、`log_净利润` |
| 现金流 | `经营现金流比`、`现金比率`、`毛现金流质量` |
| 偿债 | `资产负债率` |
| 增长 | 净利润/营收/营业利润/总资产同比增长率 |
| 趋势 | 盈利率、ROE、现金流比 t-1 期滞后特征 |
| NLP | `negative_score`、`negative_score_max`、`log_risk_word_ratio`、`log_uncertainty_ratio`、`log_mda_length` |

**数据处理：**
- 无穷值替换为 NaN，全局中位数填充
- 极端值 Winsorize（5%–95%）

#### 验证方法

扩展窗口时序交叉验证（避免未来数据泄露）：

```
Fold1：训练 2020        → 验证 2021
Fold2：训练 2020–2021   → 验证 2022
Fold3：训练 2020–2022   → 验证 2023
Fold4：训练 2020–2023   → 验证 2024
```

#### 模型

- Logistic Regression（`class_weight='balanced'`）
- XGBoost（`scale_pos_weight` 处理不平衡）
- LightGBM（early stopping + `scale_pos_weight`）
- 集成模型（三模型概率均值）

---

## 结果

| 模型 | 均值 ROC-AUC | 均值 PR-AUC |
|------|-------------|------------|
| Logistic Regression | 0.563 | 0.026 |
| XGBoost | **0.880** | **0.121** |
| LightGBM | 0.751 | 0.056 |
| 集成 | 0.728 | 0.037 |

> 随机基线 PR-AUC ≈ 0.007（正样本占比）
> XGBoost PR-AUC 约为随机基线的 **17 倍**

---

## 局限性

1. **正样本极少**：医药生物行业 ST 率低，16 个正样本中各 Fold 验证集仅 2–4 个，结果波动较大，统计可靠性有限
2. **早期 Fold 不稳定**：Fold1 训练集仅 3 个正样本，模型几乎无法有效学习，重点参考 Fold3、Fold4 的结果
3. **ST 原因多样**：审计否定、信息披露违规等事件与财务数据相关性本身较弱，财务特征难以提前捕捉
4. **NLP 模型局限**：FinBERT 为英文金融模型的中文迁移版，对 A 股年报的适配性存在一定偏差
5. **可扩展性**：方法论框架具备扩展性，扩充至全行业后正样本将大幅增加，预计 PR-AUC 可提升至 0.3–0.5

---

## 环境依赖

```bash
pip install akshare pdfplumber transformers torch sentencepiece
pip install pandas numpy scipy scikit-learn xgboost lightgbm matplotlib
```

**运行环境：** Google Colab + Google Drive

---

## 数据说明

- `master2.csv`：451 家医药生物公司 2020–2025 年财务与 NLP 特征宽表，共 2,333 行
- `BDT_CO.csv`：医药生物行业 ST 事件数据，含首次 ST 日期、ST 原因等字段，共 519 条
