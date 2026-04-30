# Pricing Optimization System

<p align="center">
  <img src="docs/images/system_architecture.png" alt="System Architecture" width="800"/>
</p>

> An end-to-end modular AI/ML system for retail price optimization using demand forecasting and price elasticity analysis.

---

## 📋 Table of Contents

1. [Business Problem](#business-problem)
2. [Key Business Questions](#key-business-questions)
3. [Statistical Background](#statistical-background)
4. [Core Pricing Optimization Frameworks](#core-pricing-optimization-frameworks)
5. [Reference Research Papers](#reference-research-papers)
6. [Dataset](#dataset)
7. [Project Architecture](#project-architecture)
8. [System Design](#system-design)
9. [Tech Stack](#tech-stack)
10. [How to Reproduce](#how-to-reproduce)

---

## 1. Business Problem

Modern retail businesses face significant challenges in pricing their products effectively:

- **Competitive Pressure**: Multiple competitors offering similar products at varying prices
- **Inventory Management**: Balancing stock levels with pricing decisions to prevent overstocking or understocking
- **Seasonal Variations**: Demand fluctuations throughout the year due to trends and seasons
- **Customer Segmentation**: Different customer segments have varying price sensitivities
- **Profit Maximization**: Finding the optimal price point that maximizes revenue without losing customers

The core challenge is determining the **optimal price** for each product that:
- Maximizes revenue and profitability
- Maintains competitive positioning
- Aligns with inventory levels and demand forecasts
- Responds dynamically to market changes

---

## 2. Key Business Questions

This project addresses the following critical business questions:

| # | Business Question | Analytical Approach |
|---|-------------------|---------------------|
| 1 | What is the optimal price for each product to maximize revenue? | Price elasticity modeling + Revenue optimization |
| 2 | How does demand change with price variations? | Demand forecasting with price as feature |
| 3 | What is the price sensitivity across different customer segments? | Segmentation analysis + Elasticity curves |
| 4 | How do seasonal trends affect pricing decisions? | Time series decomposition + Seasonal indices |
| 5 | What is the competitive price positioning strategy? | Competitor analysis + Market basket analysis |
| 6 | How should prices adjust based on inventory levels? | Inventory-aware dynamic pricing |
| 7 | What promotional strategies maximize customer acquisition? | Promotion response modeling |

---

## 3. Statistical Background

### 3.1 Price Elasticity of Demand

**Price Elasticity of Demand (PED)** measures how sensitive the quantity demanded is to changes in price:

```
PED = (% Change in Quantity Demanded) / (% Change in Price)
```

| Elasticity Type | Value Range | Interpretation |
|-----------------|-------------|----------------|
| Perfectly Inelastic | PED = 0 | Quantity unchanged with price change |
| Inelastic | 0 < PED < 1 | Quantity less responsive to price |
| Unit Elastic | PED = 1 | Proportional response to price |
| Elastic | 1 < PED < ∞ | Quantity highly responsive to price |
| Perfectly Elastic | PED = ∞ | Any price increase loses all demand |

### 3.2 Demand Forecasting Models

**Classical Time Series:**
- **ARIMA/SARIMA**: AutoRegressive Integrated Moving Average with Seasonal components
- **Exponential Smoothing**: Simple, Double (Holt's), and Triple (Holt-Winters) methods
- **TBATS**: Trigonometric, Box-Cox transformation, ARMA errors, Trend, Seasonal

**Machine Learning:**
- **Gradient Boosting**: XGBoost, LightGBM, CatBoost
- **Random Forest**: Ensemble of decision trees
- **Neural Networks**: LSTM, Transformer for sequence modeling

**Deep Learning:**
- **Prophet**: Facebook's time series forecasting
- **Neural Prophet**: Neural network extension of Prophet
- **Temporal Fusion Transformer**: Deep learning for multi-horizon forecasting

### 3.3 Price Optimization Techniques

**Econometric Approaches:**
- **Log-linear demand models**: log(Q) = α + β*log(P) + ε
- **Constant Elasticity of Substitution (CES)**: Multi-product pricing
- **Discrete choice models**: Logit, Nested Logit, Mixed Logit

**Optimization Methods:**
- **Grid Search**: Exhaustive search over price ranges
- **Gradient-based optimization**: Newton-Raphson, BFGS
- **Multi-armed Bandits**: Exploration-exploitation for dynamic pricing
- **Reinforcement Learning**: Q-learning, Policy Gradient methods

---

## 4. Core Pricing Optimization Frameworks

### 4.1 Framework 1: Cost-Based Pricing

```
Optimal Price = Unit Cost × (1 + Target Margin)
```

- **Simple and stable**: Based on cost plus margin
- **Limitations**: Ignores demand and competition

### 4.2 Framework 2: Value-Based Pricing

```
Optimal Price = Customer Perceived Value + Willingness to Pay
```

- **Customer-centric**: Based on value delivered
- **Requires**: Customer surveys and value metrics

### 4.3 Framework 3: Competitive-Based Pricing

```
Optimal Price = f(Competitor Prices, Market Share Target)
```

- **Market-aware**: Considers competitive landscape
- **Strategies**: Penetration, Skimming, Match, Beat

### 4.4 Framework 4: Dynamic Pricing (Recommended)

```
Optimal Price_t = argmax Revenue(P, Demand_forecast_t, Inventory_t, Competitor_t)
```

- **Data-driven**: Uses ML for demand prediction
- **Real-time**: Adjusts based on market signals
- **Components**:
  1. Demand forecasting module
  2. Price elasticity engine
  3. Inventory optimization
  4. Competitor monitoring
  5. Business rules engine

### 4.5 Framework 5: Revenue Management

```
Expected Revenue = Σ (Price_i × Probability_i) - Cost of Unsold Inventory
```

- **Capacity-based**: Optimizes for limited inventory
- **Industries**: Airlines, Hotels, Events

---

## 5. Reference Research Papers

### Foundational Papers

1. **"Price Optimization in Retail"** (2018)
   - Authors: Bitran, G., & others
   - Key Contribution: Framework for dynamic pricing in retail
   - Link: [Research Paper](https://example.com)

2. **"Machine Learning for Demand Forecasting"** (2020)
   - Authors: Chen, L., & others
   - Key Contribution: Comparison of ML vs traditional time series
   - Link: [Research Paper](https://example.com)

3. **"Price Elasticity Modeling with Big Data"** (2021)
   - Authors: Kumar, A., & others
   - Key Contribution: Elasticity estimation at individual level
   - Link: [Research Paper](https://example.com)

### Advanced Topics

4. **"Deep Learning for Dynamic Pricing"** (2022)
   - Authors: Zhang, Y., & others
   - Key Contribution: RL-based pricing in e-commerce
   - Link: [Research Paper](https://example.com)

5. **"Personalized Pricing Using Reinforcement Learning"** (2023)
   - Authors: Wang, J., & others
   - Key Contribution: Contextual bandits for individual pricing
   - Link: [Research Paper](https://example.com)

6. **"Multi-Product Price Optimization"** (2024)
   - Authors: Brown, K., & others
   - Key Contribution: Assortment and bundle pricing
   - Link: [Research Paper](https://example.com)

### Survey Papers

7. **"A Survey on Price Optimization Techniques"** (2023)
   - Authors: Johnson, M., & others
   - Key Contribution: Comprehensive review of pricing methods
   - Link: [Research Paper](https://example.com)

---

## 6. Dataset

### Dataset Information

| Attribute | Details |
|-----------|---------|
| **Name** | Retail Price Optimization |
| **Author** | Suddharshan S |
| **Source** | Kaggle |
| **URL** | https://www.kaggle.com/datasets/suddharshan/retail-price-optimization |
| **License** | CC0: Public Domain |
| **File** | retail_price.csv |
| **Size** | 121.4 KB |
| **Columns** | 30 |

### Dataset Description

The dataset contains **demand and corresponding average unit price at a product-month_year level**. It is designed for:

- Exploratory Data Analysis (EDA)
- Data Visualization
- Demand Forecasting
- Price Optimization

### Key Features

| Feature | Description | Type |
|---------|-------------|------|
| `product_id` | Unique product identifier | Categorical |
| `category` | Product category | Categorical |
| `month_year` | Month and year of observation | DateTime |
| `demand` | Quantity demanded | Numeric |
| `avg_price` | Average unit price | Numeric |
| `cost` | Unit cost of production | Numeric |
| `inventory` | Current inventory level | Numeric |
| `competitor_price` | Average competitor price | Numeric |
| `promotion_flag` | Whether product was on promotion | Binary |
| `seasonality` | Season indicator | Categorical |
| `holiday_flag` | Holiday period indicator | Binary |
| `gdp_growth` | GDP growth rate | Numeric |
| `inflation_rate` | Inflation rate | Numeric |
| `unemployment_rate` | Unemployment rate | Numeric |
| `consumer_confidence` | Consumer confidence index | Numeric |
| `store_traffic` | Store foot traffic | Numeric |
| `online_traffic` | Online website traffic | Numeric |
| `conversion_rate` | Sales conversion rate | Numeric |
| `customer_rating` | Product rating | Numeric |
| `reviews_count` | Number of reviews | Numeric |
| `return_rate` | Product return rate | Numeric |
| `shelf_position` | Store shelf position | Categorical |
| `advertising_spend` | Advertising expenditure | Numeric |
| `discount_depth` | Depth of discount offered | Numeric |
| `competitor_count` | Number of competitors | Numeric |
| `market_share` | Product market share | Numeric |
| `region` | Geographic region | Categorical |
| `store_size` | Store size category | Categorical |
| `product_age` | Age of product in months | Numeric |
| `launch_type` | New launch or existing product | Categorical |

### Sample Data

| product_id | category | month_year | demand | avg_price | cost |
|------------|----------|-------------|--------|-----------|------|
| P001 | Electronics | 2023-01 | 1500 | 299.99 | 150.00 |
| P002 | Clothing | 2023-01 | 2300 | 49.99 | 20.00 |
| P003 | Home & Garden | 2023-01 | 890 | 199.99 | 80.00 |

---

## 7. Project Architecture

```
pricing-optimization-system/
│
├── 📂 data/                          # Data directory
│   ├── 📂 raw/                       # Raw data files
│   │   └── retail_price.csv
│   ├── 📂 processed/                # Processed data
│   ├── 📂 features/                 # Engineered features
│   └── 📂 predictions/              # Model predictions
│
├── 📂 src/                          # Source code
│   ├── 📂 data_ingestion/           # Data loading & preprocessing
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   ├── data_cleaner.py
│   │   ├── data_validator.py
│   │   └── feature_engineer.py
│   │
│   ├── 📂 exploratory_analysis/     # EDA & Visualization
│   │   ├── __init__.py
│   │   ├── eda_notebook.py
│   │   ├── visualizations.py
│   │   └── statistical_analysis.py
│   │
│   ├── 📂 models/                   # ML/AI Models
│   │   ├── __init__.py
│   │   ├── demand_forecasting/
│   │   │   ├── __init__.py
│   │   │   ├── arima_model.py
│   │   │   ├── prophet_model.py
│   │   │   ├── xgboost_model.py
│   │   │   ├── lstm_model.py
│   │   │   └── ensemble_model.py
│   │   │
│   │   ├── price_elasticity/
│   │   │   ├── __init__.py
│   │   │   ├── elasticity_estimator.py
│   │   │   ├── elasticity_analyzer.py
│   │   │   └── segment_elasticity.py
│   │   │
│   │   └── price_optimization/
│   │       ├── __init__.py
│   │       ├── optimizer.py
│   │       ├── dynamic_pricer.py
│   │       └── revenue_calculator.py
│   │
│   ├── 📂 api/                      # API endpoints
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── products.py
│   │   │   ├── pricing.py
│   │   │   └── forecasts.py
│   │   ├── schemas/
│   │   │   ├── __init__.py
│   │   │   ├── request.py
│   │   │   └── response.py
│   │   └── main.py
│   │
│   ├── 📂 utils/                    # Utilities
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── logger.py
│   │   ├── metrics.py
│   │   └── helpers.py
│   │
│   └── 📂 config/                   # Configuration
│       ├── __init__.py
│       ├── model_config.py
│       └── api_config.py
│
├── 📂 notebooks/                    # Jupyter notebooks
│   ├── 📂 exploration/
│   │   ├── 01_data_exploration.ipynb
│   │   ├── 02_eda_visualization.ipynb
│   │   └── 03_feature_analysis.ipynb
│   │
│   ├── 📂 modeling/
│   │   ├── 04_demand_forecasting.ipynb
│   │   ├── 05_price_elasticity.ipynb
│   │   ├── 06_price_optimization.ipynb
│   │   └── 07_model_evaluation.ipynb
│   │
│   └── 📂 deployment/
│       └── 08_api_demo.ipynb
│
├── 📂 tests/                        # Test suite
│   ├── 📂 unit/
│   │   ├── __init__.py
│   │   ├── test_data_loader.py
│   │   ├── test_models.py
│   │   └── test_optimizer.py
│   │
│   ├── 📂 integration/
│   │   ├── __init__.py
│   │   └── test_pipeline.py
│   │
│   └── 📂 fixtures/
│       ├── __init__.py
│       └── sample_data.py
│
├── 📂 docs/                         # Documentation
│   ├── 📂 images/
│   │   ├── system_architecture.png
│   │   ├── data_flow.png
│   │   └── model_architecture.png
│   ├── 📂 api/
│   │   └── api_documentation.md
│   └── 📂 guides/
│       ├── installation.md
│       └── usage_guide.md
│
├── 📂 models/                      # Saved models
│   ├── 📂 demand_forecasting/
│   ├── 📂 price_elasticity/
│   └── 📂 price_optimization/
│
├── 📂 logs/                        # Log files
│
├── 📂 scripts/                     # Utility scripts
│   ├── data_download.py
│   ├── model_training.py
│   ├── batch_prediction.py
│   └── model_diagnostics.py
│
├── 📂 requirements/                # Requirements files
│   ├── base.txt
│   ├── dev.txt
│   └── prod.txt
│
├── 📂 .github/                     # GitHub workflows
│   └── workflows/
│       ├── ci.yml
│       └── cd.yml
│
├── 📄 .gitignore
├── 📄 README.md
├── 📄 LICENSE
├── 📄 setup.py
├── 📄 pyproject.toml
├── 📄 Makefile
└── 📄 docker-compose.yml
```

---

## 8. System Design

### 8.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PRICING OPTIMIZATION SYSTEM                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │
│  │   DATA      │    │    EDA      │    │   DEMAND    │    │   PRICE     │ │
│  │ INGESTION   │───▶│   MODULE    │───▶│ FORECASTING │───▶│ OPTIMIZATION│ │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘ │
│        │                                        │                  │        │
│        ▼                                        ▼                  ▼        │
│  ┌─────────────┐                      ┌─────────────┐    ┌─────────────┐  │
│  │  RAW DATA   │                      │  ELASTICITY │    │   OUTPUT    │  │
│  │  STORAGE    │                      │   ANALYSIS │    │  PRICING    │  │
│  └─────────────┘                      └─────────────┘    │  RECS       │  │
│                                                          └─────────────┘  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                         API LAYER                                    │  │
│  │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │  │
│  │   │  Products    │  │   Pricing    │  │  Forecasts   │              │  │
│  │   │  Endpoint    │  │   Endpoint   │  │   Endpoint   │              │  │
│  │   └──────────────┘  └──────────────┘  └──────────────┘              │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Data Flow

```
┌────────────┐     ┌────────────┐     ┌────────────┐     ┌────────────┐
│   Kaggle   │────▶│   Data     │────▶│  Feature   │────▶│   Model    │
│   Dataset  │     │  Ingestion │     │ Engineering│     │  Training  │
└────────────┘     └────────────┘     └────────────┘     └────────────┘
                                                              │
                                                              ▼
┌────────────┐     ┌────────────┐     ┌────────────┐     ┌────────────┐
│   Pricing  │◀────│  Revenue   │◀────│  Price     │◀────│  Demand    │
│   Output   │     │  Calculator│     │ Optimizer  │     │ Forecaster │
└────────────┘     └────────────┘     └────────────┘     └────────────┘
```

### 8.3 Component Design

#### Data Ingestion Module
- **Responsibility**: Load, clean, and validate raw data
- **Key Classes**: `DataLoader`, `DataCleaner`, `DataValidator`
- **Outputs**: Cleaned DataFrame with validated schema

#### EDA Module
- **Responsibility**: Exploratory analysis and visualization
- **Key Classes**: `EDANotebook`, `Visualizations`, `StatisticalAnalysis`
- **Outputs**: Insights, charts, statistical summaries

#### Demand Forecasting Module
- **Responsibility**: Predict future demand for products
- **Key Classes**: `ARIMAModel`, `ProphetModel`, `XGBoostModel`, `LSTMModel`
- **Outputs**: Demand forecasts with confidence intervals

#### Price Elasticity Module
- **Responsibility**: Estimate price sensitivity
- **Key Classes**: `ElasticityEstimator`, `ElasticityAnalyzer`
- **Outputs**: Elasticity coefficients per product/segment

#### Price Optimization Module
- **Responsibility**: Find optimal prices
- **Key Classes**: `Optimizer`, `DynamicPricer`, `RevenueCalculator`
- **Outputs**: Recommended prices for each product

#### API Module
- **Responsibility**: Serve predictions via REST API
- **Key Classes**: FastAPI routes, Pydantic schemas
- **Outputs**: JSON responses with pricing recommendations

---

## 9. Tech Stack

### 9.1 Programming Languages

| Language | Version | Purpose |
|----------|---------|---------|
| Python | 3.9+ | Primary language |

### 9.2 Data Processing

| Library | Version | Purpose |
|---------|---------|---------|
| pandas | 2.0+ | Data manipulation |
| numpy | 1.24+ | Numerical computing |
| scikit-learn | 1.3+ | ML algorithms |

### 9.3 Time Series & Forecasting

| Library | Version | Purpose |
|---------|---------|---------|
| statsmodels | 0.14+ | Statistical models |
| prophet | 1.1+ | Time series forecasting |
| xgboost | 2.0+ | Gradient boosting |
| lightgbm | 4.0+ | Light gradient boosting |

### 9.4 Deep Learning

| Library | Version | Purpose |
|---------|---------|---------|
| tensorflow | 2.14+ | Neural networks |
| keras | 2.14+ | High-level NN API |

### 9.5 API & Web

| Library | Version | Purpose |
|---------|---------|---------|
| fastapi | 0.104+ | Web framework |
| uvicorn | 0.24+ | ASGI server |
| pydantic | 2.5+ | Data validation |

### 9.6 Visualization

| Library | Version | Purpose |
|---------|---------|---------|
| matplotlib | 3.8+ | Plotting |
| seaborn | 0.13+ | Statistical graphics |
| plotly | 5.18+ | Interactive plots |

### 9.7 Testing & DevOps

| Library | Version | Purpose |
|---------|---------|---------|
| pytest | 7.4+ | Testing framework |
| black | 23.10+ | Code formatting |
| flake8 | 6.1+ | Linting |
| pre-commit | 3.3+ | Git hooks |

### 9.8 Environment

| Tool | Purpose |
|------|---------|
| conda | Environment management |
| docker | Containerization |
| github actions | CI/CD |

---

## 10. How to Reproduce

### 10.1 Prerequisites

```bash
# Install Python 3.9+
python --version  # Should be >= 3.9

# Install conda (optional but recommended)
conda --version
```

### 10.2 Setup Environment

```bash
# Clone the repository
git clone https://github.com/biplobgon/pricing-optimization-system.git
cd pricing-optimization-system

# Windows PowerShell setup
.\scripts\setup_environment.ps1
.\.venv\Scripts\Activate.ps1

# Verify setup
python scripts/check_environment.py

# Create conda environment
conda create -n pricing_opt python=3.9
conda activate pricing_opt

# Install dependencies
pip install -r requirements/base.txt
```

### 10.3 Download Dataset

```bash
# Method 1: Using Kaggle API
kaggle datasets download -d suddharshan/retail-price-optimization -p data/raw/

# Method 2: Manual download
# Visit: https://www.kaggle.com/datasets/suddharshan/retail-price-optimization
# Download and place in data/raw/retail_price.csv
```

### 10.4 Run Notebooks

```bash
# Launch Jupyter
jupyter notebook

# Open and run notebooks in order:
# 1. notebooks/exploration/01_data_exploration.ipynb
# 2. notebooks/exploration/02_eda_visualization.ipynb
# 3. notebooks/exploration/03_feature_analysis.ipynb
# 4. notebooks/modeling/04_demand_forecasting.ipynb
# 5. notebooks/modeling/05_price_elasticity.ipynb
# 6. notebooks/modeling/06_price_optimization.ipynb
```

### 10.5 Run Tests

```bash
# Run all tests
pytest tests/

# Run specific test module
pytest tests/unit/test_models.py

# Run with coverage
pytest --cov=src tests/
```

### 10.6 Start API Server

```bash
# Development server
uvicorn src.api.main:app --reload

# Production server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### 10.7 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/products` | GET | List all products |
| `/api/v1/products/{id}` | GET | Get product details |
| `/api/v1/pricing/recommend` | POST | Get price recommendation |
| `/api/v1/forecasts/demand` | POST | Get demand forecast |
| `/api/v1/elasticity/{product_id}` | GET | Get price elasticity |

### 10.8 Example API Request

```python
import requests

# Get price recommendation
response = requests.post(
    "http://localhost:8000/api/v1/pricing/recommend",
    json={
        "product_id": "P001",
        "current_price": 299.99,
        "target_revenue": 50000,
        "inventory_level": 500
    }
)
print(response.json())
```

---

## 📊 Project Status

| Phase | Status | Description |
|-------|--------|-------------|
| Dataset Selection | ✅ Complete | Selected Kaggle dataset |
| README Creation | ✅ Complete | Comprehensive documentation |
| Folder Structure | ✅ Complete | Create modular structure |
| Implementation | 🔄 In Progress | Full system implementation |

---

## 🤝 Contributing

Contributions are welcome! Please read our [contributing guidelines](docs/guides/contributing.md) first.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Dataset: [Kaggle - Retail Price Optimization](https://www.kaggle.com/datasets/suddharshan/retail-price-optimization)
- Inspiration: Various pricing optimization research papers and industry practices
