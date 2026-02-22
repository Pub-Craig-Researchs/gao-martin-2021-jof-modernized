# Volatility, Valuation Ratios, and Bubbles: An Empirical Measure of Market Sentiment

**(The Journal of Finance, 2021, Can Gao & Ian W. R. Martin)**

This project modernizes an empirical replication codebase in Python for the aforementioned academic paper, based on the official implement. The authors construct a forward-looking "Market Sentiment and Bubbles" measure, denoted as $B_t$. This index synthesizes the lower bound of expected returns implied by option market data, risk-free interest rates, and the fundamental equity premium predicted by Valuation Ratios. Pseudo data can be found at JoF.

## Code Structure

Because the project processes a large volume of data involving both high-frequency option data and low-frequency fundamental metrics, the computational workflow is divided into 4 sequential scripts, running in order from `Code_0` to `Code_3`.

- **`Code_0_computing_LVIX.py`**:
  - **Functionality**: Calculates **LVIX**, a measure of the risk-neutral expected price lower bound. It applies standard academic algorithms for extracting implied bounds from options (similar to SVIX), using the Bid-Ask prices of call and put options with the same maturity date.
  - **Input**: Daily constituent stock prices and monthly option quotes located in `/Pseudo_Data`.
  - **Output**: Datasets containing the LVIX metric across various horizons (1 to 24 months) for major indices.

- **`Code_1_full_sample_tables for_Er-g.py`**:
  - **Functionality**: Explores the predictive power of historical valuation ratios (Dividend Yield, $dp_t$, $y_t$) on future realized returns ($r_{t+1}$), dividend growth ($g_{t+1}$), and the excess return over growth ($r_{t+1} - g_{t+1}$) via full-sample Predictive Regressions. It applies Hansen-Hodrick standard errors to correct for serial correlation.
  - **Input**: CRSP fundamental stock market indices and risk-free interest rates.
  - **Output**: Regression tables showing statistical significance.

- **`Code_2_estimating_E[g-r]_and bootstrap.py`**:
  - **Functionality**: The core estimation module. It uses Expanding Windows to extrapolate time-varying conditional expectations $E_t[r_{t+1}-g_{t+1}]$. This script incorporates both linear (ARn) and nonlinear (NLN) models and calculates forecasting bands and Confidence Intervals via Bootstrap resampling.
  - **Output**: Expected future fundamental differential ($r-g$) tables across multiple window sizes and model specifications.

- **`Code_3_comparing to other index.py`**:
  - **Functionality**: Assembles the LVIX and $E[r-g]$ components generated in the prior steps to derive the core bubble sentiment index: $B_t = LVIX_t + Rf_t - E_t[r_{t+1} - g_{t+1}]$.
  - **Validation**: Performs cross-validation with established macroeconomic sentiment and credit indicators (e.g., NFCI / ANFCI / EBP) using Moving Block Bootstrap techniques to examine correlations and lead-lag dynamics.
  - **Output**: Excel files containing significance tests for these correlations.

---

## Bug Fixes for Modern Python/NumPy

The original codebase was authored in an older ecosystem. Rapid iterations of Python versions and core libraries—especially NumPy and Pandas—rendered much of its API usage deprecated or broken. To ensure this project runs seamlessly on **Python 3.12+** and **NumPy 1.24+**, the following critical fixes were applied:

### 1. Hardcoded Path Resolution (`FileNotFoundError`)

* **Bug**: The original scripts universally used hardcoded relative logic like `os.path.dirname(os.path.abspath("Code_xxx.py"))` when loading data. This meant the code would crash with a ``FileNotFoundError`` unless it was executed perfectly from within the exact directory containing the script.
- **Fix**: Replaced the hardcoded strings with dynamic script-path resolution using `os.path.abspath(__file__)`.

### 2. Deprecated NumPy APIs (`AttributeError: module 'numpy' has no attribute...`)

* **Bug 2.1**: The codebase heavily relied on `np.mat(...)` to cast arrays into specific Matrix objects and utilized the `*` operator for matrix multiplication. These matrix objects and operators have been officially deprecated by NumPy, leading to numerous attribute errors and dimension mismatches in modern versions.
- **Bug 2.2**: The function `np.asscalar()`, used to extract a pure scalar, was entirely removed from NumPy for performance and consistency reasons.
- **Bug 2.3**: Direct attribute access via `np.int()` throws an exception in recent versions.
- **Fixes**:
  - All instances of `np.mat` and `*` were substituted with `np.array` and the standard matrix multiplication operator `@`, aligning the code with modern, official array mathematics.
  - Outdated methods like `np.asscalar()` were replaced with `array.item()` to correctly extract scalar values.
  - Eradicated `np.int()`, directly utilizing built-in Python `int()` casting.

### 3. Deprecated Pandas I/O Logic (`AttributeError: 'ExcelWriter' object has no attribute 'save'`)

* **Bug**: Recent iterations of `pandas` entirely deprecated and removed `pandas.ExcelWriter.save()`, enforcing the use of `.close()` which conforms to standard underlying I/O stream paradigms.
- **Fix**: Replaced all `writer.save()` occurrences with `writer.close()`.

### 4. Broadcasting and Array Dimension Mismatches (`ValueError: setting an array element...`)

* **Bug**: In the core pricing module (`Code_0`), if the extracted `spotprice` evaluating to a single-element 1D array rather than a scalar, its use in reciprocal distribution (`np.sum(...) * (1/spotprice)`) produced a full array object whose dimensions exceeded expectations. Assigning this back to a fixed-size receiver matrix (`vix[size, 2]`) raised a shape mismatch error.
- **Fix**: Implemented explicit scalar extraction logic: `sp_val = spotprice[0] if isinstance(...) and spotprice.size > 0 else spotprice`, ensuring the multiplier acts strictly as a constant.

### 5. 1D to 2D Array Alignment (`IndexError: too many indices for array`)

* **Bug**: After rectifying the `np.mat` to `np.array` transition in `Code_2`, a side effect emerged: 1D variables (e.g., `Vec`) passed into prediction functions—previously auto-promoted to 2D by `np.mat`—degraded into standard 1D arrays structured as `(m,)`. Subsequent attempts to execute 2D slicing syntax like `[0, 1]` on these variables triggered an "index out of bounds" error.
- **Fix**: Applied `np.atleast_2d(Vec)` safely at critical junctures to guarantee forced dimensional promotion, making the objects cleanly compatible with downstream indexing and matrix products.

---

## How to Run

1. **Environment Setup**: Install the necessary dependencies: `pip install numpy pandas scipy openpyxl xlsxwriter`
2. Execute the modules sequentially to flawlessly generate all results within the `/Pseudo_results` directory:

   ```bash
   python Code_for_JF_GMdecom/Code_0_computing_LVIX.py
   python Code_for_JF_GMdecom/Code_1_full_sample_tables\ for_Er-g.py
   python Code_for_JF_GMdecom/Code_2_estimating_E[g-r]_and\ bootstrap.py
   python Code_for_JF_GMdecom/Code_3_comparing\ to\ other\ index.py
   ```

