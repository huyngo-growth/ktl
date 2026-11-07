# -*- coding: utf-8 -*-
"""Bài thực hành: mô tả, tương quan, VIF, winsorization và OLS có tương tác.

Chạy từ thư mục dự án bằng: py main.py
Chỉ dùng pandas và numpy; sai số chuẩn gom cụm theo doanh nghiệp được tính
bằng công thức sandwich, p-value dùng xấp xỉ chuẩn lớn mẫu.
"""

from __future__ import annotations

import math
import sys
from html import escape as xml_escape
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data_source"
OUT = ROOT / "output"
CHARTS = OUT / "charts"
YEARS = list(range(2021, 2026))
LOWER_P, UPPER_P = 0.01, 0.99

VAR_LABELS = {
    "asset_turnover": "Hiệu quả hoạt động (doanh thu thuần / tổng tài sản)",
    "payables_ratio": "Phải trả người bán / tổng tài sản",
    "liquidity": "Thanh khoản hiện hành (tài sản ngắn hạn / nợ ngắn hạn)",
    "size_ln": "Quy mô (ln tổng tài sản, tỷ đồng)",
}

# Ký hiệu ba chữ dùng riêng cho bảng/heatmap tương quan để tên biến gọn.
SHORT_LABELS = {
    "asset_turnover": "ATO",          # Asset turnover
    "payables_ratio": "TPR",          # Trade payables ratio
    "liquidity": "LIQ",               # Liquidity/current ratio
    "size_ln": "SIZ",                 # Firm size
    "high_liquidity_dummy": "DLI",    # Dummy liquidity
    "interaction": "TLI",             # TPR x LIQ
    "interaction_dummy": "TLD",       # TPR x DLI
}

# Mã ngắn dùng thống nhất trong toàn bộ bảng kết quả.
RESULT_LABELS = {
    **SHORT_LABELS,
    "payables_c": "TPR",
    "liquidity_c": "LIQ",
    "Hằng số": "CON",
    "Năm_2022": "Y22",
    "Năm_2023": "Y23",
    "Năm_2024": "Y24",
    "Năm_2025": "Y25",
}


def classify_statement_group(frame: pd.DataFrame) -> pd.Series:
    """Phân nhóm theo mẫu biểu BCTC Vietcap, không đoán ngành từ tên/mã.

    Các cột isb/isi/iss lần lượt là mẫu KQKD riêng cho ngân hàng, bảo hiểm
    và công ty chứng khoán trong README nguồn. Nguồn điền 0 vào cột không áp
    dụng, vì vậy chỉ xem là thuộc nhóm khi có ít nhất một giá trị khác 0.
    """
    result = pd.Series("Phi tài chính (mẫu biểu chung)", index=frame.index, dtype="object")
    prefix_groups = [
        ("Bảo hiểm", "isi"),
        ("Chứng khoán", "iss"),
        ("Ngân hàng", "isb"),
    ]
    for group_name, prefix in prefix_groups:
        cols = [c for c in frame.columns if c.startswith(prefix)]
        if not cols:
            continue
        numeric = frame[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        has_special_statement = numeric.abs().gt(1e-12).any(axis=1)
        result.loc[has_special_statement] = group_name
    return result


def fmt(value: float, digits: int = 4) -> str:
    if pd.isna(value):
        return "—"
    return f"{value:,.{digits}f}"


def table_html(frame: pd.DataFrame, digits: int = 4, text: bool = False) -> str:
    shown = frame.copy()
    count_cols = {"Obs", "Số quan sát", "Số mã", "Số công ty", "Dưới p1", "Trên p99",
                  "Tổng giá trị bị chặn"}
    for col in shown.columns:
        if pd.api.types.is_numeric_dtype(shown[col]):
            col_digits = 0 if (col in count_cols or pd.api.types.is_integer_dtype(shown[col])) else digits
            shown[col] = shown[col].map(lambda x, d=col_digits: fmt(x, d))
    return shown.to_html(index=True, border=0, classes="data text" if text else "data", escape=True)


def descriptive(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = {}
    for col in columns:
        x = pd.to_numeric(frame[col], errors="coerce").dropna()
        rows[col] = {
            "Obs": int(x.count()), "Mean": x.mean(), "SD": x.std(ddof=1),
            "Min": x.min(), "p25": x.quantile(.25), "p50": x.quantile(.50),
            "p75": x.quantile(.75), "Max": x.max(), "Skewness": x.skew(),
        }
    result = pd.DataFrame.from_dict(rows, orient="index")
    result.index = [SHORT_LABELS.get(x, x) for x in result.index]
    return result


def vif_table(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """VIF_j = 1/(1-R_j^2), R_j^2 từ hồi quy X_j lên các X còn lại."""
    values = frame[columns].astype(float).to_numpy()
    rows = []
    for j, name in enumerate(columns):
        y = values[:, j]
        others = np.delete(values, j, axis=1)
        design = np.column_stack([np.ones(len(y)), others])
        fitted = design @ np.linalg.lstsq(design, y, rcond=None)[0]
        tss = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 if tss <= 1e-24 else 1 - float(np.sum((y - fitted) ** 2)) / tss
        vif = np.inf if r2 >= 1 - 1e-12 else 1 / (1 - r2)
        rows.append({"Biến giải thích": name, "R2 phụ": r2, "VIF": vif})
    return pd.DataFrame(rows).set_index("Biến giải thích")


def ols_clustered(frame: pd.DataFrame, regressors: list[str], model_name: str,
                  include_year_effects: bool = True, within: bool = False):
    """OLS, sai số chuẩn cluster theo ticker; p-value xấp xỉ chuẩn lớn mẫu.

    within=True: hiệu ứng cố định công ty (FE) bằng phép biến đổi within, tức
    trừ trung bình của từng công ty khỏi Y và mọi biến X (kể cả dummy năm).
    Công ty chỉ có 1 năm không đóng góp biến động within nên được bỏ ra.
    """
    if within:
        frame = frame[frame.groupby("ticker")["ticker"].transform("size") > 1]
    blocks = [frame[regressors].astype(float).reset_index(drop=True)]
    if include_year_effects:
        years = pd.get_dummies(frame["year"].astype(int), prefix="Năm", drop_first=True, dtype=float)
        blocks.append(years.reset_index(drop=True))
    xdf = pd.concat(blocks, axis=1)
    y_series = frame["asset_turnover"].astype(float).reset_index(drop=True)
    if within:
        firm = frame["ticker"].reset_index(drop=True)
        xdf = xdf - xdf.groupby(firm).transform("mean")
        y_series = y_series - y_series.groupby(firm).transform("mean")
    else:
        xdf.insert(0, "Hằng số", 1.0)
    x = xdf.to_numpy(dtype=float)
    y = y_series.to_numpy(dtype=float)
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ beta
    bread = np.linalg.pinv(x.T @ x)

    # Ma trận phương sai sandwich với cụm là công ty: cho phép sai số
    # trong cùng mã cổ phiếu tương quan qua các năm.
    meat = np.zeros((x.shape[1], x.shape[1]), dtype=float)
    groups = frame["ticker"].to_numpy()
    for ticker in pd.unique(groups):
        idx = groups == ticker
        score = x[idx].T @ residual[idx]
        meat += np.outer(score, score)
    n, k = x.shape
    cluster_count = pd.Series(groups).nunique()
    if cluster_count > 1 and n > k:
        correction = (cluster_count / (cluster_count - 1)) * ((n - 1) / (n - k))
    else:
        correction = 1.0
    covariance = correction * bread @ meat @ bread
    se = np.sqrt(np.maximum(np.diag(covariance), 0))
    t_stats = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    # Với hơn 1.000 cụm, xấp xỉ chuẩn là phù hợp cho báo cáo bài tập.
    p_values = np.array([math.erfc(abs(float(t)) / math.sqrt(2)) if np.isfinite(t) else np.nan
                         for t in t_stats])
    ci_low, ci_high = beta - 1.96 * se, beta + 1.96 * se
    result = pd.DataFrame({
        "Hệ số OLS": beta, "SE gom cụm": se, "t/z": t_stats,
        "p-value xấp xỉ": p_values, "CI 95% thấp": ci_low, "CI 95% cao": ci_high,
    }, index=xdf.columns)
    fitted = x @ beta
    tss = float(np.sum((y - y.mean()) ** 2))
    rss = float(np.sum(residual ** 2))
    r2 = 1 - rss / tss if tss else np.nan
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k) if n > k else np.nan
    info = {"Mô hình": model_name, "Obs": n, "Số công ty": cluster_count,
            "R2": r2, "R2 hiệu chỉnh": adj_r2, "RSS": rss}
    return result, info, residual


def normality_diagnostics(series_map: dict[str, pd.Series | np.ndarray]) -> pd.DataFrame:
    """Skewness, Pearson kurtosis, CV và Jarque–Bera (H0: phân phối chuẩn)."""
    rows = []
    for name, values in series_map.items():
        x = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        n = len(x)
        mean = float(np.mean(x))
        sample_sd = float(np.std(x, ddof=1)) if n > 1 else np.nan
        centered = x - mean
        m2 = float(np.mean(centered ** 2))
        if n < 3 or m2 <= 0:
            skewness, kurtosis, jb, p_value = np.nan, np.nan, np.nan, np.nan
        else:
            m3 = float(np.mean(centered ** 3))
            m4 = float(np.mean(centered ** 4))
            skewness = m3 / (m2 ** 1.5)
            kurtosis = m4 / (m2 ** 2)  # Pearson kurtosis: chuẩn = 3
            jb = n / 6 * (skewness ** 2 + ((kurtosis - 3) ** 2) / 4)
            # Với chi-square 2 bậc tự do, survival function = exp(-JB/2).
            p_value = math.exp(-jb / 2) if jb < 1500 else 0.0
        cv = sample_sd / abs(mean) * 100 if abs(mean) > 1e-12 and np.isfinite(sample_sd) else np.nan
        conclusion = ("Không bác bỏ H0 chuẩn" if np.isfinite(p_value) and p_value >= .05
                      else "Bác bỏ H0 chuẩn")
        rows.append({"Biến số": name, "Obs": n, "Skewness": skewness,
                     "Kurtosis (chuẩn=3)": kurtosis, "CV (%)": cv,
                     "JB statistic": jb, "JB p-value": p_value,
                     "Kết luận ở mức 5%": conclusion})
    return pd.DataFrame(rows).set_index("Biến số")


def _scale(value: float, low: float, high: float, start: float, end: float) -> float:
    if high <= low:
        return (start + end) / 2
    return start + (value - low) * (end - start) / (high - low)


def _padded(values, include_zero: bool = False) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    low, high = float(np.nanmin(values)), float(np.nanmax(values))
    if include_zero:
        low = min(low, 0.0)
        high = max(high, 0.0)
    pad = (high - low) * 0.12
    if pad == 0:
        pad = max(abs(high) * 0.12, 0.1)
    return low - pad, high + pad


def chart_year_means(year_means: pd.DataFrame, destination: Path) -> None:
    """Biểu đồ hai trục kiểu Stata: phải trả và hiệu quả trung bình theo năm."""
    width, height = 1000, 580
    left, right, top, bottom = 95, 905, 75, 470
    years = year_means.index.astype(int).tolist()
    payable = (year_means["payables_ratio"] * 100).to_numpy()
    turnover = year_means["asset_turnover"].to_numpy()
    y1_low, y1_high = _padded(payable)
    y2_low, y2_high = _padded(turnover)
    x_positions = np.linspace(left, right, len(years))
    blue, red, grid, ink = "#245b86", "#b44c4c", "#d9e0e7", "#263746"
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
             '<title id="title">Tỷ lệ phải trả và hiệu quả hoạt động trung bình theo năm</title>',
             '<desc id="desc">Biểu đồ hai trục từ năm 2021 đến 2025, đường xanh là phải trả người bán trên tổng tài sản, đường đỏ là vòng quay tổng tài sản.</desc>',
             '<rect width="100%" height="100%" fill="#ffffff"/>',
             f'<text x="{width/2}" y="34" text-anchor="middle" font-family="Arial" font-size="22" fill="{blue}">Phải trả người bán và hiệu quả hoạt động theo năm</text>',
             f'<rect x="{left}" y="{top}" width="{right-left}" height="{bottom-top}" fill="#ffffff" stroke="#66727e" stroke-width="1"/>']
    for idx in range(6):
        frac = idx / 5
        y = bottom - frac * (bottom - top)
        v1 = y1_low + frac * (y1_high - y1_low)
        v2 = y2_low + frac * (y2_high - y2_low)
        parts += [f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="{grid}"/>',
                  f'<text x="{left-12}" y="{y+5:.1f}" text-anchor="end" font-family="Arial" font-size="13" fill="{ink}">{v1:.2f}%</text>',
                  f'<text x="{right+12}" y="{y+5:.1f}" text-anchor="start" font-family="Arial" font-size="13" fill="{ink}">{v2:.3f}</text>']
    points1, points2 = [], []
    for x, year, p, t in zip(x_positions, years, payable, turnover):
        y1 = _scale(p, y1_low, y1_high, bottom, top)
        y2 = _scale(t, y2_low, y2_high, bottom, top)
        points1.append(f"{x:.1f},{y1:.1f}")
        points2.append(f"{x:.1f},{y2:.1f}")
        parts += [f'<text x="{x:.1f}" y="{bottom+28}" text-anchor="middle" font-family="Arial" font-size="14" fill="{ink}">{year}</text>',
                  f'<circle cx="{x:.1f}" cy="{y1:.1f}" r="5" fill="{blue}"/>',
                  f'<polygon points="{x:.1f},{y2-7:.1f} {x+7:.1f},{y2:.1f} {x:.1f},{y2+7:.1f} {x-7:.1f},{y2:.1f}" fill="{red}"/>']
    parts += [f'<polyline points="{" ".join(points1)}" fill="none" stroke="{blue}" stroke-width="2.2"/>',
              f'<polyline points="{" ".join(points2)}" fill="none" stroke="{red}" stroke-width="2.2"/>',
              f'<text transform="translate(24 {((top+bottom)/2):.1f}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="15" fill="{ink}">Phải trả người bán / tổng tài sản (%)</text>',
              f'<text transform="translate(980 {((top+bottom)/2):.1f}) rotate(90)" text-anchor="middle" font-family="Arial" font-size="15" fill="{ink}">Vòng quay tổng tài sản (lần)</text>',
              f'<text x="{(left+right)/2}" y="{bottom+56}" text-anchor="middle" font-family="Arial" font-size="15" fill="{ink}">Năm tài chính</text>',
              f'<line x1="285" y1="548" x2="330" y2="548" stroke="{blue}" stroke-width="2.2"/><circle cx="307" cy="548" r="5" fill="{blue}"/><text x="340" y="553" font-family="Arial" font-size="14" fill="{ink}">Phải trả / tổng tài sản</text>',
              f'<line x1="575" y1="548" x2="620" y2="548" stroke="{red}" stroke-width="2.2"/><polygon points="597,541 604,548 597,555 590,548" fill="{red}"/><text x="630" y="553" font-family="Arial" font-size="14" fill="{ink}">Hiệu quả hoạt động</text>',
              '</svg>']
    destination.write_text("\n".join(parts), encoding="utf-8")


def chart_descriptive(desc: pd.DataFrame, destination: Path) -> None:
    """Bốn dải thống kê riêng thang đo: min–max, p25–p75, mean và median."""
    width, height = 1000, 670
    plot_left, plot_right = 325, 925
    short_names = ["Vòng quay tổng tài sản", "Phải trả người bán / tài sản",
                   "Thanh khoản hiện hành", "Quy mô: ln(tổng tài sản)"]
    blue, red, ink, pale = "#245b86", "#b44c4c", "#263746", "#d9e0e7"
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
             '<title id="title">Thống kê mô tả sau Winsor2</title>',
             '<desc id="desc">Mỗi biến có thang đo riêng. Đường mảnh là min đến max, đường đậm là p25 đến p75, chấm tròn là trung vị và hình thoi là trung bình.</desc>',
             '<rect width="100%" height="100%" fill="#ffffff"/>',
             f'<text x="{width/2}" y="36" text-anchor="middle" font-family="Arial" font-size="22" fill="{blue}">Thống kê mô tả sau Winsor2 (p1–p99)</text>']
    for i, (idx, row) in enumerate(desc.iterrows()):
        y = 120 + i * 130
        low, high = float(row["Min"]), float(row["Max"])
        p25, median, mean, p75 = map(float, [row["p25"], row["p50"], row["Mean"], row["p75"]])
        x_min, x_max = plot_left, plot_right
        xp25 = _scale(p25, low, high, x_min, x_max)
        xp50 = _scale(median, low, high, x_min, x_max)
        xmean = _scale(mean, low, high, x_min, x_max)
        xp75 = _scale(p75, low, high, x_min, x_max)
        parts += [f'<text x="30" y="{y+5}" font-family="Arial" font-size="15" fill="{ink}">{xml_escape(short_names[i])}</text>',
                  f'<line x1="{x_min}" y1="{y}" x2="{x_max}" y2="{y}" stroke="{pale}" stroke-width="4"/>',
                  f'<line x1="{xp25:.1f}" y1="{y}" x2="{xp75:.1f}" y2="{y}" stroke="{blue}" stroke-width="12"/>',
                  f'<circle cx="{xp50:.1f}" cy="{y}" r="7" fill="#ffffff" stroke="{blue}" stroke-width="3"/>',
                  f'<polygon points="{xmean:.1f},{y-9} {xmean+9:.1f},{y} {xmean:.1f},{y+9} {xmean-9:.1f},{y}" fill="{red}"/>',
                  f'<text x="{(x_min+x_max)/2}" y="{y+32}" text-anchor="middle" font-family="Arial" font-size="12" fill="{ink}">Min {low:.3f} · p25 {p25:.3f} · p50 {median:.3f} · Mean {mean:.3f} · p75 {p75:.3f} · Max {high:.3f}</text>']
    parts += [f'<line x1="235" y1="625" x2="280" y2="625" stroke="{blue}" stroke-width="12"/><text x="290" y="630" font-family="Arial" font-size="13" fill="{ink}">Khoảng p25–p75</text>',
              f'<circle cx="455" cy="625" r="7" fill="#ffffff" stroke="{blue}" stroke-width="3"/><text x="470" y="630" font-family="Arial" font-size="13" fill="{ink}">Trung vị</text>',
              f'<polygon points="620,616 629,625 620,634 611,625" fill="{red}"/><text x="638" y="630" font-family="Arial" font-size="13" fill="{ink}">Trung bình</text>',
              f'<text x="{width/2}" y="657" text-anchor="middle" font-family="Arial" font-size="12" fill="#5f6c78">Mỗi dòng dùng thang đo riêng; không so sánh độ dài giữa các dòng.</text>', '</svg>']
    destination.write_text("\n".join(parts), encoding="utf-8")


def chart_interaction(frame: pd.DataFrame, ols_result: pd.DataFrame, destination: Path) -> pd.DataFrame:
    """Margins-style plot: dự báo Y theo X ở ba mức thanh khoản."""
    width, height = 1000, 610
    left, right, top, bottom = 95, 930, 75, 470
    x_values = np.linspace(frame["payables_ratio"].quantile(.10),
                           frame["payables_ratio"].quantile(.90), 41)
    liquidity_levels = frame["liquidity"].quantile([.25, .50, .75])
    b = ols_result["Hệ số OLS"]
    mean_ap = frame["payables_ratio"].mean()
    mean_liq = frame["liquidity"].mean()
    mean_size = frame["size_ln"].mean()
    rows = []
    labels = {0.25: "Thanh khoản thấp (p25)", 0.5: "Thanh khoản trung vị (p50)",
              0.75: "Thanh khoản cao (p75)"}
    for q, liq in liquidity_levels.items():
        for x in x_values:
            ap_c, liq_c = x - mean_ap, liq - mean_liq
            pred = (b["Hằng số"] + b["payables_c"] * ap_c + b["liquidity_c"] * liq_c +
                    b["size_ln"] * mean_size + b["interaction"] * ap_c * liq_c)
            rows.append({"payables_ratio": x, "liquidity_quantile": q,
                         "liquidity": liq, "predicted_asset_turnover": pred,
                         "label": labels[float(q)]})
    pred_df = pd.DataFrame(rows)
    y_low, y_high = _padded(pred_df["predicted_asset_turnover"])
    x_low, x_high = float(x_values.min()), float(x_values.max())
    colors = {0.25: "#8a5a9e", 0.5: "#245b86", 0.75: "#b44c4c"}
    ink, grid = "#263746", "#d9e0e7"
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
             '<title id="title">Biểu đồ tương tác giữa phải trả người bán và thanh khoản</title>',
             '<desc id="desc">Dự báo vòng quay tài sản theo tỷ lệ phải trả người bán ở ba mức thanh khoản, giữ quy mô ở trung bình và năm gốc 2021.</desc>',
             '<rect width="100%" height="100%" fill="#ffffff"/>',
             '<text x="500" y="34" text-anchor="middle" font-family="Arial" font-size="22" fill="#245b86">Tác động tương tác: phải trả người bán × thanh khoản</text>',
             f'<rect x="{left}" y="{top}" width="{right-left}" height="{bottom-top}" fill="#ffffff" stroke="#66727e" stroke-width="1"/>']
    for idx in range(6):
        frac = idx / 5
        y = bottom - frac * (bottom - top)
        value = y_low + frac * (y_high - y_low)
        parts += [f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="{grid}"/>',
                  f'<text x="{left-12}" y="{y+5:.1f}" text-anchor="end" font-family="Arial" font-size="13" fill="{ink}">{value:.2f}</text>']
    for idx in range(6):
        frac = idx / 5
        x = left + frac * (right - left)
        value = (x_low + frac * (x_high - x_low)) * 100
        parts += [f'<text x="{x:.1f}" y="{bottom+28}" text-anchor="middle" font-family="Arial" font-size="13" fill="{ink}">{value:.1f}%</text>']
    legend_x = [175, 430, 690]
    for pos, (q, group) in zip(legend_x, pred_df.groupby("liquidity_quantile", sort=True)):
        color = colors[float(q)]
        points = []
        for _, row in group.iterrows():
            x = _scale(row["payables_ratio"], x_low, x_high, left, right)
            y = _scale(row["predicted_asset_turnover"], y_low, y_high, bottom, top)
            points.append(f"{x:.1f},{y:.1f}")
        parts += [f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2.5"/>',
                  f'<line x1="{pos}" y1="535" x2="{pos+38}" y2="535" stroke="{color}" stroke-width="2.5"/>',
                  f'<circle cx="{pos+19}" cy="535" r="4" fill="{color}"/>',
                  f'<text x="{pos+47}" y="540" font-family="Arial" font-size="13" fill="{ink}">{xml_escape(group.iloc[0]["label"])}</text>']
    parts += [f'<text transform="translate(25 {(top+bottom)/2}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="15" fill="{ink}">Vòng quay tổng tài sản dự báo (lần)</text>',
              f'<text x="{(left+right)/2}" y="{bottom+58}" text-anchor="middle" font-family="Arial" font-size="15" fill="{ink}">Phải trả người bán / tổng tài sản</text>',
              f'<text x="500" y="585" text-anchor="middle" font-family="Arial" font-size="12" fill="#5f6c78">Dự báo OLS tại quy mô trung bình, năm gốc 2021; trục X giới hạn từ p10 đến p90.</text>', '</svg>']
    destination.write_text("\n".join(parts), encoding="utf-8")
    return pred_df


def chart_dummy_interaction(frame: pd.DataFrame, ols_result: pd.DataFrame,
                            destination: Path) -> pd.DataFrame:
    """Margins-style plot cho dummy thanh khoản 0/1."""
    width, height = 1000, 600
    left, right, top, bottom = 95, 930, 75, 460
    x_values = np.linspace(frame["payables_ratio"].quantile(.10),
                           frame["payables_ratio"].quantile(.90), 41)
    b = ols_result["Hệ số OLS"]
    mean_ap = frame["payables_ratio"].mean()
    mean_size = frame["size_ln"].mean()
    rows = []
    labels = {0: "D=0: tài sản ngắn hạn < nợ ngắn hạn",
              1: "D=1: tài sản ngắn hạn ≥ nợ ngắn hạn"}
    for dummy in [0, 1]:
        for x in x_values:
            ap_c = x - mean_ap
            pred = (b["Hằng số"] + b["payables_c"] * ap_c +
                    b["high_liquidity_dummy"] * dummy + b["size_ln"] * mean_size +
                    b["interaction_dummy"] * ap_c * dummy)
            rows.append({"payables_ratio": x, "high_liquidity_dummy": dummy,
                         "predicted_asset_turnover": pred, "label": labels[dummy]})
    pred_df = pd.DataFrame(rows)
    y_low, y_high = _padded(pred_df["predicted_asset_turnover"])
    x_low, x_high = float(x_values.min()), float(x_values.max())
    colors = {0: "#8a5a9e", 1: "#245b86"}
    ink, grid = "#263746", "#d9e0e7"
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
             '<title id="title">Tương tác giữa phải trả người bán và dummy thanh khoản</title>',
             '<desc id="desc">Hai đường dự báo vòng quay tổng tài sản cho nhóm có và không có đủ tài sản ngắn hạn trang trải nợ ngắn hạn.</desc>',
             '<rect width="100%" height="100%" fill="#ffffff"/>',
             '<text x="500" y="34" text-anchor="middle" font-family="Arial" font-size="22" fill="#245b86">Tương tác với biến giả thanh khoản 0–1</text>',
             f'<rect x="{left}" y="{top}" width="{right-left}" height="{bottom-top}" fill="#ffffff" stroke="#66727e" stroke-width="1"/>']
    for idx in range(6):
        frac = idx / 5
        y = bottom - frac * (bottom - top)
        value = y_low + frac * (y_high - y_low)
        parts += [f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="{grid}"/>',
                  f'<text x="{left-12}" y="{y+5:.1f}" text-anchor="end" font-family="Arial" font-size="13" fill="{ink}">{value:.2f}</text>']
    for idx in range(6):
        frac = idx / 5
        x = left + frac * (right - left)
        value = (x_low + frac * (x_high - x_low)) * 100
        parts.append(f'<text x="{x:.1f}" y="{bottom+28}" text-anchor="middle" font-family="Arial" font-size="13" fill="{ink}">{value:.1f}%</text>')
    for dummy, group in pred_df.groupby("high_liquidity_dummy", sort=True):
        color = colors[int(dummy)]
        points = []
        for _, row in group.iterrows():
            x = _scale(row["payables_ratio"], x_low, x_high, left, right)
            y = _scale(row["predicted_asset_turnover"], y_low, y_high, bottom, top)
            points.append(f"{x:.1f},{y:.1f}")
        parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2.5"/>')
    parts += [f'<text transform="translate(25 {(top+bottom)/2}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="15" fill="{ink}">Vòng quay tổng tài sản dự báo (lần)</text>',
              f'<text x="{(left+right)/2}" y="{bottom+58}" text-anchor="middle" font-family="Arial" font-size="15" fill="{ink}">Phải trả người bán / tổng tài sản</text>',
              f'<line x1="155" y1="535" x2="195" y2="535" stroke="{colors[0]}" stroke-width="2.5"/><text x="205" y="540" font-family="Arial" font-size="13" fill="{ink}">{xml_escape(labels[0])}</text>',
              f'<line x1="550" y1="535" x2="590" y2="535" stroke="{colors[1]}" stroke-width="2.5"/><text x="600" y="540" font-family="Arial" font-size="13" fill="{ink}">{xml_escape(labels[1])}</text>',
              '<text x="500" y="580" text-anchor="middle" font-family="Arial" font-size="12" fill="#5f6c78">Dự báo OLS tại quy mô trung bình, năm gốc 2021; trục X giới hạn từ p10 đến p90.</text>', '</svg>']
    destination.write_text("\n".join(parts), encoding="utf-8")
    return pred_df


def chart_correlation_heatmap(corr: pd.DataFrame, destination: Path) -> None:
    """Heatmap Pearson với thang màu xanh âm – trắng 0 – đỏ dương."""
    labels = corr.columns.tolist()
    n = len(labels)
    width, height = 920, 820
    cell = 78
    x0, y0 = 230, 155
    blue, white, red = (63, 91, 176), (247, 247, 247), (190, 30, 45)
    ink = "#202b3c"

    def color(value: float) -> str:
        value = max(-1.0, min(1.0, float(value)))
        if value < 0:
            t = value + 1
            rgb = tuple(round(blue[i] + t * (white[i] - blue[i])) for i in range(3))
        else:
            t = value
            rgb = tuple(round(white[i] + t * (red[i] - white[i])) for i in range(3))
        return "#%02x%02x%02x" % rgb

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
             '<title id="title">Ma trận hệ số tương quan Pearson</title>',
             '<desc id="desc">Heatmap tương quan giữa biến phụ thuộc, biến giải thích, biến điều tiết, quy mô, dummy và hai biến tương tác.</desc>',
             '<rect width="100%" height="100%" fill="#ffffff"/>',
             f'<text x="{width/2}" y="42" text-anchor="middle" font-family="Arial" font-size="23" font-weight="bold" fill="{ink}">Ma trận hệ số tương quan Pearson</text>']
    for j, label in enumerate(labels):
        x = x0 + j * cell + cell / 2
        parts.append(f'<text transform="translate({x:.1f} {y0-12}) rotate(-45)" text-anchor="start" font-family="Arial" font-size="13" fill="{ink}">{xml_escape(label)}</text>')
    for i, row_label in enumerate(labels):
        y = y0 + i * cell
        parts.append(f'<text x="{x0-12}" y="{y+cell/2+5:.1f}" text-anchor="end" font-family="Arial" font-size="13" fill="{ink}">{xml_escape(row_label)}</text>')
        for j, col_label in enumerate(labels):
            value = float(corr.loc[row_label, col_label])
            x = x0 + j * cell
            fill = color(value)
            text_color = "#ffffff" if abs(value) >= .55 else ink
            parts += [f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="#ffffff" stroke-width="1.5"/>',
                      f'<text x="{x+cell/2}" y="{y+cell/2+5}" text-anchor="middle" font-family="Arial" font-size="13" fill="{text_color}">{value:.3f}</text>']
    bar_x, bar_y, bar_h = x0 + n * cell + 32, y0, n * cell
    segments = 100
    for k in range(segments):
        v = 1 - 2 * k / (segments - 1)
        y = bar_y + k * bar_h / segments
        parts.append(f'<rect x="{bar_x}" y="{y:.1f}" width="22" height="{bar_h/segments+1:.1f}" fill="{color(v)}"/>')
    for v, text_value in [(1, "1.00"), (.5, "0.50"), (0, "0.00"), (-.5, "−0.50"), (-1, "−1.00")]:
        y = bar_y + (1 - v) / 2 * bar_h
        parts.append(f'<text x="{bar_x+30}" y="{y+5:.1f}" font-family="Arial" font-size="12" fill="{ink}">{text_value}</text>')
    parts += [f'<text x="{width/2}" y="{height-24}" text-anchor="middle" font-family="Arial" font-size="12" fill="#5f6c78">Đỏ: tương quan dương · Xanh: tương quan âm · Đường chéo luôn bằng 1</text>', '</svg>']
    destination.write_text("\n".join(parts), encoding="utf-8")


DATA_TAB_TEMPLATE = """<div id="tab-data" hidden>
<section class="card"><h3>Dữ liệu trước khi lọc (__N_RAW__ công ty–năm)</h3>
<p class="legend">Toàn bộ công ty–năm ghép được từ báo cáo kết quả kinh doanh và bảng cân đối kế toán, FY2021–FY2025. Đơn vị: tỷ đồng. Cột Trạng thái cho biết dòng được giữ lại hay bị loại ở bước lọc nào (xét theo thứ tự các bước ở mục 2).</p>
<div class="scroll">__STATUS_TABLE__</div>
<div class="dt-controls"><input type="search" id="raw-q" placeholder="Tìm mã cổ phiếu" aria-label="Tìm mã cổ phiếu"><select id="raw-f" aria-label="Lọc theo trạng thái"><option value="">Tất cả</option><option value="Giữ lại">Giữ lại</option><option value="Loại">Bị loại</option></select><span id="raw-count"></span><a class="dl" href="data/data_truoc_loc.csv" download>Tải CSV</a></div>
<div class="scroll"><table class="data dt" id="raw-table"></table></div><div class="pager" id="raw-pager"></div></section>
<section class="card"><h3>Dữ liệu sau khi lọc – mẫu phân tích (__N_CLEAN__ công ty–năm)</h3>
<p class="legend">Đơn vị khoản mục: tỷ đồng. ATO, TPR, LIQ, SIZ là giá trị trước winsor hóa; các cột có hậu tố _w là sau Winsor2 p1–p99 và là giá trị dùng trong hồi quy. DLI = 1 nếu LIQ ≥ 1.</p>
<div class="dt-controls"><input type="search" id="clean-q" placeholder="Tìm mã cổ phiếu" aria-label="Tìm mã cổ phiếu"><span id="clean-count"></span><a class="dl" href="data/data_sau_loc.csv" download>Tải CSV</a></div>
<div class="scroll"><table class="data dt" id="clean-table"></table></div><div class="pager" id="clean-pager"></div></section>
<script type="application/json" id="raw-json">__RAW_JSON__</script>
<script type="application/json" id="clean-json">__CLEAN_JSON__</script>
<script>
(function(){
function mk(id){const meta=JSON.parse(document.getElementById(id+'-json').textContent);const cols=meta.cols,rows=meta.rows,fi=meta.filterCol;
const st={q:'',f:'',page:0},size=50;const tb=document.getElementById(id+'-table'),pg=document.getElementById(id+'-pager'),q=document.getElementById(id+'-q'),f=document.getElementById(id+'-f'),cnt=document.getElementById(id+'-count');
const fmt=v=>v===null?'—':(typeof v==='number'?v.toLocaleString('en-US',{maximumFractionDigits:4}):v);
function render(){const qq=st.q.trim().toUpperCase();const r=rows.filter(x=>(!qq||String(x[0]).toUpperCase().startsWith(qq))&&(!st.f||(fi>=0&&String(x[fi]).startsWith(st.f))));
const pages=Math.max(1,Math.ceil(r.length/size));if(st.page>=pages)st.page=pages-1;const part=r.slice(st.page*size,st.page*size+size);
tb.innerHTML='<thead><tr>'+cols.map(c=>'<th>'+c+'</th>').join('')+'</tr></thead><tbody>'+part.map(x=>'<tr>'+x.map(v=>'<td'+(typeof v==='number'?'':' class="txt"')+'>'+fmt(v)+'</td>').join('')+'</tr>').join('')+'</tbody>';
cnt.textContent=r.length.toLocaleString('en-US')+' dòng';pg.innerHTML='';
const b=(l,p,d)=>{const e=document.createElement('button');e.type='button';e.textContent=l;e.disabled=d;e.onclick=()=>{st.page=p;render();};pg.appendChild(e);};
b('‹ Trước',st.page-1,st.page===0);const sp=document.createElement('span');sp.textContent=' Trang '+(st.page+1)+' / '+pages+' ';pg.appendChild(sp);b('Sau ›',st.page+1,st.page>=pages-1);}
q.addEventListener('input',()=>{st.q=q.value;st.page=0;render();});if(f)f.addEventListener('change',()=>{st.f=f.value;st.page=0;render();});render();}
mk('raw');mk('clean');})();
</script></div>"""


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUT.mkdir(parents=True, exist_ok=True)
    CHARTS.mkdir(parents=True, exist_ok=True)
    bs_path = DATA / "Balance Sheet.csv"
    is_path = DATA / "Income Statement.csv"
    if not bs_path.exists() or not is_path.exists():
        raise FileNotFoundError("Thiếu CSV nguồn trong thư mục data_source.")

    bs = pd.read_csv(bs_path, low_memory=False)
    inc = pd.read_csv(is_path, low_memory=False)
    for df in (bs, inc):
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df["public_date"] = pd.to_datetime(df["public_date"], errors="coerce")
    bs = bs[(bs["period_kind"] == "YEAR") & bs["year"].isin(YEARS)].copy()
    inc = inc[(inc["period_kind"] == "YEAR") & inc["year"].isin(YEARS)].copy()

    # Mỗi mã-năm giữ bản công bố mới nhất nếu nguồn có dòng trùng.
    bs = bs.sort_values("public_date").drop_duplicates(["ticker", "year"], keep="last")
    inc = inc.sort_values("public_date").drop_duplicates(["ticker", "year"], keep="last")
    # Nhóm ngành rộng dựa trên mẫu biểu KQKD riêng trong chính dữ liệu nguồn.
    # Đây không phải hệ thống phân ngành ICB/GICS chi tiết; mục đích là nhận diện
    # ba nhóm tài chính có cấu trúc BCTC không so sánh trực tiếp với mẫu chung.
    inc["industry_group"] = classify_statement_group(inc)
    merged = bs.merge(inc, on=["ticker", "year"], how="inner", suffixes=("_bs", "_is"))

    # Chuẩn hóa kiểu dữ liệu cho các dòng có tên trong README gốc của nguồn.
    for col in ["bsa1", "bsa23", "bsa53", "bsa54", "bsa55", "bsa57", "bsa67", "bsa78",
                "isa1", "isa2", "isa3"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    # Kiểm tra tính đúng đắn của dữ liệu gốc (phi tài chính) bằng các đẳng thức kế toán.
    nonfin = merged[merged["industry_group"] == "Phi tài chính (mẫu biểu chung)"]
    def gap(a: pd.Series, b: pd.Series) -> pd.Series:
        return (a - b).abs() / b.abs().where(b.abs() > 0)
    data_checks = pd.DataFrame([
        {"Kiểm tra": "Trùng mã–năm sau khi ghép", "Số dòng vi phạm": int(merged.duplicated(["ticker", "year"]).sum()), "Xử lý": "Giữ bản công bố mới nhất"},
        {"Kiểm tra": "Doanh thu thuần ≠ doanh thu − giảm trừ (lệch > 0,5%)", "Số dòng vi phạm": int((gap(nonfin["isa1"] + nonfin["isa2"], nonfin["isa3"]) > .005).sum()), "Xử lý": "Không có"},
        {"Kiểm tra": "Tổng tài sản ≠ TS ngắn hạn + TS dài hạn (lệch > 0,5%)", "Số dòng vi phạm": int((gap(nonfin["bsa1"] + nonfin["bsa23"], nonfin["bsa53"]) > .005).sum()), "Xử lý": "Loại"},
        {"Kiểm tra": "Tổng tài sản ≠ nợ phải trả + vốn chủ sở hữu (lệch > 0,5%)", "Số dòng vi phạm": int((gap(nonfin["bsa54"] + nonfin["bsa78"], nonfin["bsa53"]) > .005).sum()), "Xử lý": "Giữ: sai ở khoản VCSH của nguồn, khoản này không dùng trong mô hình"},
        {"Kiểm tra": "Nợ phải trả ≠ nợ ngắn hạn + nợ dài hạn (lệch > 0,5%)", "Số dòng vi phạm": int((gap(nonfin["bsa55"] + nonfin["bsa67"], nonfin["bsa54"]) > .005).sum()), "Xử lý": "Không có"},
        {"Kiểm tra": "Phải trả người bán > nợ ngắn hạn", "Số dòng vi phạm": int((nonfin["bsa57"] > nonfin["bsa55"] * 1.0001).sum()), "Xử lý": "Không có"},
    ]).set_index("Kiểm tra")

    # Lọc tuần tự để có thể kiểm toán chính xác số dòng bị loại ở từng bước.
    sample_steps = []
    def add_step(step: str, frame: pd.DataFrame) -> None:
        sample_steps.append({"Bước": step, "Số quan sát": len(frame),
                             "Số mã": frame["ticker"].nunique()})
    add_step("Ghép được báo cáo KQKD và bảng cân đối theo mã-năm", merged)
    industry_distribution = (merged.groupby("industry_group")
                             .agg(**{"Số quan sát": ("ticker", "size"),
                                     "Số mã": ("ticker", "nunique")})
                             .sort_values("Số quan sát", ascending=False))
    # Loại ngành tài chính trước khi tạo tỷ số. Với các ngành này, doanh thu,
    # tài sản ngắn hạn, nợ ngắn hạn và phải trả có nội hàm/mẫu biểu khác.
    valid = merged[merged["industry_group"] == "Phi tài chính (mẫu biểu chung)"].copy()
    add_step("Loại ngân hàng, chứng khoán và bảo hiểm", valid)
    raw_cols = ["isa3", "bsa53", "bsa55", "bsa1", "bsa57"]
    valid = valid.dropna(subset=raw_cols).copy()
    add_step("Có đủ 5 khoản mục gốc để tạo biến", valid)
    valid = valid[valid["bsa53"] > 0].copy()
    add_step("Tổng tài sản > 0", valid)
    # Bảng cân đối phải khớp: tổng tài sản = TSNH + TSDH = nợ phải trả + VCSH.
    # Dòng không khớp là lỗi trong dữ liệu nguồn nên loại hẳn (winsor không sửa được).
    # Chỉ loại khi phía tài sản không khớp (ảnh hưởng trực tiếp tới biến của mô hình).
    # Dòng lệch nợ + VCSH được giữ: nguồn sai ở khoản VCSH, khoản này không dùng trong mô hình.
    bad_balance = valid[gap(valid["bsa1"] + valid["bsa23"], valid["bsa53"]) > .005]
    equity_gap_rows = int((gap(valid["bsa54"] + valid["bsa78"], valid["bsa53"]) > .005).sum())
    dropped_bad_balance = len(bad_balance)
    dropped_rows = bad_balance.assign(**{"Lý do loại": "Bảng cân đối không khớp"})[
        ["ticker", "year", "Lý do loại", "bsa53", "bsa1", "bsa23", "bsa54", "bsa78", "bsa55", "bsa57", "isa3"]]
    valid = valid[valid["isa3"] > 0].copy()
    add_step("Doanh thu thuần > 0", valid)
    valid = valid[valid["bsa55"] > 0].copy()
    add_step("Nợ ngắn hạn > 0", valid)
    valid = valid[(valid["bsa1"] >= 0) & (valid["bsa57"] >= 0)].copy()
    add_step("Tài sản ngắn hạn và phải trả người bán không âm", valid)

    valid["asset_turnover"] = valid["isa3"] / valid["bsa53"]
    valid["payables_ratio"] = valid["bsa57"] / valid["bsa53"]
    valid["liquidity"] = valid["bsa1"] / valid["bsa55"]
    valid["size_ln"] = np.log(valid["bsa53"] / 1_000_000_000)
    metric_cols = list(VAR_LABELS)
    valid = valid.replace([np.inf, -np.inf], np.nan).dropna(subset=metric_cols)
    add_step("Các biến mô hình hữu hạn và đầy đủ", valid)
    sample_flow = pd.DataFrame(sample_steps).set_index("Bước")
    # Ngành ICB cấp 2 của từng mã, lấy từ danh sách mã của Vietcap (cùng nguồn với dữ liệu BCTC).
    icb_path = DATA / "nganh_icb_vietcap.csv"
    icb_map = (pd.read_csv(icb_path, dtype=str).set_index("ticker")["nganh_icb2"].to_dict()
               if icb_path.exists() else {})
    if valid.empty:
        raise ValueError("Không còn dòng hợp lệ sau khi áp dụng tiêu chí chọn mẫu.")

    # Thống kê trước khi winsor hóa, trên cùng mẫu phân tích cuối cùng.
    desc_before = descriptive(valid, metric_cols)
    wins = valid.copy()
    thresholds = []
    for col in metric_cols:
        low = float(valid[col].quantile(LOWER_P))
        high = float(valid[col].quantile(UPPER_P))
        before = valid[col]
        wins[col] = before.clip(lower=low, upper=high)
        thresholds.append({
            "Biến": SHORT_LABELS[col], "p1": low, "p99": high,
            "Dưới p1": int((before < low).sum()),
            "Trên p99": int((before > high).sum()),
            "Tổng giá trị bị chặn": int(((before < low) | (before > high)).sum()),
        })
    thresholds_df = pd.DataFrame(thresholds).set_index("Biến")
    desc_after = descriptive(wins, metric_cols)

    # Corr trên các biến đã winsor hóa.
    corr = wins[metric_cols].corr(method="pearson")
    corr.index = [SHORT_LABELS.get(x, x) for x in corr.index]
    corr.columns = [SHORT_LABELS.get(x, x) for x in corr.columns]

    # Tương tác: (X - trung bình X) * (Z - trung bình Z).
    # Lập sau winsorization; năm giữ nguyên, không winsor hóa.
    wins["payables_c"] = wins["payables_ratio"] - wins["payables_ratio"].mean()
    wins["liquidity_c"] = wins["liquidity"] - wins["liquidity"].mean()
    wins["interaction"] = wins["payables_c"] * wins["liquidity_c"]
    # Dummy có ngưỡng kinh tế rõ ràng: 1 nếu tài sản ngắn hạn đủ trang trải
    # nợ ngắn hạn (current ratio >= 1), ngược lại bằng 0. Dummy không winsor hóa.
    wins["high_liquidity_dummy"] = (valid.loc[wins.index, "liquidity"] >= 1).astype(int)
    wins["interaction_dummy"] = wins["payables_c"] * wins["high_liquidity_dummy"]
    baseline_vars = ["payables_c", "liquidity_c", "size_ln"]
    interaction_vars = baseline_vars + ["interaction"]
    dummy_interaction_vars = ["payables_c", "high_liquidity_dummy", "size_ln", "interaction_dummy"]
    vif_base = vif_table(wins, baseline_vars)
    vif_inter = vif_table(wins, interaction_vars)
    vif_dummy = vif_table(wins, dummy_interaction_vars)

    # OLS0 chỉ có Y và X; OLS1 là mô hình cơ sở; OLS2 tương tác liên tục;
    # OLS3 dùng dummy thanh khoản để giải thích chênh lệch độ dốc giữa hai nhóm.
    ols0, info0, residual0 = ols_clustered(wins, ["payables_c"],
                                           "Mô hình 0: OLS đơn giản Y ~ X",
                                           include_year_effects=False)
    ols1, info1, residual1 = ols_clustered(wins, baseline_vars,
                                           "Mô hình 1: không có tương tác")
    ols2, info2, residual2 = ols_clustered(wins, interaction_vars,
                                           "Mô hình 2: có tương tác")
    ols3, info3, residual3 = ols_clustered(wins, dummy_interaction_vars,
                                           "Mô hình 3: tương tác với dummy thanh khoản")
    model_info = pd.DataFrame([info0, info1, info2, info3]).set_index("Mô hình")

    # Kiểm định F (H0: R² = 0) cho các mô hình pooled; p-value xấp xỉ Wilson–Hilferty.
    def chi2_sf(x: float, d: int) -> float:
        z = ((x / d) ** (1 / 3) - (1 - 2 / (9 * d))) / math.sqrt(2 / (9 * d))
        return 0.5 * math.erfc(z / math.sqrt(2))
    f_vals, f_ps = [], []
    for res, inf in [(ols0, info0), (ols1, info1), (ols2, info2), (ols3, info3)]:
        k, n_ = len(res), inf["Obs"]
        fv = (inf["R2"] / (k - 1)) / ((1 - inf["R2"]) / (n_ - k))
        f_vals.append(fv); f_ps.append(chi2_sf(fv * (k - 1), k - 1))
    model_info["F"] = f_vals
    model_info["p-value (F)"] = f_ps

    # A1: kiểm tra phi tuyến bằng cách thêm TPRc² vào mô hình 1.
    quad, _, _ = ols_clustered(wins.assign(tpr_sq=wins["payables_c"] ** 2), baseline_vars + ["tpr_sq"], "M1 + TPR²")
    # A4: Breusch–Pagan cho mô hình 2 (hồi quy phần dư bình phương lên các biến X).
    yrs_bp = pd.get_dummies(wins["year"].astype(int), prefix="Năm", drop_first=True, dtype=float)
    x_bp = np.column_stack([np.ones(len(wins)), wins[interaction_vars].astype(float).to_numpy(), yrs_bp.to_numpy()])
    e2 = residual2 ** 2
    e2_fit = x_bp @ np.linalg.lstsq(x_bp, e2, rcond=None)[0]
    bp_r2 = 1 - float(((e2 - e2_fit) ** 2).sum()) / float(((e2 - e2.mean()) ** 2).sum())
    bp_lm, bp_df = len(e2) * bp_r2, x_bp.shape[1] - 1
    bp_p = chi2_sf(bp_lm, bp_df)
    # A5: biến giải thích trễ 1 năm (TPR năm trước không thể bị ATO năm nay tác động ngược).
    lagdf = wins.sort_values(["ticker", "year"]).copy()
    grp = lagdf.groupby("ticker")
    for c in ["payables_ratio", "liquidity", "size_ln"]:
        lagdf[c + "_l1"] = grp[c].shift(1)
    lagdf["prev_year"] = grp["year"].shift(1)
    lagdf = lagdf[(lagdf["year"] - lagdf["prev_year"]) == 1].dropna(subset=["payables_ratio_l1", "liquidity_l1", "size_ln_l1"]).copy()
    lagdf["payables_c"] = lagdf["payables_ratio_l1"] - lagdf["payables_ratio_l1"].mean()
    lagdf["liquidity_c"] = lagdf["liquidity_l1"] - lagdf["liquidity_l1"].mean()
    lagdf["size_ln"] = lagdf["size_ln_l1"]
    lagdf["interaction"] = lagdf["payables_c"] * lagdf["liquidity_c"]
    lag2, lag_info2, _ = ols_clustered(lagdf, interaction_vars, "Mô hình 2 – biến trễ 1 năm")

    # Kiểm tra độ bền: cùng ba mô hình nhưng thêm hiệu ứng cố định công ty.
    fe1, fe_info1, _ = ols_clustered(wins, baseline_vars, "Mô hình 1 – FE công ty", within=True)
    fe2, fe_info2, _ = ols_clustered(wins, interaction_vars, "Mô hình 2 – FE công ty", within=True)
    fe3, fe_info3, _ = ols_clustered(wins, dummy_interaction_vars, "Mô hình 3 – FE công ty", within=True)

    def coef_cell(res: pd.DataFrame, name: str) -> str:
        if name not in res.index:
            return ""
        coef, se, pv = res.loc[name, ["Hệ số OLS", "SE gom cụm", "p-value xấp xỉ"]]
        stars = "***" if pv < .01 else "**" if pv < .05 else "*" if pv < .10 else ""
        return f"{coef:.4f}{stars} ({se:.4f})"

    compare_vars = ["payables_c", "liquidity_c", "high_liquidity_dummy", "size_ln",
                    "interaction", "interaction_dummy"]
    compare_models = [("M1 Pooled", ols1, info1), ("M1 FE", fe1, fe_info1),
                      ("M2 Pooled", ols2, info2), ("M2 FE", fe2, fe_info2),
                      ("M3 Pooled", ols3, info3), ("M3 FE", fe3, fe_info3),
                      ("M2 trễ 1 năm", lag2, lag_info2)]
    robust_compare = pd.DataFrame(
        {label: [coef_cell(res, v) for v in compare_vars] + [f"{inf['Obs']:,}", f"{inf['R2']:.4f}"]
         for label, res, inf in compare_models},
        index=[RESULT_LABELS[v] for v in compare_vars] + ["Obs", "R²"])
    robust_compare.index.name = "Biến"

    def sign_word(a: float, b: float) -> str:
        return "giữ dấu" if np.sign(a) == np.sign(b) else "đổi dấu"
    b = lambda res, v: float(res.loc[v, "Hệ số OLS"])
    pv = lambda res, v: float(res.loc[v, "p-value xấp xỉ"])
    fe_reading = (
        f"Khi thêm hiệu ứng cố định công ty, hệ số TPR ở mô hình 2 đi từ {b(ols2,'payables_c'):.4f} "
        f"xuống {b(fe2,'payables_c'):.4f} (p ≈ {pv(fe2,'payables_c'):.3g}); hệ số TLI đi từ "
        f"{b(ols2,'interaction'):.4f} xuống {b(fe2,'interaction'):.4f} (p ≈ {pv(fe2,'interaction'):.3g}). "
        f"Hệ số TLD của mô hình 3 {sign_word(b(ols3,'interaction_dummy'), b(fe3,'interaction_dummy'))}: "
        f"{b(ols3,'interaction_dummy'):.4f} trong pooled OLS và {b(fe3,'interaction_dummy'):.4f} trong FE "
        f"(p ≈ {pv(fe3,'interaction_dummy'):.3g}).")
    sig_terms = [name for name, res, v in [("TLI", fe2, "interaction"), ("TLD", fe3, "interaction_dummy")]
                 if pv(res, v) < .10]
    if sig_terms:
        fe_reading += f" Trong FE, {', '.join(sig_terms)} vẫn có ý nghĩa thống kê ở mức 10%."
    else:
        fe_reading += (" Trong FE, cả TLI và TLD đều không có ý nghĩa thống kê ở mức 10%: vai trò điều tiết "
                       "của thanh khoản không còn rõ khi chỉ xét biến động trong cùng một công ty. Hệ số TPR "
                       "vẫn dương nhưng nhỏ hơn nhiều so với pooled OLS.")
    corr_full_labels = SHORT_LABELS
    corr_full_cols = list(corr_full_labels)
    corr_full = wins[corr_full_cols].corr(method="pearson").rename(
        index=corr_full_labels, columns=corr_full_labels)
    normality = normality_diagnostics({
        "ATO": wins["asset_turnover"],
        "TPR": wins["payables_ratio"],
        "LIQ": wins["liquidity"],
        "SIZ": wins["size_ln"],
        "RE2": residual2,
        "RE3": residual3,
    })
    normality_html = normality.copy()
    normality_html["Obs"] = normality_html["Obs"].map(lambda x: f"{int(x):,}")
    normality_html["JB p-value"] = normality_html["JB p-value"].map(
        lambda x: "< 1e-300" if pd.notna(x) and x == 0 else (f"{x:.3e}" if pd.notna(x) else "—"))
    vif_base_short = vif_base.rename(index=RESULT_LABELS)
    vif_inter_short = vif_inter.rename(index=RESULT_LABELS)
    vif_dummy_short = vif_dummy.rename(index=RESULT_LABELS)
    ols0_short = ols0.rename(index=RESULT_LABELS)
    ols1_short = ols1.rename(index=RESULT_LABELS)
    ols2_short = ols2.rename(index=RESULT_LABELS)
    ols3_short = ols3.rename(index=RESULT_LABELS)
    model_dictionary = pd.DataFrame([
        {"Mã": "ATO", "Tên biến": "Vòng quay tổng tài sản", "Tên trong code": "asset_turnover", "Cách tính": "Doanh thu thuần / Tổng tài sản", "Vai trò": "Y – hiệu quả hoạt động cần giải thích"},
        {"Mã": "TPR", "Tên biến": "Tỷ lệ phải trả người bán", "Tên trong code": "payables_ratio", "Cách tính": "Phải trả người bán / Tổng tài sản", "Vai trò": "X – tín dụng từ nhà cung cấp"},
        {"Mã": "LIQ", "Tên biến": "Thanh khoản hiện hành", "Tên trong code": "liquidity", "Cách tính": "Tài sản ngắn hạn / Nợ ngắn hạn", "Vai trò": "Z – biến điều tiết"},
        {"Mã": "SIZ", "Tên biến": "Quy mô doanh nghiệp", "Tên trong code": "size_ln", "Cách tính": "ln(Tổng tài sản tính bằng tỷ đồng)", "Vai trò": "Biến kiểm soát"},
        {"Mã": "TLI", "Tên biến": "Tương tác TPR × LIQ", "Tên trong code": "interaction", "Cách tính": "(TPR − trung bình mẫu của TPR) × (LIQ − trung bình mẫu của LIQ)", "Vai trò": "Kiểm tra độ dốc TPR–ATO có đổi theo LIQ"},
        {"Mã": "DLI", "Tên biến": "Dummy đủ thanh khoản", "Tên trong code": "high_liquidity_dummy", "Cách tính": "1 nếu LIQ ≥ 1; ngược lại 0", "Vai trò": "Biến giả 0–1"},
        {"Mã": "TLD", "Tên biến": "Tương tác TPR × DLI", "Tên trong code": "interaction_dummy", "Cách tính": "(TPR − trung bình mẫu của TPR) × DLI", "Vai trò": "So sánh độ dốc giữa DLI=1 và DLI=0"},
        {"Mã": "YDM", "Tên biến": "Biến giả năm", "Tên trong code": "Năm_2022…Năm_2025", "Cách tính": "Năm gốc 2021", "Vai trò": "Kiểm soát cú sốc chung từng năm"},
        {"Mã": "CON", "Tên biến": "Hằng số", "Tên trong code": "Hằng số", "Cách tính": "Hệ số chặn do OLS ước lượng", "Vai trò": "ATO dự báo tại TPR, LIQ trung bình, SIZ = 0 (tài sản 1 tỷ), năm 2021; chỉ dùng để tính dự báo"},
        {"Mã": "Y22–Y25", "Tên biến": "Dummy từng năm", "Tên trong code": "Năm_2022…Năm_2025", "Cách tính": "1 nếu thuộc năm tương ứng; ngược lại 0", "Vai trò": "Các hệ số năm trong bảng OLS; năm gốc 2021"},
        {"Mã": "RE2", "Tên biến": "Phần dư mô hình 2", "Tên trong code": "residual2", "Cách tính": "ATO thực tế − ATO dự báo từ mô hình 2", "Vai trò": "Kiểm tra phân phối sai số mô hình tương tác liên tục"},
        {"Mã": "RE3", "Tên biến": "Phần dư mô hình 3", "Tên trong code": "residual3", "Cách tính": "ATO thực tế − ATO dự báo từ mô hình 3", "Vai trò": "Kiểm tra phân phối sai số mô hình tương tác dummy"},
        {"Mã": "PAT", "Tên biến": "ATO dự báo", "Tên trong code": "predicted_asset_turnover", "Cách tính": "Giá trị dự báo từ hệ số OLS", "Vai trò": "Dữ liệu vẽ biểu đồ tương tác"},
        {"Mã": "LPQ", "Tên biến": "Phân vị LIQ", "Tên trong code": "liquidity_quantile", "Cách tính": "p25, p50 hoặc p75 của LIQ", "Vai trò": "Chọn ba mức thanh khoản cho biểu đồ"},
        {"Mã": "GRP", "Tên biến": "Nhãn nhóm", "Tên trong code": "label", "Cách tính": "Tên nhóm hiển thị trên biểu đồ", "Vai trò": "Phân biệt các đường dự báo"},
    ]).set_index("Mã")
    statement_dictionary = pd.DataFrame([
        {"Mã biến": "ATO", "Khoản mục gốc": "Doanh thu thuần; Tổng cộng tài sản", "Mã nguồn": "isa3; bsa53", "Báo cáo doanh nghiệp": "Báo cáo KQKD; Bảng cân đối kế toán", "Cách đưa vào biến": "isa3 / bsa53"},
        {"Mã biến": "TPR", "Khoản mục gốc": "Phải trả người bán; Tổng cộng tài sản", "Mã nguồn": "bsa57; bsa53", "Báo cáo doanh nghiệp": "Bảng cân đối kế toán", "Cách đưa vào biến": "bsa57 / bsa53"},
        {"Mã biến": "LIQ", "Khoản mục gốc": "Tài sản ngắn hạn; Nợ ngắn hạn", "Mã nguồn": "bsa1; bsa55", "Báo cáo doanh nghiệp": "Bảng cân đối kế toán", "Cách đưa vào biến": "bsa1 / bsa55"},
        {"Mã biến": "SIZ", "Khoản mục gốc": "Tổng cộng tài sản", "Mã nguồn": "bsa53", "Báo cáo doanh nghiệp": "Bảng cân đối kế toán", "Cách đưa vào biến": "ln(bsa53 / 1 tỷ đồng)"},
        {"Mã biến": "DLI", "Khoản mục gốc": "Tạo từ LIQ", "Mã nguồn": "Không có trực tiếp", "Báo cáo doanh nghiệp": "Biến do nghiên cứu xây dựng", "Cách đưa vào biến": "1 nếu LIQ ≥ 1; ngược lại 0"},
        {"Mã biến": "TLI", "Khoản mục gốc": "Tạo từ TPR và LIQ", "Mã nguồn": "Không có trực tiếp", "Báo cáo doanh nghiệp": "Biến do nghiên cứu xây dựng", "Cách đưa vào biến": "(TPR − TPR̄) × (LIQ − LIQ̄); TPR̄, LIQ̄ là trung bình mẫu"},
        {"Mã biến": "TLD", "Khoản mục gốc": "Tạo từ TPR và DLI", "Mã nguồn": "Không có trực tiếp", "Báo cáo doanh nghiệp": "Biến do nghiên cứu xây dựng", "Cách đưa vào biến": "(TPR − TPR̄) × DLI; TPR̄ là trung bình mẫu"},
        {"Mã biến": "YDM", "Khoản mục gốc": "Năm tài chính", "Mã nguồn": "year", "Báo cáo doanh nghiệp": "Thông tin kỳ báo cáo", "Cách đưa vào biến": "Dummy 2022–2025; gốc 2021"},
    ]).set_index("Mã biến")
    requirement_check = pd.DataFrame([
        {"STT": "1", "Yêu cầu": "Chuẩn hóa lại dữ liệu", "Nội dung": "Lọc tuần tự, tỷ lệ/log, Winsor2 p1–p99", "Mục báo cáo": "2, 5–7"},
        {"STT": "2", "Yêu cầu": "Hoàn thiện mô hình và cách đo biến", "Nội dung": "Bảng Y, X, Z, biến kiểm soát, dummy, tương tác và công thức", "Mục báo cáo": "4"},
        {"STT": "3", "Yêu cầu": "Biến giả 0–1", "Nội dung": "DLI = 1 nếu LIQ ≥ 1; DLI = 0 nếu LIQ < 1", "Mục báo cáo": "4, 15"},
        {"STT": "4", "Yêu cầu": "OLS thông thường", "Nội dung": "Mô hình 0: Y ~ X", "Mục báo cáo": "12"},
        {"STT": "5", "Yêu cầu": "Mô hình cơ sở chưa tương tác", "Nội dung": "Mô hình 1: TPR + LIQ + SIZ + biến giả năm", "Mục báo cáo": "13"},
        {"STT": "6", "Yêu cầu": "Mô hình có tương tác", "Nội dung": "Mô hình 2: TPR × LIQ; mô hình 3: TPR × DLI", "Mục báo cáo": "14–17"},
        {"STT": "7", "Yêu cầu": "Cơ sở kinh tế cho tương tác", "Nội dung": "Thanh khoản quyết định khả năng chịu áp lực nghĩa vụ với nhà cung cấp", "Mục báo cáo": "1, 4"},
        {"STT": "8", "Yêu cầu": "Kiểm tra độ bền", "Nội dung": "Ước lượng lại với hiệu ứng cố định công ty và với biến giải thích trễ 1 năm", "Mục báo cáo": "19"},
        {"STT": "9", "Yêu cầu": "Đánh giá mô hình", "Nội dung": "Kỳ vọng dấu, kiểm tra giả định A1–A5, tính chất BLUE", "Mục báo cáo": "20"},
    ]).set_index("STT")

    # Dữ liệu và hình minh họa. SVG được tạo trực tiếp để máy chỉ cần pandas + numpy.
    year_means = wins.groupby("year")[["payables_ratio", "asset_turnover"]].mean()
    chart_year_means(year_means, CHARTS / "01_trung_binh_theo_nam.svg")
    chart_descriptive(desc_after, CHARTS / "02_thong_ke_mo_ta.svg")
    interaction_predictions = chart_interaction(wins, ols2, CHARTS / "03_bieu_do_tuong_tac_OLS.svg")
    dummy_predictions = chart_dummy_interaction(wins, ols3,
                                                 CHARTS / "04_tuong_tac_dummy_OLS.svg")
    chart_correlation_heatmap(corr_full, CHARTS / "05_ma_tran_tuong_quan.svg")
    # Nhúng nguyên SVG vào HTML để báo cáo là một file độc lập: chỉ cần mở
    # bao_cao_ket_qua.html là thấy biểu đồ, không phụ thuộc đường dẫn ảnh ngoài.
    def inline_svg(filename: str) -> str:
        svg = (CHARTS / filename).read_text(encoding="utf-8")
        return svg.replace("<svg ", '<svg class="inline-chart" ', 1)

    chart_year_inline = inline_svg("01_trung_binh_theo_nam.svg")
    chart_desc_inline = inline_svg("02_thong_ke_mo_ta.svg")
    chart_interaction_inline = inline_svg("03_bieu_do_tuong_tac_OLS.svg")
    chart_dummy_inline = inline_svg("04_tuong_tac_dummy_OLS.svg")
    chart_corr_inline = inline_svg("05_ma_tran_tuong_quan.svg")
    dummy_distribution = (wins.groupby("high_liquidity_dummy").size()
                          .rename("Số quan sát").to_frame())
    dummy_distribution["Tỷ lệ mẫu"] = dummy_distribution["Số quan sát"] / len(wins)
    dummy_distribution.index = ["DLI=0", "DLI=1"]

    ap_beta = float(ols2.loc["payables_c", "Hệ số OLS"])
    ap_p = float(ols2.loc["payables_c", "p-value xấp xỉ"])
    int_beta = float(ols2.loc["interaction", "Hệ số OLS"])
    int_p = float(ols2.loc["interaction", "p-value xấp xỉ"])
    dummy_int_beta = float(ols3.loc["interaction_dummy", "Hệ số OLS"])
    dummy_int_p = float(ols3.loc["interaction_dummy", "p-value xấp xỉ"])
    max_vif = float(vif_inter["VIF"].replace([np.inf, -np.inf], np.nan).max())
    max_vif_dummy = float(vif_dummy["VIF"].replace([np.inf, -np.inf], np.nan).max())
    if ap_beta > 0 and int_beta > 0:
        interaction_reading = ("Trong mẫu này, tại mức thanh khoản trung bình, tỷ lệ phải trả người bán có liên hệ "
                               "dương với vòng quay tài sản. Hệ số tương tác dương cho thấy độ dốc dương mạnh hơn "
                               "khi thanh khoản cao hơn. Kết quả này khác dấu tác động chính được tóm tắt trong bài "
                               "nghiên cứu tham khảo; không được diễn giải là ‘thanh khoản làm dịu tác động âm’.")
    elif ap_beta < 0 and int_beta > 0:
        interaction_reading = ("Tỷ lệ phải trả người bán có liên hệ âm tại mức thanh khoản trung bình; tương tác dương "
                               "cho thấy mối liên hệ âm yếu đi khi thanh khoản tăng.")
    else:
        interaction_reading = ("Dấu của hệ số chính và hệ số tương tác phải được đọc cùng nhau; biểu đồ tương tác cho "
                               "thấy độ dốc dự báo ở ba mức thanh khoản.")
    liq1, liq2 = float(ols1.loc["liquidity_c", "Hệ số OLS"]), float(ols2.loc["liquidity_c", "Hệ số OLS"])
    if np.sign(liq1) != np.sign(liq2):
        liq_reading = (f"Hệ số LIQ đổi dấu từ {liq1:.4f} ở mô hình 1 sang {liq2:.4f} ở mô hình 2. Trong mô hình 2, "
                       "hệ số này là độ dốc của LIQ tại mức TPR trung bình nên không cùng nghĩa với mô hình 1; "
                       "kết hợp với VIF cao, hệ số LIQ nhạy với đặc tả và không nên diễn giải riêng.")
    else:
        liq_reading = (f"Hệ số LIQ giữ dấu giữa mô hình 1 ({liq1:.4f}) và mô hình 2 ({liq2:.4f}).")
    negative_equity = int((wins["bsa55"] > wins["bsa53"]).sum())
    tpr_p1 = float(thresholds_df.loc["TPR", "p1"])
    tpr_p99 = float(thresholds_df.loc["TPR", "p99"])
    vif_reading = (f"VIF lớn nhất của mô hình tương tác là {max_vif:.2f}, vượt mốc tham khảo 10. "
                   "Đây là dấu hiệu đa cộng tuyến giữa thanh khoản và tích số; hệ số có thể nhạy với đặc tả mô hình. "
                   "Không nên xóa biến tương tác chỉ để hạ VIF, vì nó là biến cốt lõi của giả thuyết; cần báo cáo giới hạn và kiểm tra độ bền.")

    # Lưu toàn bộ bảng để người học mở bằng Excel nếu muốn.
    raw_grouped_cols = ["ticker", "year", "industry_group", "public_date_bs", "public_date_is"] + raw_cols
    raw_grouped = merged[raw_grouped_cols].copy()
    raw_grouped["eligible_nonfinancial"] = (
        raw_grouped["industry_group"] == "Phi tài chính (mẫu biểu chung)"
    ).astype(int)
    raw_grouped.to_csv(OUT / "00_du_lieu_goc_5_khoan_muc_co_nhom_nganh.csv",
                       index=False, encoding="utf-8-sig")
    industry_distribution.to_csv(OUT / "00_phan_bo_nhom_nganh_truoc_loai.csv",
                                 encoding="utf-8-sig")
    sample_flow.to_csv(OUT / "00_luong_chon_mau.csv", encoding="utf-8-sig")
    dropped_rows.to_csv(OUT / "00_dong_bi_loai_khi_kiem_tra.csv", index=False, encoding="utf-8-sig")
    model_dictionary.to_csv(OUT / "00_tu_dien_bien.csv", encoding="utf-8-sig")
    statement_dictionary.to_csv(OUT / "00_nguon_khoan_muc_va_cach_tinh_bien.csv",
                                encoding="utf-8-sig")
    dummy_distribution.to_csv(OUT / "00_phan_bo_dummy_thanh_khoan.csv", encoding="utf-8-sig")
    desc_before.to_csv(OUT / "01_thong_ke_truoc_winsor.csv", encoding="utf-8-sig")
    thresholds_df.to_csv(OUT / "02_nguong_winsor2_p1_p99.csv", encoding="utf-8-sig")
    desc_after.to_csv(OUT / "03_thong_ke_sau_winsor.csv", encoding="utf-8-sig")
    corr.to_csv(OUT / "04_tuong_quan.csv", encoding="utf-8-sig")
    corr_full.to_csv(OUT / "04b_ma_tran_tuong_quan_day_du.csv", encoding="utf-8-sig")
    normality.to_csv(OUT / "04c_skewness_kurtosis_CV_JB.csv", encoding="utf-8-sig")
    vif_base_short.to_csv(OUT / "05_VIF_mo_hinh_1.csv", encoding="utf-8-sig")
    vif_inter_short.to_csv(OUT / "06_VIF_mo_hinh_2.csv", encoding="utf-8-sig")
    vif_dummy_short.to_csv(OUT / "06b_VIF_mo_hinh_3_dummy.csv", encoding="utf-8-sig")
    model_info.to_csv(OUT / "07_tom_tat_OLS.csv", encoding="utf-8-sig")
    ols0_short.to_csv(OUT / "07b_he_so_OLS_don_gian.csv", encoding="utf-8-sig")
    ols1_short.to_csv(OUT / "08_he_so_OLS_khong_tuong_tac.csv", encoding="utf-8-sig")
    ols2_short.to_csv(OUT / "09_he_so_OLS_co_tuong_tac.csv", encoding="utf-8-sig")
    ols3_short.to_csv(OUT / "09b_he_so_OLS_tuong_tac_dummy.csv", encoding="utf-8-sig")
    year_means.rename(columns=SHORT_LABELS).to_csv(
        OUT / "10_trung_binh_theo_nam.csv", encoding="utf-8-sig")
    interaction_predictions.rename(columns={
        "payables_ratio": "TPR", "liquidity_quantile": "LPQ",
        "liquidity": "LIQ", "predicted_asset_turnover": "PAT", "label": "GRP",
    }).to_csv(OUT / "11_du_bao_cho_bieu_do_tuong_tac.csv", index=False, encoding="utf-8-sig")
    dummy_predictions.rename(columns={
        "payables_ratio": "TPR", "high_liquidity_dummy": "DLI",
        "predicted_asset_turnover": "PAT", "label": "GRP",
    }).to_csv(OUT / "11b_du_bao_tuong_tac_dummy.csv", index=False, encoding="utf-8-sig")
    robust_compare.to_csv(OUT / "12_so_sanh_pooled_FE.csv", encoding="utf-8-sig")
    lag2.rename(index=RESULT_LABELS).to_csv(OUT / "12d_he_so_bien_tre_mo_hinh_2.csv", encoding="utf-8-sig")
    fe1.rename(index=RESULT_LABELS).to_csv(OUT / "12a_he_so_FE_mo_hinh_1.csv", encoding="utf-8-sig")
    fe2.rename(index=RESULT_LABELS).to_csv(OUT / "12b_he_so_FE_mo_hinh_2.csv", encoding="utf-8-sig")
    fe3.rename(index=RESULT_LABELS).to_csv(OUT / "12c_he_so_FE_mo_hinh_3.csv", encoding="utf-8-sig")
    wins.to_csv(OUT / "du_lieu_mau_sau_winsor.csv", index=False, encoding="utf-8-sig")

    changed = int(thresholds_df["Tổng giá trị bị chặn"].sum())

    # ---- Diễn giải từng kết quả (dùng số thật của lần chạy) ----
    def cf_(res, v): return float(res.loc[v, "Hệ số OLS"])
    def pv_(res, v): return float(res.loc[v, "p-value xấp xỉ"])
    def sig(p):
        return ("có ý nghĩa thống kê ở mức 1%" if p < .01 else "có ý nghĩa thống kê ở mức 5%" if p < .05
                else "có ý nghĩa thống kê ở mức 10%" if p < .10 else "không có ý nghĩa thống kê")
    def box(t): return f'<div class="read"><b>Ý nghĩa kết quả:</b> {t}</div>'
    db, da, nm, ym = desc_before, desc_after, normality, year_means
    R5 = box(f"ATO có trung bình {db.loc['ATO','Mean']:.2f} lần nhưng trung vị chỉ {db.loc['ATO','p50']:.2f} và Max tới {db.loc['ATO','Max']:.1f}: "
             f"trung bình bị kéo lên bởi một số ít công ty có vòng quay rất cao (Skewness {db.loc['ATO','Skewness']:.1f}). "
             f"LIQ cũng vậy: Max {db.loc['LIQ','Max']:.0f} lần trong khi trung vị {db.loc['LIQ','p50']:.2f}. "
             f"Trung vị TPR là {db.loc['TPR','p50']*100:.1f}%, tức một nửa số công ty–năm có phải trả người bán dưới {db.loc['TPR','p50']*100:.1f}% tổng tài sản. "
             "Các đuôi dài này là lý do cần winsor hóa trước khi hồi quy.")
    R6 = box(f"Mỗi biến có khoảng {int(thresholds_df['Dưới p1'].median())} giá trị bị nâng lên p1 và {int(thresholds_df['Trên p99'].median())} giá trị bị hạ xuống p99, "
             f"đúng khoảng 1% mỗi đuôi của {len(wins):,} quan sát. Ví dụ mọi công ty có ATO trên {thresholds_df.loc['ATO','p99']:.2f} lần đều được đặt bằng {thresholds_df.loc['ATO','p99']:.2f}.")
    R7 = box(f"Sau winsor hóa, độ lệch chuẩn của ATO giảm từ {db.loc['ATO','SD']:.2f} xuống {da.loc['ATO','SD']:.2f}, của LIQ từ {db.loc['LIQ','SD']:.2f} xuống {da.loc['LIQ','SD']:.2f}; "
             "trung vị, p25 và p75 giữ nguyên. Winsor chỉ thu hẹp hai đuôi, không làm thay đổi phần giữa của dữ liệu.")
    R8 = box(f"Cả bốn biến và phần dư đều bác bỏ giả thuyết phân phối chuẩn (p-value gần 0). Kurtosis của ATO là {nm.loc['ATO','Kurtosis (chuẩn=3)']:.1f}, cao hơn nhiều mức 3 của phân phối chuẩn, tức đuôi dày. "
             "Với mẫu lớn và sai số chuẩn gom cụm, điều này không làm hệ số OLS bị chệch; nó chỉ nhắc rằng p-value là xấp xỉ và cần đọc thận trọng.")
    R9 = box("TPR trung bình theo năm: " + ", ".join(f"{int(y)}: {r.payables_ratio*100:.1f}%" for y, r in ym.iterrows())
             + ". ATO trung bình: " + ", ".join(f"{int(y)}: {r.asset_turnover:.2f}" for y, r in ym.iterrows())
             + f". Năm có TPR cao nhất là {int(ym.payables_ratio.idxmax())}, năm có ATO cao nhất là {int(ym.asset_turnover.idxmax())}. "
             "Biến động theo năm nhỏ so với khác biệt giữa các công ty; biến giả năm trong mô hình dùng để hấp thụ các biến động chung này.")
    cr = corr_full
    pair_max = max(abs(cr.loc['TPR','LIQ']), abs(cr.loc['TPR','SIZ']), abs(cr.loc['LIQ','SIZ']))
    R10 = box(f"ATO tương quan dương với TPR (r = {cr.loc['ATO','TPR']:.2f}), âm nhẹ với LIQ (r = {cr.loc['ATO','LIQ']:.2f}) và SIZ (r = {cr.loc['ATO','SIZ']:.2f}). "
              f"Tương quan giữa các biến giải thích chính TPR, LIQ, SIZ cao nhất chỉ {pair_max:.2f}, nên mô hình cơ sở ít bị đa cộng tuyến. "
              f"TLI tương quan {cr.loc['TLI','LIQ']:.2f} với LIQ và TLD tương quan {cr.loc['TLD','TPR']:.2f} với TPR là do hai biến tương tác được tạo từ chính các biến này.")
    R11 = box(f"Mô hình 1 có VIF lớn nhất {vif_base['VIF'].max():.2f}, dưới mốc 5 nên không đáng lo. Ở mô hình 2, LIQ và TLI có VIF khoảng {max_vif:.1f}: "
              f"sai số chuẩn của hai hệ số này bị phóng to khoảng √VIF ≈ {max_vif**0.5:.1f} lần so với khi không có đa cộng tuyến, nên hệ số LIQ ở mô hình 2 kém ổn định. "
              f"Mô hình 3 có VIF lớn nhất {max_vif_dummy:.2f}, chấp nhận được.")
    b0 = cf_(ols0, 'payables_c')
    R12 = box(f"Hệ số TPR = {b0:.4f} ({sig(pv_(ols0,'payables_c'))}). Vì TPR là tỷ lệ, nên đọc theo 1 điểm phần trăm: TPR cao hơn 0,01 gắn với ATO cao hơn khoảng {b0/100:.3f} lần; "
              f"công ty có TPR cao hơn 10 điểm % có ATO cao hơn khoảng {b0/10:.2f} lần. R² = {info0['R2']:.3f}: riêng TPR giải thích khoảng {info0['R2']*100:.1f}% khác biệt về ATO. "
              f"Hằng số {cf_(ols0,'Hằng số'):.3f} là ATO trung bình tại TPR trung bình (vì TPR đã trừ trung bình).")
    b1 = cf_(ols1, 'payables_c'); y22 = cf_(ols1, 'Năm_2022')
    R13 = box(f"Khi giữ LIQ, SIZ và năm không đổi, 1 điểm % TPR gắn với ATO cao hơn {b1/100:.3f} lần ({sig(pv_(ols1,'payables_c'))}); hệ số giảm từ {b0:.2f} ở mô hình 0 xuống {b1:.2f}, "
              f"tức một phần quan hệ thô đến từ thanh khoản và quy mô. LIQ tăng 1 lần gắn với ATO thay đổi {cf_(ols1,'liquidity_c'):.3f} ({sig(pv_(ols1,'liquidity_c'))}). "
              f"SIZ tăng 1 đơn vị (tổng tài sản lớn gấp khoảng 2,7 lần) gắn với ATO thay đổi {cf_(ols1,'size_ln'):.3f} ({sig(pv_(ols1,'size_ln'))}): công ty lớn hơn có vòng quay thấp hơn. "
              f"Y22 = {y22:.3f} nghĩa là năm 2022 ATO cao hơn năm gốc 2021 khoảng {y22:.3f} lần sau khi kiểm soát các biến khác ({sig(pv_(ols1,'Năm_2022'))}).")
    b21, b24 = cf_(ols2, 'payables_c'), cf_(ols2, 'interaction')
    liq_q = wins['liquidity'].quantile([.25, .5, .75]); liq_m = wins['liquidity'].mean()
    me = {q: b21 + b24 * (v - liq_m) for q, v in liq_q.items()}
    R14 = box(f"Tác động biên của TPR = α1 + α4 × (LIQ − trung bình). Với α1 = {b21:.3f} và α4 = {b24:.3f}: tại LIQ p25 ({liq_q[.25]:.2f}), 1 điểm % TPR gắn với ATO cao hơn {me[.25]/100:.3f} lần; "
              f"tại p50 ({liq_q[.5]:.2f}) là {me[.5]/100:.3f}; tại p75 ({liq_q[.75]:.2f}) là {me[.75]/100:.3f}. Hệ số TLI dương và {sig(pv_(ols2,'interaction'))}: "
              "trong pooled OLS, công ty thanh khoản cao hơn có quan hệ TPR–ATO dốc hơn. Lưu ý LIQ trung bình mẫu khá cao do đuôi phải, nên α1 là độ dốc tại mức LIQ cao hơn trung vị.")
    b31, b34, bd = cf_(ols3, 'payables_c'), cf_(ols3, 'interaction_dummy'), cf_(ols3, 'high_liquidity_dummy')
    R15 = box(f"Nhóm DLI = 0 (tài sản ngắn hạn không đủ trả nợ ngắn hạn) có độ dốc TPR là {b31:.3f} ({sig(pv_(ols3,'payables_c'))}), tức 1 điểm % TPR gắn với ATO cao hơn {b31/100:.3f} lần. "
              f"Nhóm DLI = 1 có độ dốc {b31+b34:.3f} ({b31:.3f} + {b34:.3f}); chênh lệch độ dốc {b34:.3f} {sig(pv_(ols3,'interaction_dummy'))}. "
              f"Hệ số DLI = {bd:.3f}: tại TPR trung bình, nhóm đủ thanh khoản có ATO cao hơn nhóm còn lại khoảng {bd:.2f} lần. "
              f"Theo cách đọc biến giả trong bài giảng: nhóm DLI = 1 có hệ số chặn mới α<sub>0</sub> + α<sub>2</sub> và độ dốc mới α<sub>1</sub> + α<sub>4</sub> = {b31+b34:.3f}, lớn hơn α<sub>1</sub> = {b31:.3f}.")
    R16 = box("Ba đường đều dốc lên, nghĩa là TPR cao hơn đi kèm ATO dự báo cao hơn ở cả ba mức thanh khoản. Đường p75 dốc nhất và khoảng cách giữa các đường mở rộng khi TPR tăng: "
              "đây là hình ảnh trực quan của hệ số tương tác dương. Biểu đồ tính từ pooled OLS nên mang cùng giới hạn của mô hình 2 (xem mục 19).")
    R17 = box(f"Đường DLI = 1 dốc hơn đường DLI = 0 (độ dốc {b31+b34:.2f} so với {b31:.2f}) và nằm cao hơn ở phần lớn khoảng TPR. "
              "Trong pooled OLS, quan hệ TPR–ATO mạnh hơn ở nhóm có đủ tài sản ngắn hạn trang trải nợ ngắn hạn.")
    R18 = box(f"R² hiệu chỉnh tăng từ {info0['R2 hiệu chỉnh']:.3f} (mô hình 0) lên {info1['R2 hiệu chỉnh']:.3f} (mô hình 1), {info2['R2 hiệu chỉnh']:.3f} (mô hình 2) và {info3['R2 hiệu chỉnh']:.3f} (mô hình 3). "
              f"Thêm tương tác có cải thiện mức giải thích, nhưng mọi mô hình chỉ giải thích khoảng {info2['R2']*100:.0f}% khác biệt về ATO; phần lớn còn lại đến từ yếu tố ngoài mô hình như ngành và chiến lược kinh doanh. "
              "R² thấp là bình thường với dữ liệu doanh nghiệp và không làm hệ số sai, nhưng không nên dùng mô hình để dự báo ATO của từng công ty. "
              f"Kiểm định F với H<sub>0</sub>: R² = 0 bị bác bỏ ở mọi mô hình (p-value lớn nhất {max(f_ps):.2g}). Theo bài giảng, mục tiêu của bài là đo lường mệnh đề X → Y (dấu +/−, có hay không có ý nghĩa) nên chỉ cần R² &gt; 0; R² rất cao chỉ cần khi mục tiêu là cảnh báo, dự báo.")
    R19 = box(f"Pooled OLS trả lời câu hỏi: công ty có TPR cao hơn công ty khác thì ATO có cao hơn không. FE trả lời: khi chính một công ty tăng TPR qua các năm thì ATO của nó có tăng không. "
              f"Trong FE, 1 điểm % TPR gắn với ATO cao hơn {cf_(fe1,'payables_c')/100:.3f} lần ở mô hình 1 ({sig(pv_(fe1,'payables_c'))}), nhỏ hơn nhiều so với pooled OLS. "
              f"TLI ({sig(pv_(fe2,'interaction'))}) và TLD ({sig(pv_(fe3,'interaction_dummy'))}) trong FE cho thấy: kết luận vững là TPR có liên hệ dương với ATO, "
              "còn vai trò điều tiết của thanh khoản chưa đủ bằng chứng.")

    def sgn(res, v):
        if v not in res.index: return "—"
        c_, p_ = float(res.loc[v, "Hệ số OLS"]), float(res.loc[v, "p-value xấp xỉ"])
        st = "***" if p_ < .01 else "**" if p_ < .05 else "*" if p_ < .10 else " (không ý nghĩa)"
        return ("+" if c_ > 0 else "−") + st
    sign_table = pd.DataFrame([
        {"Biến": "TPR", "Dấu kỳ vọng": "−", "Cơ sở": "Bài tham khảo: phụ thuộc tín dụng nhà cung cấp phản ánh thiếu tiền và áp lực thanh toán, làm giảm hiệu quả", "Pooled": sgn(ols2, "payables_c"), "FE công ty": sgn(fe2, "payables_c"), "Trễ 1 năm": sgn(lag2, "payables_c")},
        {"Biến": "LIQ", "Dấu kỳ vọng": "+/−", "Cơ sở": "Thanh khoản tốt giúp hoạt động liên tục (+), nhưng giữ nhiều tài sản ngắn hạn nhàn rỗi làm vòng quay thấp (−)", "Pooled": sgn(ols2, "liquidity_c"), "FE công ty": sgn(fe2, "liquidity_c"), "Trễ 1 năm": sgn(lag2, "liquidity_c")},
        {"Biến": "SIZ", "Dấu kỳ vọng": "+/−", "Cơ sở": "Lợi thế quy mô (+), nhưng công ty lớn thường nhiều tài sản dài hạn nên vòng quay thấp (−)", "Pooled": sgn(ols2, "size_ln"), "FE công ty": sgn(fe2, "size_ln"), "Trễ 1 năm": sgn(lag2, "size_ln")},
        {"Biến": "TLI", "Dấu kỳ vọng": "+", "Cơ sở": "Bài tham khảo: thanh khoản cao làm yếu tác động bất lợi của khoản phải trả", "Pooled": sgn(ols2, "interaction"), "FE công ty": sgn(fe2, "interaction"), "Trễ 1 năm": sgn(lag2, "interaction")},
        {"Biến": "TLD", "Dấu kỳ vọng": "+", "Cơ sở": "Như TLI, so sánh nhóm đủ thanh khoản (LIQ ≥ 1) với nhóm còn lại", "Pooled": sgn(ols3, "interaction_dummy"), "FE công ty": sgn(fe3, "interaction_dummy"), "Trễ 1 năm": "—"},
    ]).set_index("Biến")
    tpr_pos = all(float(r.loc["payables_c", "Hệ số OLS"]) > 0 for r in (ols2, fe2, lag2))
    tpr_sig = [pv_(r, "payables_c") < .05 for r in (ols2, fe2, lag2)]
    int_sig = [pv_(ols2, "interaction") < .05, pv_(fe2, "interaction") < .05, pv_(lag2, "interaction") < .05]
    methods = ["pooled OLS", "FE công ty", "biến trễ 1 năm"]
    sig_m = [m for m, ok in zip(methods, int_sig) if ok]
    inter_txt = ("TLI đúng dấu kỳ vọng (+) và có ý nghĩa ở " + ", ".join(sig_m) if sig_m else "TLI không có ý nghĩa ở cách ước lượng nào") \
        + ("" if int_sig[1] else ", nhưng mất ý nghĩa khi thêm hiệu ứng cố định công ty: vai trò điều tiết của thanh khoản chủ yếu đến từ khác biệt giữa các công ty, không phải từ thay đổi trong cùng một công ty") + "."
    R20a = box(("TPR mang dấu dương ở cả ba cách ước lượng" if tpr_pos else "Dấu của TPR không nhất quán giữa ba cách ước lượng")
               + ", trái với kỳ vọng âm của bài tham khảo. Các giải thích có thể: (1) khác biệt ngành – doanh nghiệp thương mại, phân phối vừa mua chịu nhiều vừa có vòng quay tài sản cao tự nhiên; "
               "(2) quan hệ ngược – doanh thu cao kéo theo mua hàng nhiều nên khoản phải trả tăng, tức ATO tác động lên TPR; (3) bài này đo hiệu quả bằng vòng quay tài sản, trong khi nhiều nghiên cứu dùng ROA hoặc ROE. "
               + inter_txt + " Hướng mở rộng: thêm hiệu ứng cố định ngành chi tiết (ICB), dùng ROA làm biến phụ thuộc thay thế, và dùng biến công cụ (2SLS) như bài gốc để xử lý nội sinh.")
    corr_tld = abs(float(corr_full.loc["TPR", "TLD"]))
    quad_c, quad_p = float(quad.loc["tpr_sq", "Hệ số OLS"]), float(quad.loc["tpr_sq", "p-value xấp xỉ"])
    assumption_table = pd.DataFrame([
        {"Giả định": "A1", "Nội dung": "Tuyến tính theo tham số (hàm bậc 1 của các α)", "Kiểm tra trong bài": "Thêm TPRc² vào mô hình 1",
         "Kết quả": f"Hệ số TPR² = {quad_c:.3f}, p ≈ {quad_p:.3g}",
         "Đánh giá / xử lý": ("Có dấu hiệu phi tuyến: quan hệ TPR–ATO " + ("cong lên" if quad_c > 0 else "cong xuống") + ", mô hình tuyến tính là xấp xỉ") if quad_p < .05 else "Không thấy dấu hiệu phi tuyến rõ"},
        {"Giả định": "A2", "Nội dung": "Không đa cộng tuyến hoàn hảo: Xi ≠ hằng số, Xi ≠ f(Xj)", "Kiểm tra trong bài": "|Corr| ≥ 0,8 hoặc VIF ≥ 10 là nghiêm trọng",
         "Kết quả": f"VIF lớn nhất: M1 {vif_base['VIF'].max():.2f}; M2 {max_vif:.2f}; M3 {max_vif_dummy:.2f}. |Corr(TPR, TLD)| = {corr_tld:.2f}",
         "Đánh giá / xử lý": "Không hoàn hảo nên OLS vẫn ước lượng được. M2 đa cộng tuyến nghiêm trọng do tích số chứa LIQ: SE của LIQ, TLI bị phóng to; vẫn giữ biến tương tác vì là biến cốt lõi"},
        {"Giả định": "A3", "Nội dung": "Cov(ui, uj) = 0: mẫu ngẫu nhiên, sai số độc lập", "Kiểm tra trong bài": "Cấu trúc dữ liệu bảng",
         "Kết quả": "Mỗi công ty lặp lại tối đa 5 năm; mẫu là toàn bộ công ty phi tài chính đủ dữ liệu, không chọn ngẫu nhiên",
         "Đánh giá / xử lý": "Vi phạm: sai số của cùng công ty tương quan qua các năm. Xử lý bằng sai số chuẩn gom cụm theo mã"},
        {"Giả định": "A4", "Nội dung": "Var(ui) = σ² không đổi", "Kiểm tra trong bài": "Breusch–Pagan trên mô hình 2 (H0: phương sai không đổi)",
         "Kết quả": f"LM = {bp_lm:,.1f}, bậc tự do {bp_df}, p ≈ {bp_p:.3g}",
         "Đánh giá / xử lý": "Bác bỏ H0: phương sai thay đổi. Sai số chuẩn gom cụm vững với vấn đề này" if bp_p < .05 else "Không bác bỏ H0"},
        {"Giả định": "A5", "Nội dung": "Cov(Xi, ε) = 0: X ngoại sinh", "Kiểm tra trong bài": "So sánh pooled, FE công ty và biến trễ 1 năm",
         "Kết quả": f"Hệ số TPR (M2): pooled {cf_(ols2,'payables_c'):.2f}; FE {cf_(fe2,'payables_c'):.2f}; trễ 1 năm {cf_(lag2,'payables_c'):.2f}",
         "Đánh giá / xử lý": "Khó đáp ứng nhất. Hệ số thay đổi mạnh giữa các cách ước lượng cho thấy có nội sinh do bỏ sót biến; FE và biến trễ chỉ giảm bớt, chưa loại bỏ hẳn"},
    ]).set_index("Giả định")
    consistent = "TPR có liên hệ dương với ATO" + (" và có ý nghĩa ở cả ba cách ước lượng" if all(tpr_sig) else "")
    R20b = box("Theo bài giảng, ước lượng OLS là BLUE khi thỏa các giả định: <b>không chệch</b> E(α̂) = α cần A1, A2, A5; <b>hiệu quả</b> (phương sai nhỏ nhất) cần A3, A4; <b>vững</b> (α̂ hội tụ về α khi n tăng) cần A5 và là tính chất quan trọng nhất. "
               "Trong bài này A3, A4 bị vi phạm nên OLS không còn hiệu quả nhất, nhưng suy luận vẫn dùng được nhờ sai số chuẩn gom cụm. A5 là điểm yếu chính: chưa thể khẳng định tính không chệch và vững, nên các hệ số chỉ được diễn giải là mối liên hệ, không phải tác động nhân quả. "
               f"Vì vậy bài kiểm tra độ bền bằng ba cách (pooled OLS, FE công ty, biến trễ 1 năm). Theo nguyên tắc không có phương pháp nào tốt nhất, tốt nhất là các phương pháp cùng dẫn đến một kết quả: kết quả nhất quán là {consistent}; "
               + ("vai trò điều tiết của thanh khoản thì " + ("không vững vì mất ý nghĩa khi thêm hiệu ứng cố định công ty, nên chưa đủ bằng chứng để kết luận." if not int_sig[1] else "vững qua các cách ước lượng.")))
    industry_icb = (wins.assign(nganh=wins["ticker"].map(lambda t: icb_map.get(t, "Không có trong danh sách hiện hành")))
                    .groupby("nganh").agg(**{"Số quan sát": ("ticker", "size"), "Số mã": ("ticker", "nunique")})
                    .sort_values("Số quan sát", ascending=False))
    industry_icb["Tỷ lệ quan sát"] = industry_icb["Số quan sát"] / len(wins)
    industry_icb.index.name = "Ngành (ICB cấp 2)"
    industry_icb.to_csv(OUT / "00_phan_bo_nganh_icb.csv", encoding="utf-8-sig")
    gallery_items = [
        ("Bảng", "Nhóm ngành theo mẫu biểu BCTC trước khi loại", table_html(industry_distribution, digits=0)),
        ("Bảng", "Luồng chọn mẫu", table_html(sample_flow, digits=0)),
        ("Bảng", "Phân bổ mẫu phân tích theo ngành ICB cấp 2", table_html(industry_icb)),
    ]
    gallery_items += [
        ("Bảng", "Chú thích tên biến", table_html(model_dictionary, text=True)),
        ("Bảng", "Nguồn khoản mục và cách tính biến", table_html(statement_dictionary, text=True)),
        ("Bảng", "Phân bổ biến giả thanh khoản DLI", table_html(dummy_distribution)),
        ("Bảng", "Thống kê mô tả trước winsor hóa", table_html(desc_before)),
        ("Bảng", "Ngưỡng Winsor2 p1–p99", table_html(thresholds_df)),
        ("Bảng", "Thống kê mô tả sau winsor hóa", table_html(desc_after)),
        ("Biểu đồ", "Thống kê mô tả sau Winsor2", chart_desc_inline),
        ("Bảng", "Skewness, Kurtosis, CV và Jarque–Bera", table_html(normality_html, digits=3)),
        ("Biểu đồ", "Phải trả người bán và hiệu quả hoạt động theo năm", chart_year_inline),
        ("Bảng", "Ma trận tương quan Pearson", table_html(corr_full)),
        ("Biểu đồ", "Heatmap tương quan Pearson", chart_corr_inline),
        ("Bảng", "VIF mô hình 1", table_html(vif_base_short)),
        ("Bảng", "VIF mô hình 2", table_html(vif_inter_short)),
        ("Bảng", "VIF mô hình 3", table_html(vif_dummy_short)),
        ("Bảng", "Mô hình 0: OLS đơn giản ATO ~ TPR", table_html(ols0_short)),
        ("Bảng", "Mô hình 1: cơ sở chưa có tương tác", table_html(ols1_short)),
        ("Bảng", "Mô hình 2: tương tác TPR × LIQ", table_html(ols2_short)),
        ("Bảng", "Mô hình 3: tương tác TPR × DLI", table_html(ols3_short)),
        ("Biểu đồ", "Tương tác phải trả người bán × thanh khoản", chart_interaction_inline),
        ("Biểu đồ", "Tương tác với biến giả thanh khoản 0–1", chart_dummy_inline),
        ("Bảng", "Tóm tắt các mô hình", table_html(model_info)),
        ("Bảng", "So sánh pooled OLS, hiệu ứng cố định công ty và biến trễ 1 năm", table_html(robust_compare)),
        ("Bảng", "Kỳ vọng dấu so với kết quả", table_html(sign_table, text=True)),
        ("Bảng", "Kiểm tra các giả định A1–A5", table_html(assumption_table, text=True)),
    ]
    counters = {"Bảng": 0, "Biểu đồ": 0}
    gallery_parts = []
    for kind, title, html in gallery_items:
        counters[kind] += 1
        gallery_parts.append(f'<section class="card gallery"><h3>{kind} {counters[kind]}. {title}</h3><div class="scroll">{html}</div></section>')
    gallery_html = "\n".join(gallery_parts)

    # ---- Tab dữ liệu: toàn bộ dữ liệu trước và sau lọc ----
    import json as _json
    NONFIN = "Phi tài chính (mẫu biểu chung)"
    status = pd.Series("Giữ lại", index=merged.index, dtype="object")
    rules = [
        (merged["industry_group"] != NONFIN, "Loại: ngân hàng, chứng khoán, bảo hiểm"),
        (merged[raw_cols].isna().any(axis=1), "Loại: thiếu khoản mục"),
        (~(merged["bsa53"] > 0), "Loại: tổng tài sản ≤ 0"),
        (~(merged["isa3"] > 0), "Loại: doanh thu thuần ≤ 0"),
        (~(merged["bsa55"] > 0), "Loại: nợ ngắn hạn ≤ 0"),
        ((merged["bsa1"] < 0) | (merged["bsa57"] < 0), "Loại: TSNH hoặc phải trả âm"),
    ]
    for mask, label in reversed(rules):
        status[mask.fillna(False)] = label
    status[(status == "Giữ lại") & ~merged.index.isin(wins.index)] = "Loại: biến không hữu hạn"
    assert int((status == "Giữ lại").sum()) == len(wins)
    bil = lambda x: None if pd.isna(x) else round(float(x) / 1e9, 3)
    icb_of = lambda t: icb_map.get(t, "Không có trong danh sách hiện hành")
    num = lambda x, d=4: None if pd.isna(x) else round(float(x), d)
    raw_cols_vn = ["Mã", "Năm", "Ngành (ICB cấp 2)", "Trạng thái", "Doanh thu thuần", "Tổng tài sản",
                   "TS ngắn hạn", "Nợ ngắn hạn", "Phải trả người bán"]
    mo = merged.assign(st_=status).sort_values(["ticker", "year"])
    raw_rows = [[r.ticker, str(int(r.year)), icb_of(r.ticker), r.st_, bil(r.isa3), bil(r.bsa53), bil(r.bsa1), bil(r.bsa55), bil(r.bsa57)]
                for r in mo.itertuples()]
    clean_cols_vn = ["Mã", "Năm", "Ngành (ICB cấp 2)", "Doanh thu thuần", "Tổng tài sản", "TS ngắn hạn", "Nợ ngắn hạn", "Phải trả người bán",
                     "ATO", "TPR", "LIQ", "SIZ", "ATO_w", "TPR_w", "LIQ_w", "SIZ_w", "DLI"]
    vo = valid.loc[wins.index]
    cl = wins.assign(x_ato=vo["asset_turnover"], x_tpr=vo["payables_ratio"], x_liq=vo["liquidity"], x_siz=vo["size_ln"]).sort_values(["ticker", "year"])
    clean_rows = [[r.ticker, str(int(r.year)), icb_of(r.ticker), bil(r.isa3), bil(r.bsa53), bil(r.bsa1), bil(r.bsa55), bil(r.bsa57),
                   num(r.x_ato), num(r.x_tpr), num(r.x_liq), num(r.x_siz), num(r.asset_turnover), num(r.payables_ratio),
                   num(r.liquidity), num(r.size_ln), int(r.high_liquidity_dummy)] for r in cl.itertuples()]
    data_dir = OUT / "data"; data_dir.mkdir(exist_ok=True)
    pd.DataFrame(raw_rows, columns=raw_cols_vn).to_csv(data_dir / "data_truoc_loc.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(clean_rows, columns=clean_cols_vn).to_csv(data_dir / "data_sau_loc.csv", index=False, encoding="utf-8-sig")
    status_counts = status.value_counts().rename("Số dòng").to_frame()
    status_counts.index.name = "Trạng thái"
    def jdump(cols, rows, fcol):
        return _json.dumps({"cols": cols, "rows": rows, "filterCol": fcol}, ensure_ascii=False, separators=(",", ":"))
    data_tab_html = (DATA_TAB_TEMPLATE
        .replace("__N_RAW__", f"{len(raw_rows):,}").replace("__N_CLEAN__", f"{len(clean_rows):,}")
        .replace("__STATUS_TABLE__", table_html(status_counts, text=True))
        .replace("__RAW_JSON__", jdump(raw_cols_vn, raw_rows, 3))
        .replace("__CLEAN_JSON__", jdump(clean_cols_vn, clean_rows, -1)))

    # ---- Tab mô hình: tất cả các mô hình ước lượng trong bài ----
    def F(x): return f'<div class="formula">{x}</div>'
    E = "ε<sub>i,t</sub>"
    base = "α<sub>0</sub> + α<sub>1</sub>·TPRc<sub>i,t</sub>"
    ctrl = "α<sub>2</sub>·LIQc<sub>i,t</sub> + α<sub>3</sub>·SIZ<sub>i,t</sub>"
    fe_rhs = "α<sub>1</sub>·TPRc<sub>i,t</sub> + "
    model_specs = [
        ("Mô hình 0 — OLS đơn giản", "ATO<sub>i,t</sub> = " + base + " + " + E,
         "Mối liên hệ thô giữa tỷ lệ phải trả người bán và hiệu quả hoạt động, chưa có biến kiểm soát.",
         "Pooled OLS, sai số chuẩn gom cụm theo mã", info0["Obs"], "12"),
        ("Mô hình 1 — Mô hình cơ sở", "ATO<sub>i,t</sub> = " + base + " + " + ctrl + " + δ<sub>t</sub> + " + E,
         "Tác động riêng biệt của TPR khi giữ thanh khoản, quy mô và năm không đổi.",
         "Pooled OLS có hiệu ứng cố định năm, sai số chuẩn gom cụm theo mã", info1["Obs"], "13"),
        ("Mô hình 2 — Tương tác liên tục", "ATO<sub>i,t</sub> = " + base + " + " + ctrl + " + α<sub>4</sub>·(TPRc × LIQc)<sub>i,t</sub> + δ<sub>t</sub> + " + E,
         "Kiểm tra độ dốc TPR–ATO có thay đổi theo mức thanh khoản không. Tác động biên của TPR = α<sub>1</sub> + α<sub>4</sub>·LIQc.",
         "Pooled OLS có hiệu ứng cố định năm, sai số chuẩn gom cụm theo mã", info2["Obs"], "14, 16"),
        ("Mô hình 3 — Tương tác với biến giả", "ATO<sub>i,t</sub> = " + base + " + α<sub>2</sub>·DLI<sub>i,t</sub> + α<sub>3</sub>·SIZ<sub>i,t</sub> + α<sub>4</sub>·(TPRc × DLI)<sub>i,t</sub> + δ<sub>t</sub> + " + E,
         "So sánh hai nhóm: DLI = 1 có hệ số chặn mới α<sub>0</sub> + α<sub>2</sub> và độ dốc mới α<sub>1</sub> + α<sub>4</sub>.",
         "Pooled OLS có hiệu ứng cố định năm, sai số chuẩn gom cụm theo mã", info3["Obs"], "15, 17"),
        ("Mô hình 1 FE — Hiệu ứng cố định công ty", "ATO<sub>i,t</sub> = " + fe_rhs + ctrl + " + δ<sub>t</sub> + f<sub>i</sub> + " + E,
         "Kiểm tra độ bền: chỉ dùng biến động trong cùng một công ty qua các năm, loại bỏ đặc điểm không đổi theo thời gian như ngành.",
         "OLS trên dữ liệu biến đổi within, sai số chuẩn gom cụm theo mã", fe_info1["Obs"], "19"),
        ("Mô hình 2 FE", "ATO<sub>i,t</sub> = " + fe_rhs + ctrl + " + α<sub>4</sub>·(TPRc × LIQc)<sub>i,t</sub> + δ<sub>t</sub> + f<sub>i</sub> + " + E,
         "Mô hình 2 có thêm hiệu ứng cố định công ty.", "OLS trên dữ liệu biến đổi within, sai số chuẩn gom cụm theo mã", fe_info2["Obs"], "19"),
        ("Mô hình 3 FE", "ATO<sub>i,t</sub> = " + fe_rhs + "α<sub>2</sub>·DLI<sub>i,t</sub> + α<sub>3</sub>·SIZ<sub>i,t</sub> + α<sub>4</sub>·(TPRc × DLI)<sub>i,t</sub> + δ<sub>t</sub> + f<sub>i</sub> + " + E,
         "Mô hình 3 có thêm hiệu ứng cố định công ty.", "OLS trên dữ liệu biến đổi within, sai số chuẩn gom cụm theo mã", fe_info3["Obs"], "19"),
        ("Mô hình 2 trễ 1 năm", "ATO<sub>i,t</sub> = α<sub>0</sub> + α<sub>1</sub>·TPRc<sub>i,t−1</sub> + α<sub>2</sub>·LIQc<sub>i,t−1</sub> + α<sub>3</sub>·SIZ<sub>i,t−1</sub> + α<sub>4</sub>·(TPRc × LIQc)<sub>i,t−1</sub> + δ<sub>t</sub> + " + E,
         "Kiểm tra độ bền với nội sinh do quan hệ ngược: TPR năm trước không thể bị ATO năm nay tác động.",
         "Pooled OLS có hiệu ứng cố định năm (năm gốc 2022), sai số chuẩn gom cụm theo mã", lag_info2["Obs"], "19, 20"),
        ("Kiểm tra A1 — Mô hình 1 thêm bình phương TPR", "ATO<sub>i,t</sub> = " + base + " + " + ctrl + " + α<sub>4</sub>·TPRc<sup>2</sup><sub>i,t</sub> + δ<sub>t</sub> + " + E,
         "Kiểm tra giả định tuyến tính: α<sub>4</sub> có ý nghĩa thống kê nghĩa là quan hệ TPR–ATO cong.",
         "Pooled OLS có hiệu ứng cố định năm, sai số chuẩn gom cụm theo mã", info1["Obs"], "20"),
        ("Kiểm tra A4 — Hồi quy phụ Breusch–Pagan", "ε̂<sup>2</sup><sub>i,t</sub> = γ<sub>0</sub> + γ<sub>1</sub>·TPRc<sub>i,t</sub> + γ<sub>2</sub>·LIQc<sub>i,t</sub> + γ<sub>3</sub>·SIZ<sub>i,t</sub> + γ<sub>4</sub>·(TPRc × LIQc)<sub>i,t</sub> + δ<sub>t</sub> + v<sub>i,t</sub>",
         "Hồi quy phần dư bình phương của mô hình 2 lên các biến giải thích. LM = n·R² ~ χ²(k); H<sub>0</sub>: phương sai sai số không đổi.",
         "OLS", info2["Obs"], "20"),
    ]
    model_cards = "".join(
        f'<section class="card"><h3>{t}</h3>{F(eq)}<p>{d}</p>'
        f'<p class="legend">Ước lượng: {how}. Số quan sát: {int(n):,}. Kết quả ở mục {sec} của báo cáo đầy đủ.</p></section>'
        for t, eq, d, how, n, sec in model_specs)
    legend_rows = [
        ("ATO<sub>i,t</sub>", "Vòng quay tổng tài sản của công ty i năm t = doanh thu thuần / tổng tài sản (biến phụ thuộc)"),
        ("TPRc<sub>i,t</sub>", "Tỷ lệ phải trả người bán / tổng tài sản, đã trừ trung bình mẫu (biến độc lập chính)"),
        ("LIQc<sub>i,t</sub>", "Thanh khoản hiện hành = tài sản ngắn hạn / nợ ngắn hạn, đã trừ trung bình mẫu (biến điều tiết)"),
        ("SIZ<sub>i,t</sub>", "ln(tổng tài sản, tỷ đồng) (biến kiểm soát)"),
        ("DLI<sub>i,t</sub>", "Biến giả = 1 nếu LIQ ≥ 1, ngược lại = 0"),
        ("δ<sub>t</sub>", "Hiệu ứng cố định năm: biến giả từng năm so với năm gốc"),
        ("f<sub>i</sub>", "Hiệu ứng cố định công ty: hệ số chặn riêng cho từng công ty"),
        ("ε<sub>i,t</sub>", "Sai số: thông tin có thể giải thích ATO nhưng chưa đưa vào mô hình"),
        ("t − 1", "Giá trị của năm trước"),
    ]
    models_tab_html = ('<div id="tab-models" hidden>'
        '<section class="card"><h3>Ký hiệu dùng trong các mô hình</h3><table class="data text"><tbody>'
        + "".join(f"<tr><th>{a}</th><td>{b}</td></tr>" for a, b in legend_rows)
        + '</tbody></table><p class="legend">Mọi biến liên tục đã được winsor hóa p1–p99. i: công ty, t: năm.</p></section>'
        + model_cards + '</div>')
    # ---- Tab code: toàn bộ mã nguồn main.py ----
    source_code = Path(__file__).read_text(encoding="utf-8")
    code_tab_html = (
        '<div id="tab-code" hidden><section class="card"><h3>Mã nguồn main.py</h3>'
        f'<p class="legend">Toàn bộ code Python ({len(source_code.splitlines()):,} dòng) đọc dữ liệu, lọc mẫu, winsor hóa, tính thống kê, chạy OLS/FE và sinh ra mọi bảng, biểu đồ trong trang này. '
        'Chạy bằng lệnh <code>py main.py</code> trong thư mục bài tập; chỉ cần thư viện pandas và numpy. Mỗi bảng được tạo bởi một biến DataFrame cùng tên trong code, ví dụ ols2 là mô hình 2, robust_compare là bảng so sánh pooled–FE.</p>'
        '<div class="dt-controls"><button type="button" class="btn" id="copy-code">Sao chép code</button><a class="dl" href="code/main.py" download>Tải main.py</a></div>'
        f'<pre class="code"><code class="language-python">{xml_escape(source_code)}</code></pre></section>'
        '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css">'
        '<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>'
        '<script>(function(){const c=document.querySelector("pre.code code");if(window.hljs){hljs.highlightElement(c);}'
        'const b=document.getElementById("copy-code");b.addEventListener("click",async()=>{try{await navigator.clipboard.writeText(c.innerText);b.textContent="Đã sao chép";}'
        'catch(e){const r=document.createRange();r.selectNodeContents(c);const sel=getSelection();sel.removeAllRanges();sel.addRange(r);b.textContent="Đã chọn code, bấm Ctrl+C";}'
        'setTimeout(()=>{b.textContent="Sao chép code";},2500);});})();</script></div>')
    report = f'''<!doctype html><html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Kết quả phân tích trade payables</title>
<style>
body{{font-family:Arial,"Segoe UI",sans-serif;line-height:1.58;color:#202b3c;background:#f4f7fb;margin:0}}
main{{max-width:1180px;margin:28px auto;padding:0 20px 48px}} header,.card{{background:#fff;border:1px solid #e0e7f0;border-radius:14px;padding:22px 26px;margin:16px 0;box-shadow:0 4px 18px #243b530c}}
h1,h2{{color:#123d64}} h1{{margin:0 0 6px}} h2{{margin-top:0}} .tag{{display:inline-block;background:#e9f3ff;color:#145a91;border-radius:999px;padding:4px 11px;font-size:14px}}
.note{{background:#fff8e6;border-left:4px solid #e6a700;padding:12px 16px}} .data{{border-collapse:collapse;width:100%;font-size:14px;margin:10px 0 18px}}
.data th,.data td{{border-bottom:1px solid #e6ebf1;padding:8px 10px;text-align:right;white-space:normal;vertical-align:top}} .data th:first-child,.data td:first-child{{text-align:left;white-space:normal}}
.scroll{{overflow-x:auto}} code{{background:#eef2f7;padding:2px 5px;border-radius:4px}} a{{color:#075ea8}} .formula{{background:#f6f8fb;border:1px solid #e0e7f0;border-radius:10px;padding:14px 18px;margin:12px 0;font-family:"Cambria Math",Cambria,"Times New Roman",serif;font-size:19px;line-height:1.9;text-align:center;overflow-x:auto}} .formula sub,.formula sup{{font-size:.68em}} .legend{{font-size:14px;color:#4a5a6a}} .data.text td{{text-align:left}} .defs li{{margin:6px 0}} .data td.txt{{text-align:left}} .dt td,.dt th{{white-space:nowrap}} .dt-controls{{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:12px 0}} .dt-controls input,.dt-controls select{{font:inherit;padding:6px 10px;border:1px solid #c9d6e3;border-radius:8px}} .dl{{margin-left:auto;padding:6px 14px;border:1px solid #123d64;border-radius:999px;text-decoration:none}} .pager{{display:flex;gap:8px;align-items:center;margin-top:8px}} .pager button{{font:inherit;padding:5px 12px;border:1px solid #c9d6e3;background:#fff;border-radius:8px;cursor:pointer}} .pager button:disabled{{opacity:.4;cursor:default}} pre.code{{max-height:75vh;overflow:auto;background:#f6f8fa;border:1px solid #e0e7f0;border-radius:10px;padding:14px;font-size:13px;line-height:1.5;margin:0}} pre.code code{{font-family:Consolas,"Courier New",monospace;background:none;padding:0}} .btn{{font:inherit;padding:6px 14px;border:1px solid #c9d6e3;background:#fff;border-radius:999px;cursor:pointer}} .inline-chart{{display:block;width:100%;height:auto;margin:14px auto 6px;border:1px solid #dce3ea;background:#fff}}
.tabs{{display:flex;gap:8px;margin:14px 0 4px;flex-wrap:wrap}} .tabs button{{font:inherit;font-size:15px;padding:8px 18px;border:1px solid #c9d6e3;background:#fff;color:#123d64;border-radius:999px;cursor:pointer}} .tabs button[aria-selected="true"]{{background:#123d64;color:#fff;border-color:#123d64}} .tabs button:focus-visible{{outline:2px solid #e6a700;outline-offset:2px}}
.read{{background:#eef6f0;border-left:4px solid #3c8d5a;padding:12px 16px;margin:14px 0 4px}} .gallery h3{{margin:0 0 8px;color:#123d64}} [hidden]{{display:none!important}}
</style></head><body><main>
<header><span class="tag">Dữ liệu thị trường Việt Nam • FY2021–FY2025</span><h1>Trade payables, thanh khoản và hiệu quả hoạt động</h1>
</header>
<nav class="tabs" role="tablist"><button type="button" role="tab" id="btn-full" aria-selected="true" onclick="showTab('full')">Báo cáo đầy đủ</button><button type="button" role="tab" id="btn-models" aria-selected="false" onclick="showTab('models')">Mô hình</button><button type="button" role="tab" id="btn-tables" aria-selected="false" onclick="showTab('tables')">Chỉ bảng và biểu đồ</button><button type="button" role="tab" id="btn-data" aria-selected="false" onclick="showTab('data')">Dữ liệu trước và sau lọc</button><button type="button" role="tab" id="btn-code" aria-selected="false" onclick="showTab('code')">Code Python</button></nav>
<div id="tab-full">
<section class="card"><h2>1. Tóm tắt đề tài nghiên cứu</h2>
<p>Đề tài xem xét việc doanh nghiệp sử dụng <b>khoản phải trả người bán</b> — một dạng tín dụng thương mại do nhà cung cấp cho phép trả chậm — có liên hệ như thế nào với <b>hiệu quả hoạt động</b>, đo bằng vòng quay tổng tài sản. Nghiên cứu đồng thời hỏi liệu <b>thanh khoản</b> có làm mối liên hệ này thay đổi hay không.</p>
<p>Về kinh tế, trả chậm có thể hỗ trợ vốn lưu động và duy trì hoạt động, nhưng phụ thuộc quá nhiều cũng có thể phản ánh thiếu tiền hoặc tạo áp lực thanh toán. Doanh nghiệp thanh khoản tốt có khả năng quản lý nghĩa vụ với nhà cung cấp khác doanh nghiệp thiếu thanh khoản; đó là cơ sở của biến tương tác. Mục tiêu của bài thực hành là đo <b>mối liên hệ có điều kiện</b> bằng OLS, không khẳng định khoản phải trả là nguyên nhân gây thay đổi hiệu quả.</p></section>
<section class="card"><h2>2. Mẫu và cách xử lý</h2><p>Đã ghép báo cáo kết quả kinh doanh với bảng cân đối theo mã và năm. Sau tiêu chí hợp lệ còn <b>{len(wins):,} quan sát công ty–năm</b> thuộc <b>{wins['ticker'].nunique():,} mã</b>; số quan sát theo năm: {', '.join(f'{int(y)}: {int(n):,}' for y,n in wins.groupby('year').size().items())}. Mẫu không cân bằng.</p>
<p><b>Vì sao chọn các công ty này?</b> Phạm vi tải là toàn bộ mã có trong dữ liệu thị trường HOSE, HNX và UPCoM. Doanh nghiệp niêm yết được chọn vì báo cáo tài chính năm công khai giúp lấy cùng các khoản mục giữa nhiều công ty và nhiều kỳ. Một dòng là một công ty trong một năm; cách tổ chức này tạo dữ liệu bảng để so sánh khác biệt giữa công ty và thay đổi theo thời gian.</p>
<p><b>Vì sao chọn 5 năm 2021–2025?</b> Năm năm đủ dài để mỗi doanh nghiệp có nhiều quan sát và để mô hình kiểm soát các cú sốc chung theo năm, nhờ đó kết quả không phụ thuộc vào riêng một kỳ. Đồng thời đây là giai đoạn gần nhất, giúp dữ liệu còn phù hợp với bối cảnh hiện tại và hạn chế mất mẫu hoặc thay đổi quá lớn về cấu trúc doanh nghiệp/chế độ báo cáo khi kéo kỳ nghiên cứu quá xa. Khoảng thời gian này cân bằng giữa <i>tính cập nhật</i>, <i>độ lớn mẫu</i> và <i>khả năng xử lý bài tập</i>; không có nghĩa 5 năm luôn là lựa chọn tối ưu cho mọi nghiên cứu.</p>
<p><b>Vì sao loại công ty tài chính?</b> Ngân hàng, chứng khoán và bảo hiểm có mẫu báo cáo, nguồn tạo doanh thu, cấu trúc tài sản–nợ và quy định thanh khoản khác doanh nghiệp sản xuất, thương mại, dịch vụ. Ví dụ tiền gửi khách hàng của ngân hàng không tương đương nợ nhà cung cấp; “tài sản ngắn hạn/nợ ngắn hạn” và vòng quay tài sản vì thế không có cùng nội hàm. Nếu gộp chung, hệ số có thể phản ánh khác biệt mô hình kinh doanh giữa hai nhóm thay vì quan hệ giữa tín dụng thương mại và hiệu quả. Chương trình nhận diện ba nhóm này từ các mẫu cột KQKD riêng <code>isb</code>, <code>iss</code>, <code>isi</code> của nguồn và loại trước khi tạo tỷ số.</p>
<h3>Nhóm ngành theo mẫu biểu BCTC trước khi loại</h3><div class="scroll">{table_html(industry_distribution, digits=0)}</div>
<p><b>Vì sao các tiêu chí lọc gắn với mô hình?</b> Cần doanh thu thuần dương để tính vòng quay tài sản (Y); tổng tài sản dương làm mẫu số cho Y và tỷ lệ phải trả (X); nợ ngắn hạn dương làm mẫu số cho thanh khoản (Z); tài sản ngắn hạn và phải trả người bán không âm để tỷ lệ có ý nghĩa. Chỉ giữ dòng có đủ các khoản mục cho cùng mã-năm để mọi biến trong OLS được tính trên cùng quan sát. Không bắt buộc một mã phải có đủ cả 5 năm, nhằm tránh loại bỏ toàn bộ công ty chỉ vì thiếu một kỳ.</p>
<p>Về lý thuyết, khoản phải trả người bán là tín dụng từ nhà cung cấp; thanh khoản hiện hành phản ánh khả năng đáp ứng nghĩa vụ ngắn hạn nên có cơ sở để điều tiết mối liên hệ đó; vòng quay tổng tài sản đại diện cho doanh thu tạo ra từ tài sản. Quy mô là biến kiểm soát vì doanh nghiệp lớn/nhỏ có thể khác nhau về hoạt động và nguồn lực.</p>
<h3>Luồng chọn mẫu có thể kiểm toán</h3><div class="scroll">{table_html(sample_flow, digits=0)}</div>
<h3>Phân bổ mẫu phân tích theo ngành ICB cấp 2</h3><p class="legend">Ngành theo chuẩn ICB cấp 2 do Vietcap phân loại, cùng nguồn với dữ liệu báo cáo tài chính.</p><div class="scroll">{table_html(industry_icb)}</div>
<p>Mẫu có {negative_equity:,} công ty–năm có nợ ngắn hạn lớn hơn tổng tài sản, tức vốn chủ sở hữu âm; các quan sát này được giữ vì là doanh nghiệp khó khăn thật, không phải lỗi dữ liệu. Vòng quay tài sản dùng tổng tài sản cuối kỳ thay vì bình quân đầu–cuối kỳ để không mất năm 2021.</p>
<p>Winsor2 p1/p99 được tính riêng cho từng biến liên tục. Có <b>{changed:,} ô giá trị</b> bị chặn về ngưỡng (một quan sát có thể bị chặn ở nhiều biến); không xóa dòng nên Obs trước và sau giữ nguyên.</p>
<p class="note"><b>Giới hạn phạm vi:</b> đây là mẫu rộng của các công ty phi tài chính có đủ dữ liệu, không phải mẫu ngẫu nhiên đại diện cho mọi doanh nghiệp Việt Nam và không phải mẫu gốc của bài báo. Nhóm ngành ở đây chỉ là phân loại rộng theo <b>mẫu biểu BCTC</b> để loại ba nhóm tài chính; chưa phải phân ngành ICB/GICS chi tiết cho từng ngành sản xuất, bất động sản, công nghệ…</p></section>
<section class="card"><h2>3. Cấu trúc bài theo yêu cầu</h2><div class="scroll">{table_html(requirement_check, text=True)}</div></section>
<section class="card"><h2>4. Bảng chú thích tên biến rút gọn</h2><p>Tất cả các bảng kết quả phía dưới dùng thống nhất các mã trong bảng này.</p><div class="scroll">{table_html(model_dictionary, text=True)}</div>
<h3>Khoản mục lấy từ báo cáo nào?</h3><div class="scroll">{table_html(statement_dictionary, text=True)}</div>
<p>Các mã <code>isa*</code> lấy từ Báo cáo kết quả kinh doanh; <code>bsa*</code> lấy từ Bảng cân đối kế toán theo bảng ánh xạ trong README của bộ dữ liệu. Các biến DLI, TLI và TLD do nghiên cứu tự xây dựng, không phải khoản mục có sẵn trong báo cáo doanh nghiệp.</p>
<p><b>Mô hình 0 — OLS thông thường:</b></p><div class="formula">ATO<sub>i,t</sub> = α<sub>0</sub> + α<sub>1</sub>·TPRc<sub>i,t</sub> + ε<sub>i,t</sub></div><p>Đây là hồi quy tuyến tính đơn để nhìn mối liên hệ thô giữa tỷ lệ phải trả người bán và hiệu quả hoạt động, trước khi thêm biến kiểm soát.</p>
<p><b>Mô hình 1 — tác động riêng lẻ:</b></p><div class="formula">ATO<sub>i,t</sub> = α<sub>0</sub> + α<sub>1</sub>·TPRc<sub>i,t</sub> + α<sub>2</sub>·LIQc<sub>i,t</sub> + α<sub>3</sub>·SIZ<sub>i,t</sub> + δ<sub>t</sub> + ε<sub>i,t</sub></div><p class="legend">δ<sub>t</sub>: hiệu ứng cố định năm (biến giả 2022–2025, năm gốc 2021), đại diện cho các cú sốc vĩ mô chung như tăng trưởng GDP, lạm phát, lãi suất. ε<sub>i,t</sub>: sai số, tập hợp các thông tin có thể giải thích ATO nhưng chưa đưa vào mô hình.</p>
<p>Mô hình 1 trả lời: khi giữ thanh khoản, quy mô và năm không đổi, tỷ lệ phải trả người bán liên hệ thế nào với hiệu quả hoạt động?</p>
<p><b>Mô hình 2 — có tương tác:</b></p><div class="formula">ATO<sub>i,t</sub> = α<sub>0</sub> + α<sub>1</sub>·TPRc<sub>i,t</sub> + α<sub>2</sub>·LIQc<sub>i,t</sub> + α<sub>3</sub>·SIZ<sub>i,t</sub> + α<sub>4</sub>·(TPRc × LIQc)<sub>i,t</sub> + δ<sub>t</sub> + ε<sub>i,t</sub></div>
<p><code>TPRc</code> và <code>LIQc</code> là TPR và LIQ đã trừ trung bình mẫu; cách trừ trung bình được dùng thống nhất ở mọi mô hình nên chỉ làm thay đổi hằng số, không đổi độ dốc của mô hình 0 và 1. α1 là độ dốc của phải trả khi thanh khoản ở mức trung bình mẫu. α4 cho biết độ dốc đó thay đổi bao nhiêu khi thanh khoản tăng một đơn vị: <code>tác động biên của TPR = α1 + α4·LIQc</code>. Vì vậy phải đọc α1 và α4 cùng nhau, không đọc riêng hệ số tương tác.</p>
<p><b>Mô hình 3 — tương tác với biến giả:</b></p><div class="formula">ATO<sub>i,t</sub> = α<sub>0</sub> + α<sub>1</sub>·TPRc<sub>i,t</sub> + α<sub>2</sub>·DLI<sub>i,t</sub> + α<sub>3</sub>·SIZ<sub>i,t</sub> + α<sub>4</sub>·(TPRc × DLI)<sub>i,t</sub> + δ<sub>t</sub> + ε<sub>i,t</sub></div>
<p><code>DLI=1</code> khi tài sản ngắn hạn đủ trang trải nợ ngắn hạn (<code>current ratio ≥ 1</code>), ngược lại <code>DLI=0</code>. Ngưỡng 1 có ý nghĩa kinh tế trực tiếp, không được chọn chỉ để chia mẫu đều. Trong mô hình này, α1 là độ dốc của nhóm DLI=0 và α4 là chênh lệch độ dốc giữa nhóm DLI=1 với nhóm DLI=0.</p><div class="scroll">{table_html(dummy_distribution)}</div>
<p>Sai số chuẩn được gom cụm theo mã cổ phiếu để cho phép sai số của cùng doanh nghiệp liên hệ qua các năm. Các mô hình 1–3 là pooled OLS có biến giả năm; chưa kiểm soát đặc điểm cố hữu của từng doanh nghiệp (ngành, mô hình kinh doanh). Mục 19 ước lượng lại bằng hiệu ứng cố định công ty để kiểm tra độ bền. Cả hai cách đều không chứng minh quan hệ nhân quả.</p></section>
<section class="card"><h2>5. Thống kê mô tả trước winsor hóa</h2><div class="scroll">{table_html(desc_before)}</div>{R5}</section>
<section class="card"><h2>6. Winsor2 (1;99) nghĩa là gì?</h2>
<p><b>Winsor2 (1;99)</b> nghĩa là xử lý hai đuôi của <b>từng biến liên tục</b> theo phân vị 1% và 99%:</p>
<ul><li><b>p1:</b> ngưỡng mà khoảng 1% giá trị nằm thấp hơn.</li><li><b>p99:</b> ngưỡng mà khoảng 1% giá trị nằm cao hơn.</li><li>Giá trị thấp hơn p1 được thay bằng đúng p1.</li><li>Giá trị cao hơn p99 được thay bằng đúng p99.</li><li>98% giá trị nằm giữa hai ngưỡng được giữ nguyên.</li></ul>
<div class="formula">x<sup>w</sup> = p<sub>1</sub> nếu x &lt; p<sub>1</sub>; &nbsp; x<sup>w</sup> = x nếu p<sub>1</sub> ≤ x ≤ p<sub>99</sub>; &nbsp; x<sup>w</sup> = p<sub>99</sub> nếu x &gt; p<sub>99</sub></div>
<p><b>Ví dụ với ngưỡng thật của mẫu:</b> p1 của TPR là {tpr_p1:.6f} và p99 là {tpr_p99:.4f}. Một công ty có TPR = 0 được nâng lên {tpr_p1:.6f}; TPR = 0.90 được hạ xuống {tpr_p99:.4f}; TPR = 0.10 giữ nguyên.</p>
<p><b>Điểm quan trọng:</b> Winsor2 không xóa công ty–năm và không thay toàn bộ một dòng dữ liệu. Nó chỉ chặn từng giá trị quá xa ở biến đang xét, nên số Obs trước và sau vẫn bằng nhau. Đây khác với <i>trim</i>, vì trim sẽ loại quan sát nằm ngoài ngưỡng.</p>
<p>Mục đích là giảm mức chi phối của các giá trị cực đoan lên Mean, SD và OLS, nhưng không được xem Winsor2 là cách sửa mọi dữ liệu sai. Bài này áp dụng riêng cho ATO, TPR, LIQ và SIZ trước khi tạo biến tương tác.</p>
<h3>Ngưỡng thực tế trong mẫu</h3><div class="scroll">{table_html(thresholds_df)}</div>{R6}</section>
<section class="card"><h2>7. Thống kê mô tả sau winsor hóa</h2><div class="scroll">{table_html(desc_after)}</div><p>Skewness gần 0 nghĩa là phân phối tương đối cân xứng; dương là đuôi phải dài hơn, âm là đuôi trái dài hơn. Skewness không phải kiểm định phân phối chuẩn.</p>
{chart_desc_inline}<p>Biểu đồ dùng thang đo riêng cho từng dòng. Đường mảnh là Min–Max, đoạn xanh là p25–p75, vòng tròn là trung vị và hình thoi đỏ là trung bình.</p>{R7}</section>
<section class="card"><h2>8. Skewness, Kurtosis, CV và kiểm định Jarque–Bera</h2>
<div class="defs"><p><b>Định nghĩa.</b> Với n quan sát x<sub>1</sub>, …, x<sub>n</sub> có trung bình x̄, mô-men trung tâm bậc k là m<sub>k</sub> = (1/n)·Σ(x<sub>i</sub> − x̄)<sup>k</sup>.</p><ul>
<li><b>Skewness (độ lệch)</b>: S = m<sub>3</sub> / m<sub>2</sub><sup>3/2</sup>, đo tính đối xứng của phân phối. S = 0: đối xứng như phân phối chuẩn. S &gt; 0: đuôi phải dài, một số giá trị rất lớn kéo trung bình lên cao hơn trung vị. S &lt; 0: đuôi trái dài. Quy ước thường dùng: |S| &lt; 0,5 gần đối xứng; 0,5–1 lệch vừa; trên 1 lệch mạnh.</li>
<li><b>Kurtosis (độ nhọn, dạng Pearson)</b>: K = m<sub>4</sub> / m<sub>2</sub><sup>2</sup>, đo độ dày của hai đuôi. Phân phối chuẩn có K = 3. K &gt; 3 (leptokurtic): đuôi dày, nhiều giá trị cực đoan hơn phân phối chuẩn. K &lt; 3 (platykurtic): đuôi mỏng. Một số phần mềm báo excess kurtosis = K − 3, khi đó mốc so sánh là 0.</li>
<li><b>CV (hệ số biến thiên)</b> = SD / |x̄| × 100%: độ phân tán tương đối so với mức trung bình, dùng để so mức biến động giữa các biến có đơn vị khác nhau. CV trên 100% nghĩa là độ lệch chuẩn lớn hơn cả trung bình, biến rất phân tán. CV không dùng cho biến có trung bình gần 0 như phần dư.</li>
<li><b>Kiểm định Jarque–Bera</b>: JB = (n/6)·[S<sup>2</sup> + (K − 3)<sup>2</sup>/4]. H<sub>0</sub>: biến có phân phối chuẩn (S = 0 và K = 3); H<sub>1</sub>: không có phân phối chuẩn. Dưới H<sub>0</sub>, JB tuân theo phân phối χ<sup>2</sup> với 2 bậc tự do, nên p-value = e<sup>−JB/2</sup>; giá trị tới hạn ở mức 5% là 5,99. Quy tắc: p-value &lt; α (1%, 5% hoặc 10%) thì bác bỏ H<sub>0</sub>.</li></ul></div><div class="scroll">{table_html(normality_html, digits=3)}</div><p><b>RE2</b> là phần dư mô hình tương tác liên tục; <b>RE3</b> là phần dư mô hình tương tác dummy. <b>Kurtosis:</b> phân phối chuẩn có giá trị 3. <b>CV:</b> SD chia trị tuyệt đối của Mean, dùng để xem độ biến động tương đối; CV không có ý nghĩa cho phần dư vì trung bình phần dư gần 0. <b>Jarque–Bera:</b> H0 là phân phối chuẩn; p-value &lt; 0,05 dẫn đến bác bỏ H0.</p><p class="note">Với mẫu {len(wins):,} quan sát, JB rất nhạy nên một sai lệch nhỏ cũng có thể cho p-value gần 0. OLS không yêu cầu từng X hoặc Y phải phân phối chuẩn. Phân phối phần dư liên quan trực tiếp hơn đến suy luận mẫu nhỏ; ở đây báo cáo dùng sai số chuẩn gom cụm và xấp xỉ mẫu lớn.</p>{R8}</section>
<section class="card"><h2>9. Trung bình các biến chính theo năm</h2>{chart_year_inline}<p>Đây là dạng hai trục gần với hình Stata tham khảo. Hai đường có đơn vị khác nhau nên chỉ dùng để xem xu hướng theo năm, không so sánh trực tiếp độ cao của hai đường.</p>{R9}</section>
<section class="card"><h2>10. Ma trận tương quan Pearson</h2><div class="scroll">{table_html(corr_full)}</div>{chart_corr_inline}<p>Tên được rút gọn còn ba chữ: <b>ATO</b> hiệu quả, <b>TPR</b> tỷ lệ phải trả, <b>LIQ</b> thanh khoản, <b>SIZ</b> quy mô, <b>DLI</b> dummy thanh khoản, <b>TLI</b> tương tác TPR×LIQ và <b>TLD</b> tương tác TPR×DLI. Corr nằm từ −1 đến +1 và chỉ đo quan hệ tuyến tính cặp đôi. Tương quan cao không chứng minh quan hệ nhân quả.</p>{R10}</section>
<section class="card"><h2>11. VIF</h2><h3>Mô hình 1 — không có tương tác</h3><div class="scroll">{table_html(vif_base_short)}</div><h3>Mô hình 2 — tương tác liên tục</h3><div class="scroll">{table_html(vif_inter_short)}</div><h3>Mô hình 3 — tương tác dummy</h3><div class="scroll">{table_html(vif_dummy_short)}</div><p>VIF tính từ các biến giải thích chính, không tính biến giả năm. Quy tắc 5 hoặc 10 chỉ là mốc tham khảo. {vif_reading} VIF lớn nhất của mô hình dummy là {max_vif_dummy:.2f}.</p>{R11}</section>
<section class="card"><h2>Hướng dẫn: OLS là gì và cách đọc bảng hồi quy</h2>
<p><b>OLS (Ordinary Least Squares – bình phương nhỏ nhất thông thường)</b> là phương pháp tìm đường thẳng, hoặc mặt phẳng khi có nhiều biến, mô tả tốt nhất quan hệ giữa biến phụ thuộc Y và các biến giải thích X. "Tốt nhất" nghĩa là tổng bình phương phần dư nhỏ nhất, trong đó phần dư là chênh lệch giữa ATO thực tế và ATO mô hình dự báo.</p><p><b>PRF và SRF.</b> Hồi quy tổng thể (PRF) mô tả quan hệ thật trong toàn bộ tổng thể với các hệ số α chưa biết. Ta chỉ có mẫu, nên ước lượng hồi quy mẫu (SRF) để có α̂ và mong muốn E(α̂) = α. Trong mô hình, Y là biến phụ thuộc, α<sub>0</sub> là hệ số chặn, các X gồm biến độc lập chính (gắn với mục tiêu nghiên cứu, ở đây là TPR và tương tác với LIQ) và biến kiểm soát (gắn với việc giải thích Y, ở đây là SIZ và hiệu ứng năm). α<sub>1</sub> là <i>tác động riêng biệt</i> (isolated effect) của X<sub>1</sub> lên Y khi các yếu tố khác không đổi (ceteris paribus). Nếu một phần của X<sub>1</sub> nằm trong sai số ε thì Cov(ε, X<sub>1</sub>) ≠ 0: đó là hiện tượng nội sinh (endogeneity), khi đó α̂<sub>1</sub> không còn là tác động riêng biệt.</p>
<div class="formula">Y<sub>i,t</sub> = α<sub>0</sub> + α<sub>1</sub>·X<sub>1,i,t</sub> + α<sub>2</sub>·X<sub>2,i,t</sub> + … + α<sub>k</sub>·X<sub>k,i,t</sub> + ε<sub>i,t</sub> &nbsp;⟶&nbsp; chọn α̂ sao cho Σ ε̂<sup>2</sup><sub>i,t</sub> nhỏ nhất</div>
<p><b>Cách đọc từng cột trong bảng hồi quy:</b></p>
<ul>
<li><b>Hệ số OLS (α):</b> khi biến đó tăng 1 đơn vị và các biến khác giữ nguyên, ATO trung bình thay đổi α lần. TPR là tỷ lệ (0,10 = 10%), nên tăng 1 đơn vị nghĩa là tăng 100 điểm phần trăm; dễ đọc hơn là chia α cho 100 để có tác động của 1 điểm %.</li>
<li><b>SE gom cụm:</b> sai số chuẩn, đo độ không chắc chắn của α. Gom cụm theo mã vì cùng một công ty xuất hiện nhiều năm và sai số của nó có thể liên quan với nhau.</li>
<li><b>t/z:</b> α chia SE. Trị tuyệt đối lớn hơn khoảng 1,96 tương ứng p-value nhỏ hơn 0,05.</li>
<li><b>p-value:</b> nếu thực tế α = 0, xác suất thu được ước lượng xa 0 như vậy là bao nhiêu. Quy tắc: so p-value với mức ý nghĩa α = 1%, 5% hoặc 10%; p-value &lt; α thì bác bỏ H<sub>0</sub>: hệ số = 0, tức hệ số có ý nghĩa thống kê. p-value không phải xác suất giả thuyết đúng.</li>
<li><b>CI 95%:</b> khoảng tin cậy 95% của α. Nếu khoảng không chứa 0 thì hệ số có ý nghĩa ở mức 5%.</li>
<li><b>Hằng số (CON):</b> ATO dự báo khi các biến giải thích bằng 0; vì TPR và LIQ đã trừ trung bình nên ở đây là ATO tại mức TPR, LIQ trung bình, chỉ dùng để tính dự báo.</li>
<li><b>Y22–Y25:</b> chênh lệch ATO của từng năm so với năm gốc 2021, sau khi kiểm soát các biến khác.</li>
<li><b>R² và R² hiệu chỉnh:</b> tỷ lệ khác biệt về ATO mà mô hình giải thích được; R² hiệu chỉnh trừ bớt phần tăng do thêm biến.</li>
</ul>
<p><b>Giả định quan trọng nhất</b> để đọc α là quan hệ nhân quả: sai số ε không tương quan với các biến X. Điều này khó đúng với dữ liệu doanh nghiệp vì còn nhiều yếu tố bị bỏ sót như ngành hay năng lực quản lý, nên bài này chỉ diễn giải α là <i>mối liên hệ</i>. Các giả định khác gồm quan hệ tuyến tính và không có đa cộng tuyến hoàn hảo (kiểm tra bằng VIF ở mục 11); phương sai sai số không đều được xử lý bằng sai số chuẩn gom cụm. Mục 20 đánh giá lần lượt các giả định A1–A5.</p></section>
<section class="card"><h2>12. OLS thông thường: chỉ Y và X</h2><div class="scroll">{table_html(ols0_short)}</div><p>Mô hình này cho thấy quan hệ thô; chưa đưa thanh khoản, quy mô và năm vào nên không dùng làm kết luận cuối.</p>{R12}</section>
<section class="card"><h2>13. Mô hình cơ sở chưa có tương tác</h2><div class="scroll">{table_html(ols1_short)}</div>{R13}</section>
<section class="card"><h2>14. Mô hình có tương tác liên tục</h2><div class="scroll">{table_html(ols2_short)}</div>{R14}</section>
<section class="card"><h2>15. Mô hình có tương tác với dummy</h2><div class="scroll">{table_html(ols3_short)}</div><p>Hệ số <code>TLD</code> = <b>{dummy_int_beta:.4f}</b> (p ≈ {dummy_int_p:.4g}) là chênh lệch độ dốc giữa nhóm DLI=1 và DLI=0, không phải mức chênh lệch ATO đơn thuần.</p>{R15}</section>
<section class="card"><h2>16. Biểu đồ tương tác liên tục kiểu marginsplot</h2>{chart_interaction_inline}<p>Ba đường là giá trị Y dự báo ở thanh khoản p25, p50 và p75, giữ quy mô tại trung bình và năm ở mức gốc 2021. Đường không phải dữ liệu thô; nó được tính từ hệ số mô hình 2 để nhìn trực tiếp ý nghĩa của tương tác.</p>{R16}</section>
<section class="card"><h2>17. Biểu đồ tương tác với dummy 0–1</h2>{chart_dummy_inline}<p>Hai đường cho biết quan hệ dự báo giữa X và Y khác nhau thế nào giữa nhóm không đủ và đủ tài sản ngắn hạn trang trải nợ ngắn hạn.</p>{R17}</section>
<section class="card"><h2>18. Kiểm tra và diễn giải kết quả</h2><div class="scroll">{table_html(model_info)}</div><p>{interaction_reading}</p><p>Cụ thể, mô hình 2 cho α1 = <b>{ap_beta:.4f}</b> (p ≈ {ap_p:.4g}) và α4 = <b>{int_beta:.4f}</b> (p ≈ {int_p:.4g}). R² tăng từ {info1['R2']:.4f} lên {info2['R2']:.4f}, tức mô hình có tương tác giải thích thêm một phần biến thiên nhưng mức R² vẫn còn thấp. Đây là mối liên hệ có điều kiện trong mẫu, không phải bằng chứng nhân quả.</p><p>{liq_reading}</p><p>Cột <code>t/z</code> là hệ số chia cho sai số chuẩn gom cụm; p-value/CI dùng xấp xỉ chuẩn với mẫu lớn. Không diễn giải p-value như xác suất giả thuyết đúng.</p><p class="note"><b>Giới hạn:</b> phân ngành mới ở mức mẫu biểu BCTC; mô hình pooled chưa kiểm soát ngành nên hệ số có thể phản ánh khác biệt giữa các nhóm doanh nghiệp (xem mục 19); VIF của mô hình tương tác liên tục cao; dấu kết quả khác bài tham khảo. Cần thận trọng khi suy rộng.</p>{R18}</section>
<section class="card"><h2>19. Kiểm tra độ bền: hiệu ứng cố định công ty và biến trễ 1 năm</h2><p>Pooled OLS so sánh cả giữa các công ty lẫn trong cùng công ty. Công ty thương mại, bán lẻ thường vừa có tỷ lệ phải trả cao vừa có vòng quay tài sản cao, nên một phần quan hệ trong pooled OLS có thể đến từ khác biệt ngành. Mô hình hiệu ứng cố định (FE) trừ trung bình riêng của từng công ty, nên chỉ dùng biến động <i>trong cùng một công ty qua các năm</i>; mọi đặc điểm không đổi theo thời gian như ngành đều được kiểm soát. Hằng số bị hấp thụ vào hiệu ứng cố định; công ty chỉ có một năm bị loại khỏi FE.</p>
<div class="formula">ATO<sub>i,t</sub> = α<sub>1</sub>·TPRc<sub>i,t</sub> + α<sub>2</sub>·LIQc<sub>i,t</sub> + α<sub>3</sub>·SIZ<sub>i,t</sub> + … + δ<sub>t</sub> + f<sub>i</sub> + ε<sub>i,t</sub></div><p class="legend">f<sub>i</sub>: hiệu ứng cố định công ty, hấp thụ mọi đặc điểm không đổi theo thời gian của từng công ty (ngành, mô hình kinh doanh, năng lực quản lý ổn định). δ<sub>t</sub>: hiệu ứng cố định năm. Ước lượng bằng phép biến đổi within: trừ trung bình riêng của từng công ty khỏi Y và mọi biến X.</p>
<div class="scroll">{table_html(robust_compare)}</div><p>Ô bảng: hệ số (SE gom cụm theo mã). *** p &lt; 0.01, ** p &lt; 0.05, * p &lt; 0.10. R² của FE là R² within. Cột "M2 trễ 1 năm" dùng TPR, LIQ, SIZ của năm trước (t − 1) để giải thích ATO năm t, nên mẫu chỉ gồm 2022–2025 và năm gốc là 2022.</p>
<p>{fe_reading}</p><p class="note">Phần lớn độ lớn của hệ số trong pooled OLS đến từ khác biệt giữa các doanh nghiệp. Kết luận nên dựa trên kết quả nhất quán giữa hai cách ước lượng; kết quả chỉ xuất hiện trong pooled OLS cần được diễn giải là khác biệt giữa doanh nghiệp, không phải tác động khi một doanh nghiệp thay đổi tỷ lệ phải trả.</p>{R19}</section>
<section class="card"><h2>20. Đánh giá mô hình: kỳ vọng dấu và giả định OLS</h2>
<h3>Kỳ vọng dấu so với kết quả</h3><p class="legend">Kết quả lấy từ mô hình 2 (TLD từ mô hình 3). *** p &lt; 0,01; ** p &lt; 0,05; * p &lt; 0,10.</p><div class="scroll">{table_html(sign_table, text=True)}</div>
{R20a}
<h3>Kiểm tra các giả định A1–A5</h3><div class="scroll">{table_html(assumption_table, text=True)}</div>
{R20b}</section>
<section class="card"><h2>Nguồn và tính tái lập</h2><p>Nguồn dữ liệu báo cáo tài chính: <a href="https://stock.quanganh.org/vi/downloads">trang tải dữ liệu cổ phiếu Việt Nam của Quang Anh</a>; nguồn được trang này ghi cho chỉ số cổ phiếu là Vietcap và bổ sung SSI đối với một số chuỗi chỉ số. Bộ dữ liệu kèm theo trong thư mục <code>data_source</code>.</p><p>Bài tham khảo: <a href="https://nckh.tapchicongthuong.vn/j/nckh/articles/3252-the-effects-of-trade-payables-on-the-performance-of-firms-listed-on-the-vietnam-stock-exchange?lang=en-US">The effects of trade payables on the performance of firms listed on the Vietnam Stock Exchange</a>. Tóm tắt bài báo đề cập panel 2011–2022, thanh khoản điều tiết quan hệ và sử dụng OLS, FE, RE, 2SLS. Bài này dùng 2021–2025 để thực hành, vì vậy khác kỳ nghiên cứu gốc.</p></section>
</div>
<div id="tab-tables" hidden>{gallery_html}</div>
{models_tab_html}
{data_tab_html}
{code_tab_html}
<script>function showTab(t){{for(const k of ['full','models','tables','data','code']){{document.getElementById('tab-'+k).hidden=k!==t;document.getElementById('btn-'+k).setAttribute('aria-selected',String(k===t));}}const h={{tables:'#bang-bieu',data:'#du-lieu',code:'#code',models:'#mo-hinh'}};try{{history.replaceState(null,'',h[t]||location.pathname);}}catch(e){{}}window.scrollTo(0,0);}}if(location.hash==='#bang-bieu')showTab('tables');if(location.hash==='#du-lieu')showTab('data');if(location.hash==='#code')showTab('code');if(location.hash==='#mo-hinh')showTab('models');</script>
<footer><small>Tạo bởi main.py • phân vị Winsor: p1 và p99 • winsor không xóa công ty-năm</small></footer></main></body></html>'''
    (OUT / "bao_cao_ket_qua.html").write_text(report, encoding="utf-8")
    # File riêng chỉ gồm các bảng và biểu đồ (giống tab "Chỉ bảng và biểu đồ"), dùng để in hoặc xuất PDF.
    import re as _re
    page_style = _re.search(r"<style>.*?</style>", report, _re.S).group(0)
    gallery_doc = ('<!doctype html><html lang="vi"><head><meta charset="utf-8">'
                   '<meta name="viewport" content="width=device-width,initial-scale=1"><title>Bảng và biểu đồ – Trade payables</title>'
                   + page_style +
                   '<style>@media print{.card{box-shadow:none;break-inside:avoid;margin:0 0 10px}body{background:#fff}main{margin:0;max-width:none;padding:0}h3{break-after:avoid}.inline-chart{max-height:150mm;width:auto;max-width:100%;margin:6px auto}}</style>'
                   '</head><body><main><header class="card"><h1>Bảng và biểu đồ</h1>'
                   '<p>Trade payables, thanh khoản và hiệu quả hoạt động • FY2021–FY2025</p></header>'
                   + gallery_html + '</main></body></html>')
    (OUT / "bang_va_bieu_do.html").write_text(gallery_doc, encoding="utf-8")
    print(f"Hoàn tất: {len(wins):,} quan sát, {wins['ticker'].nunique():,} mã, {YEARS[0]}–{YEARS[-1]}.")
    print(f"Đã ghi báo cáo: {OUT / 'bao_cao_ket_qua.html'}")


if __name__ == "__main__":
    main()
