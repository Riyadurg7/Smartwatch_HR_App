"""
Report generation (Objective 2/5 output: downloadable PDF/CSV summaries).

`build_pdf(result, title)` returns PDF bytes ready for Streamlit's
st.download_button. `build_csv_bundle(result)` returns a dict of CSV strings.

Charts are rendered with matplotlib (headless) and embedded, so no browser
engine (kaleido) is required - this keeps the Hugging Face / Streamlit Cloud
deployment lightweight.
"""

from __future__ import annotations

import io
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage,
)


def _fig_to_image(fig, width_cm=16) -> RLImage:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img = RLImage(buf)
    scale = (width_cm * cm) / img.drawWidth
    img.drawWidth *= scale
    img.drawHeight *= scale
    return img


def _timeline_fig(result):
    clean = result.clean
    fig, ax = plt.subplots(figsize=(9, 2.6))
    ax.plot(clean["timestamp"], clean["hr_smooth"], lw=0.6, color="#2563eb", label="HR (smoothed)")
    if "ma_1h" in clean:
        ax.plot(clean["timestamp"], clean["ma_1h"], lw=1.0, color="#f59e0b", label="1h avg")
    anom = pd.DataFrame(result.anomalies.get("table", []))
    if not anom.empty and "is_anomaly" in anom:
        fl = anom[anom["is_anomaly"]]
        ax.scatter(fl["window_start"], fl["mean"], s=18, c="#dc2626", marker="x", label="anomaly", zorder=5)
    ax.set_ylabel("bpm"); ax.set_title("Heart-rate timeline"); ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    return fig


def _circadian_fig(result):
    circ = result.circadian
    fig, ax = plt.subplots(figsize=(4.3, 2.4))
    ax.plot(circ["hour"], circ["mean_hr"], marker="o", ms=3, color="#2563eb")
    ax.set_xlabel("hour of day"); ax.set_ylabel("avg bpm"); ax.set_title("Circadian profile")
    ax.grid(alpha=0.25)
    return fig


def _hist_fig(result):
    clean = result.clean
    fig, ax = plt.subplots(figsize=(4.3, 2.4))
    ax.hist(clean["hr_smooth"].dropna(), bins=40, color="#2563eb", alpha=0.8)
    ax.set_xlabel("bpm"); ax.set_ylabel("count"); ax.set_title("HR distribution")
    ax.grid(alpha=0.25)
    return fig


def build_pdf(result, title: str = "Smartwatch Heart-Rate Report", subject: str = "") -> bytes:
    """Return a PDF summary as bytes."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    styles = getSampleStyleSheet()
    h1 = styles["Title"]
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], spaceBefore=10, textColor=colors.HexColor("#1e3a8a"))
    body = styles["BodyText"]
    small = ParagraphStyle("small", parent=body, fontSize=8, textColor=colors.grey)

    el = []
    el.append(Paragraph(title, h1))
    meta = f"Generated {datetime.now():%Y-%m-%d %H:%M}"
    if subject:
        meta = f"Subject: {subject} &nbsp;&nbsp;|&nbsp;&nbsp; " + meta
    el.append(Paragraph(meta, small))
    el.append(Spacer(1, 6))

    # --- data quality ---
    q = result.quality
    el.append(Paragraph("Data quality", h2))
    qtab = [
        ["Quality score", f"{q['quality_score']}/100 ({q['quality_label']})"],
        ["Coverage", f"{q['coverage_pct']}%"],
        ["Duration", f"{q['duration_hours']} h"],
        ["Sampling grid", str(q["resample_rule"])],
        ["Readings after cleaning", f"{q['n_after_clean']:,}"],
        ["Implausible removed", str(q["n_implausible_removed"])],
    ]
    el.append(_kv_table(qtab))
    for w in q.get("warnings", []):
        el.append(Paragraph("• " + w, small))

    # --- key stats ---
    s = result.stats
    el.append(Paragraph("Key statistics", h2))
    stab = [
        ["Resting HR", f"{s['resting_hr']} bpm"],
        ["Lowest during sleep", f"{s.get('overnight_low_hr','n/a')} bpm"],
        ["Average HR", f"{s['mean_hr']} bpm"],
        ["Peak HR (99th pct)", f"{s['max_hr']} bpm"],
        ["HR range (5-95 pct)", f"{s['p05']}-{s['p95']} bpm"],
        ["HRV proxy", f"{s['hrv_proxy_ms']} ms"],
    ]
    el.append(_kv_table(stab))
    if s.get("resting_method"):
        el.append(Paragraph("Resting HR method: " + s["resting_method"], small))

    # --- activity & sleep context ---
    ctx = getattr(result, "context", {}) or {}
    if ctx.get("available"):
        el.append(Paragraph("Activity & sleep", h2))
        src = {"observed": "from activity file", "estimated": "estimated from heart rate by the trained model"}
        el.append(Paragraph(
            f"Activity levels: {src.get(ctx['intensity_source'], ctx['intensity_source'])}. "
            f"Sleep: {ctx['sleep_source']}.", small))
        rows = [["", "time share", "median HR"]]
        for r in ctx.get("hr_by_activity", []):
            rows.append([r["activity"], f"{r['pct']}%",
                         "n/a" if r["median_hr"] is None or r["median_hr"] != r["median_hr"] else f"{r['median_hr']} bpm"])
        t = Table(rows, colWidths=[6 * cm, 4 * cm, 6 * cm])
        t.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 9), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f4f6")]),
            ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.grey),
        ]))
        el.append(t)
        eps = ctx.get("elevated_rest_episodes", [])
        el.append(Paragraph(
            f"Periods of high heart rate while still and awake: {len(eps)}"
            + (f" ({sum(e['minutes'] for e in eps)} min in total)." if eps else "."), body))

    # --- timeline chart ---
    el.append(Paragraph("Heart-rate timeline", h2))
    el.append(_fig_to_image(_timeline_fig(result)))

    # --- side-by-side distribution + circadian ---
    row = Table(
        [[_fig_to_image(_hist_fig(result), width_cm=7.6), _fig_to_image(_circadian_fig(result), width_cm=7.6)]],
        colWidths=[8 * cm, 8 * cm],
    )
    el.append(row)

    # --- interpretation ---
    it = result.interpretation
    el.append(Paragraph("Automated interpretation", h2))
    el.append(Paragraph(it["summary_text"], body))
    if it.get("trend_text"):
        el.append(Paragraph(it["trend_text"], body))
    el.append(Spacer(1, 4))
    for al in it["alerts"]:
        tag = "ATTENTION" if al["level"] == "attention" else "Note"
        col = colors.HexColor("#b91c1c") if al["level"] == "attention" else colors.HexColor("#374151")
        el.append(Paragraph(f'<b><font color="{col}">{tag}:</font></b> {al["message"]}', body))

    # --- ML summary ---
    a = result.anomalies
    c = result.clusters
    el.append(Paragraph("Machine-learning analysis", h2))
    el.append(Paragraph(
        f"Anomaly detection ({a.get('method','-')}): {a.get('n_anomalies',0)} of "
        f"{a.get('n_windows',0)} time windows flagged as unusual. "
        f"Clustering grouped activity into {c.get('k',0)} behavioural regimes.",
        body,
    ))

    # --- disclaimer ---
    el.append(Spacer(1, 10))
    el.append(Paragraph("Disclaimer", h2))
    el.append(Paragraph(it["disclaimer"], small))

    doc.build(el)
    buf.seek(0)
    return buf.read()


def _kv_table(rows):
    t = Table(rows, colWidths=[6 * cm, 10 * cm])
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#374151")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#f3f4f6")]),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3), ("TOPPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def build_csv_bundle(result) -> dict:
    """Return {filename: csv_string} for the main tables."""
    out = {}
    if not result.clean.empty:
        out["cleaned_heartrate.csv"] = result.clean.to_csv(index=False)
    if not result.daily.empty:
        out["daily_summary.csv"] = result.daily.to_csv(index=False)
    anom = pd.DataFrame(result.anomalies.get("table", []))
    if not anom.empty:
        out["anomalies.csv"] = anom.to_csv(index=False)
    if not result.peaks.empty:
        out["exercise_episodes.csv"] = result.peaks.to_csv(index=False)
    return out
