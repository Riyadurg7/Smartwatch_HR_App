"""
Click through the Streamlit app headlessly for each data source and fail if
anything crashes.   python test_app.py
"""
import warnings

warnings.filterwarnings("ignore")
from streamlit.testing.v1 import AppTest

SOURCES = ["Sample: 7 days with activity & sleep", "Sample: heart rate only", "Synthetic (generated)"]
ok = True
for i, label in enumerate(SOURCES):
    at = AppTest.from_file("app.py", default_timeout=300)
    at.run()
    at.sidebar.radio[0].set_value(label).run()
    at.sidebar.button[0].click().run()
    errs = [e.value for e in at.error]
    warns = [w.value[:80] for w in at.warning]
    deprecations = [w for w in warns if "deprecat" in w.lower()]
    good = not at.exception and not errs and not deprecations and len(at.metric) >= 8
    ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] {label:38s} metrics={len(at.metric)} tables={len(at.dataframe)}"
          + (f" warnings={warns}" if warns else ""))
    if at.exception:
        print(at.exception)
    if i == 0:
        btn = [b for b in at.button if "benchmark" in b.label.lower()]
        if btn:
            btn[0].click().run()
        good = bool(btn) and not at.exception
        ok &= good
        print(f"[{'PASS' if good else 'FAIL'}] anomaly benchmark button   "
              + ", ".join(m.label + "=" + m.value for m in at.metric[-5:]))

print("\nAPP CHECKS PASSED" if ok else "\nAPP CHECKS FAILED")
raise SystemExit(0 if ok else 1)
