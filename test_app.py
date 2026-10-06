"""Click through the Streamlit app headlessly for each data source.  python test_app.py"""
import warnings; warnings.filterwarnings("ignore")
from streamlit.testing.v1 import AppTest
for i, label in enumerate(["Sample: 7 days with activity & sleep", "Sample: heart rate only", "Synthetic (generated)"]):
    at = AppTest.from_file("app.py", default_timeout=300)
    at.run()
    at.sidebar.radio[0].set_value(label).run()
    at.sidebar.button[0].click().run()
    errs = [e.value for e in at.error]
    print(f"{label:40s} exception={bool(at.exception)} errors={errs} metrics={len(at.metric)} "
          f"tables={len(at.dataframe)} infos={len(at.info)}")
    if at.exception: print(at.exception)
    if i == 0:
        # click the benchmark button in Model & Evaluation tab
        btn = [b for b in at.button if "benchmark" in b.label.lower()]
        btn[0].click().run()
        print("  benchmark exception:", bool(at.exception), [m.label + "=" + m.value for m in at.metric][-5:])
