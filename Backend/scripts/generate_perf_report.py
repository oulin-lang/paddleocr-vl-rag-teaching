import json
import os

def main():
    src = os.path.join("test_artifacts", "perf.json")
    if not os.path.exists(src):
        print("perf.json not found")
        return
    data = json.load(open(src, "r", encoding="utf-8"))
    bars = []
    for k, v in data.items():
        width = min(int(v // 5), 400)
        color = "#4caf50" if v < 500 else ("#ff9800" if v < 1500 else "#f44336")
        bars.append(f"<div style='margin:8px 0'><strong>{k}</strong> {v}ms<div style='background:{color};height:16px;width:{width}px'></div></div>")
    html = """
    <html><head><meta charset='utf-8'><title>Performance Report</title></head>
    <body>
    <h2>Performance Metrics</h2>
    {bars}
    </body></html>
    """.replace("{bars}", "\n".join(bars))
    out = os.path.join("test_artifacts", "perf.html")
    open(out, "w", encoding="utf-8").write(html)
    print("wrote:", out)

if __name__ == "__main__":
    main()
