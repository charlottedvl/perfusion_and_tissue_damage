# mpystats.py
import tracemalloc

class _Stats:
    def __init__(self):
        self.memory_snapshots = {}
        self.enabled = False

    def enable(self):
        self.memory_snapshots = {}
        self.enabled = True
        tracemalloc.start()

    def timer_start(self, name):
        if self.enabled:
            self.memory_snapshots[name] = {
                "start": tracemalloc.take_snapshot(),
                "peak_kb": self.memory_snapshots.get(name, {}).get("peak_kb", 0)
            }

    def timer_end(self, name):
        if self.enabled and name in self.memory_snapshots and "start" in self.memory_snapshots[name]:
            end_snapshot = tracemalloc.take_snapshot()
            start_snapshot = self.memory_snapshots[name]["start"]
            stats = end_snapshot.compare_to(start_snapshot, 'filename')
            total_mem_kb = sum([stat.size_diff for stat in stats]) / 1024  # Convert to KB
            self.memory_snapshots[name]["peak_kb"] += total_mem_kb
            self.memory_snapshots[name].pop("start", None)

    def report(self):
        if not self.enabled:
            return "Stats not enabled."
        report_lines = ["\n--- Memory Usage Stats (KB) ---"]
        for name, info in self.memory_snapshots.items():
            report_lines.append(f"{name:30s}: {info['peak_kb']:.2f} KB")
        return "\n".join(report_lines)

    def save_pdf(self, filename="memory_usage_report.pdf"):
        import io
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.utils import ImageReader
        import matplotlib.pyplot as plt

        if not self.enabled:
            print("Stats not enabled.")
            return

        # Create canvas
        c = canvas.Canvas(filename, pagesize=A4)
        width, height = A4
        y = height - 50
        c.setFont("Helvetica", 12)
        c.drawString(50, y, "--- Memory Usage Stats (KB) ---")
        y -= 30

        # Text report
        for name, info in self.memory_snapshots.items():
            c.drawString(50, y, f"{name:30s}: {info['peak_kb']:.2f} KB")
            y -= 20
            if y < 100:
                c.showPage()
                y = height - 50

        # Prepare data
        names = list(self.memory_snapshots.keys())
        values = [self.memory_snapshots[n]["peak_kb"] for n in names]
        total = sum(values)
        percentages = [(v / total) * 100 for v in values]

        # Donut chart
        fig, ax = plt.subplots(figsize=(7, 7))
        threshold = 2  # Only label slices > 2%
        explode = [0.05 if p == max(percentages) else 0 for p in percentages]
        labels = [f"{n}: {p:.1f}%" if
