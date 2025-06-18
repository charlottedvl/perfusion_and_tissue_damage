# pystats.py
import time

class _Stats:
    def __init__(self):
        self.timers = {}
        self.enabled = False

    def enable(self):
        self.timers = {}
        self.enabled = True

    def timer_start(self, name):
        if self.enabled:
            self.timers[name] = {
                "start": time.perf_counter(),
                "elapsed": self.timers.get(name, {}).get("elapsed", 0)
            }

    def timer_end(self, name):
        if self.enabled and name in self.timers and "start" in self.timers[name]:
            elapsed = time.perf_counter() - self.timers[name]["start"]
            self.timers[name]["elapsed"] += elapsed
            self.timers[name].pop("start", None)

    def report(self):
        if not self.enabled:
            return "Stats not enabled."
        report_lines = ["\n--- Performance Stats ---"]
        for name, info in self.timers.items():
            report_lines.append(f"{name:30s}: {info['elapsed']:.4f} s")
        return "\n".join(report_lines)

stats = _Stats()
  def save_pdf(self, filename="performance_stats.pdf"):
      import io
      from reportlab.lib.pagesizes import A4
      from reportlab.pdfgen import canvas
      from reportlab.lib.utils import ImageReader
      import matplotlib.pyplot as plt

      if not self.enabled:
        print("Stats not enabled.")
        return

      # Prepare the canvas
      c = canvas.Canvas(filename, pagesize=A4)
      width, height = A4
      y = height - 50
      c.setFont("Helvetica", 12)
      c.drawString(50, y, "--- Performance Stats ---")
      y -= 30

      # Draw plain text summary
      for name, info in self.timers.items():
          c.drawString(50, y, f"{name:30s}: {info['elapsed']:.4f} s")
          y -= 20
          if y < 100:
              c.showPage()
              y = height - 50

      # Generate pie chart
      names = list(self.timers.keys())
      values = [self.timers[n]["elapsed"] for n in names]

      fig, ax = plt.subplots(figsize=(6, 6))
      ax.pie(values, labels=names, autopct='%1.1f%%', startangle=140)
      ax.set_title("Time Distribution per Task")
      plt.tight_layout()

      # Save chart to buffer
      buf = io.BytesIO()
      plt.savefig(buf, format="png")
      plt.close(fig)
      buf.seek(0)
      img = ImageReader(buf)

      # Insert chart into PDF
      c.showPage()
      c.drawString(50, height - 50, "Timing Pie Chart:")
      c.drawImage(img, 50, 100, width=400, preserveAspectRatio=True, mask='auto')

      # Finalize PDF
      c.save()
      buf.close()
      print(f"Performance stats with pie chart saved to {filename}")
