# bspystats.py
import time
import io
import os
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import matplotlib.pyplot as plt

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
        report_lines = ["\n--- BSPerformance Stats ---"]
        for name, info in self.timers.items():
            report_lines.append(f"{name:30s}: {info['elapsed']:.4f} s")
        return "\n".join(report_lines)

    def save_pdf(self, filename="bsperformance_stats.pdf"):
        if not self.enabled:
            print("Stats not enabled.")
            return

        if not os.path.exists(os.path.dirname(filename)) and os.path.dirname(filename) != '':
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        # Prepare the PDF
        c = canvas.Canvas(filename, pagesize=A4)
        width, height = A4
        y = height - 50
        names = list(self.timers.keys())
        values = [self.timers[n]["elapsed"] for n in names]

        preferred_order = ["read_input", "scale_permeabilities", "read_dead_tissue", "setup_solver", "solve_system", "postprocessing"]

        ordered_names = []
        ordered_values = []
        for task in preferred_order:
            if task in names:
                idx = names.index(task)
                ordered_names.append(names[idx])
                ordered_values.append(values[idx])

        fig, ax = plt.subplots(figsize=(9, 7))

        explode = [0.1 if name == 'solve_system' else 0 for name in ordered_names]
        colors = ['#FF6F61', '#FFD700', '#88B04B', '#92A8D1', '#D3D3D3', '#C19A6B']

        wedges, texts, autotexts = ax.pie(
            ordered_values,
            labels=None,
            autopct='%1.1f%%',
            startangle=31,  # Rotate pie chart for better layout
            colors=colors,
            explode=explode,
            pctdistance=0.7
        )

        for text in texts:
            text.set_text("")

        idx_setup = ordered_names.index('setup_solver')
        idx_dead = ordered_names.index('read_dead_tissue')

        angle_setup = (wedges[idx_setup].theta2 + wedges[idx_setup].theta1) / 2.
        angle_dead = (wedges[idx_dead].theta2 + wedges[idx_dead].theta1) / 2.

        x_setup = 0.7 * np.cos(np.radians(angle_setup))
        y_setup = 0.7 * np.sin(np.radians(angle_setup))

        x_dead = 0.7 * np.cos(np.radians(angle_dead))
        y_dead = 0.7 * np.sin(np.radians(angle_dead))

        ax.text(x_setup - 0.01, y_setup + 0.15, f'{ordered_values[idx_setup]/sum(ordered_values)*100:.1f}%', ha='center', va='center', fontsize=10)
        ax.text(x_dead - 0.05, y_dead + 0.3, f'{ordered_values[idx_dead]/sum(ordered_values)*100:.1f}%', ha='center', va='center', fontsize=10)

        autotexts[idx_setup].set_text("")
        autotexts[idx_dead].set_text("")

        ax.legend(wedges, ordered_names, title="Tasks", loc="center left", bbox_to_anchor=(1.25, 0.5), fontsize=12, title_fontsize=14)
        ax.set_title("     Time Distribution per Task basic_flow_solver.py", fontsize=16)
        ax.axis('equal')

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        img = ImageReader(buf)

        for name, info in self.timers.items():
            c.drawString(50, y, f"{name:25s}: {info['elapsed']:.4f} s")
            y -= 15
        # Insert pie chart into PDF
        c.showPage()
        c.drawImage(img, 80, 200, width=480, preserveAspectRatio=True, mask='auto')
        # Now print performance stats below the chart
        y = 400
        c.setFont("Helvetica", 10)
        c.drawString(90, y, "--- Performance Stats ---")
        y -= 20

        for name, info in self.timers.items():
            c.drawString(90, y, f"{name:25s}: {info['elapsed']:.4f} s")
            y -= 15
        c.save()
        buf.close()
        print(f"Performance stats with pie chart saved to {filename}")

# Singleton instance
stats = _Stats()


