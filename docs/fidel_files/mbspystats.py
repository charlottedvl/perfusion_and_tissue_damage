# bspystats.py

import time
import io
import os
import psutil
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
            process = psutil.Process(os.getpid())
            mem_start = process.memory_info().rss / (1024 * 1024)  # in MB
            self.timers[name] = {
                "start": time.perf_counter(),
                "elapsed": self.timers.get(name, {}).get("elapsed", 0),
                "mem_start": mem_start,
                "mem_usage": self.timers.get(name, {}).get("mem_usage", 0)
            }

    def timer_end(self, name):
        if self.enabled and name in self.timers and "start" in self.timers[name]:
            elapsed = time.perf_counter() - self.timers[name]["start"]
            process = psutil.Process(os.getpid())
            mem_end = process.memory_info().rss / (1024 * 1024)  # in MB
            mem_diff = mem_end - self.timers[name]["mem_start"]

            self.timers[name]["elapsed"] += elapsed
            self.timers[name]["mem_usage"] = max(self.timers[name]["mem_usage"], mem_diff)
            self.timers[name].pop("start", None)
            self.timers[name].pop("mem_start", None)

    def report(self):
        if not self.enabled:
            return "Stats not enabled."
        report_lines = ["\n--- BSPerformance Stats ---"]
        for name, info in self.timers.items():
            report_lines.append(f"{name:25s}: {info['elapsed']:.4f} s, {info['mem_usage']:.2f} MB")
        return "\n".join(report_lines)

    def save_txt(self, filename="bsperformance_stats.txt"):
        if not self.enabled:
            print("Stats not enabled.")
            return
        with open(filename, "w") as f:
            f.write(self.report())
        print(f"Performance stats saved to {filename}")

    def save_pdf(self, filename="bs_performance_stats.pdf"):
        if not self.enabled:
            print("Stats not enabled.")
            return

        if not os.path.exists(os.path.dirname(filename)) and os.path.dirname(filename) != '':
            os.makedirs(os.path.dirname(filename), exist_ok=True)

        c = canvas.Canvas(filename, pagesize=A4)
        width, height = A4

        names = list(self.timers.keys())
        values = [self.timers[n]["elapsed"] for n in names]

        preferred_order = [
            "read_input",
            "scale_permeabilities",
            "read_dead_tissue",
            "solve_system",
            "setup_solver",
            "postprocessing"
        ]

        ordered_names = []
        ordered_values = []
        ordered_memory = []
        for task in preferred_order:
            if task in names:
                idx = names.index(task)
                ordered_names.append(names[idx])
                ordered_values.append(values[idx])
                ordered_memory.append(self.timers[names[idx]]['mem_usage'])

        fig, ax = plt.subplots(figsize=(5, 5))

        explode = [0.1 if name == 'solve_system' else 0 for name in ordered_names]
        colors = [
            "#A6CEE3",  # light blue
            "#1F78B4",  # medium blue
            "#B2DF8A",  # light green
            "#66C2A5",  # teal green
            "#33A02C",  # medium green
            "#006400"   # dark green
        ]

        wedges, texts, autotexts = ax.pie(
            ordered_values,
            labels=ordered_names,
            autopct='%1.1f%%',
            startangle=90,
            explode=explode,
            colors=colors,
            textprops={'fontsize': 9}
        )

        ax.set_title("Time Distribution per Task", fontsize=12)
        ax.axis('equal')

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight', dpi=150)
        plt.close(fig)
        buf.seek(0)
        img = ImageReader(buf)

        chart_width = 360
        chart_height = 280
        chart_x = 200
        chart_y = 420
        c.drawImage(img, chart_x, chart_y, width=chart_width, height=chart_height)

        text_x = 100
        text_y = chart_y - 30
        c.setFont("Helvetica-Bold", 11)
        c.drawString(text_x, text_y, "--- Performance Stats ---")
        text_y -= 18
        c.setFont("Helvetica", 10)

        for name in ordered_names:
            time_taken = self.timers[name]["elapsed"]
            mem_usage = self.timers[name]["mem_usage"]
            c.drawString(text_x, text_y, f"{name:25s}: {time_taken:.4f} s, {mem_usage:.2f} MB")
            text_y -= 14

        c.save()
        buf.close()
        print(f"Performance stats with pie chart saved to {filename}")

# Singleton instance
stats = _Stats()
