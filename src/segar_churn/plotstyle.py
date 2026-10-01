"""One shared chart style so every figure in the reports reads as one system."""
import matplotlib.pyplot as plt

INK = "#1F2A2E"
MUTED = "#8A8F91"
GRID = "#ECEAE4"
PALETTE = ["#0E7C66", "#D9822B", "#3B6EA8", "#B5446E", "#6B8E23", "#7A5C99"]
RISK = {"High": "#C2412D", "Medium": "#D9922B", "Low": "#7DA494"}


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#C9C6BE",
        "axes.labelcolor": INK, "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
        "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
        "axes.prop_cycle": plt.cycler(color=PALETTE),
    })
