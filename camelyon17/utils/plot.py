import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


sns.set_context(context='paper', font_scale=2)


def next_color(ax):
    return next(ax._get_lines.prop_cycler)['color']