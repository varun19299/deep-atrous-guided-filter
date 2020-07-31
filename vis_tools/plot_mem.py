import numpy as np
from plotnine import *  # ggplot2
from scipy import interpolate
import pandas as pd
import matplotlib.pyplot as plt

x_ours = np.array([32, 64, 96, 128, 256, 320, 512, 640, 800, 1024])
y_ours = np.array([550, 574, 632, 688, 1086, 1370, 2716, 3870, 6026, 8608])

x_lr = np.array([32, 64, 96, 128, 256, 320, 512, 640])
y_lr = np.array([574, 666, 804, 960, 2192, 2994, 6438, 10030])

f_ours = interpolate.interp1d(x_ours, y_ours, kind="quadratic")
f_lr = interpolate.interp1d(x_lr, y_lr, kind="quadratic", fill_value="extrapolate")

x = np.logspace(5, 10, num=41, base=2)
y_int_ours = f_ours(x)
y_int_lr = f_lr(x)

x_disp = np.logspace(5, 10, num=6, base=2)
y_disp_ours = f_ours(x_disp) / 1024
y_disp_lr = f_lr(x_disp) / 1024

x_disp = np.array(["32x64", "64x128", "128x256", "256x512", "512x1024", "1024x2048"])

df = pd.DataFrame({"x": x, "y_ours": y_int_ours, "y_lr": y_int_lr})

df_disp = pd.DataFrame({"y_disp_ours": y_disp_ours, "y_disp_lr": y_disp_lr})

df_cat = pd.Categorical(x_disp, categories=x_disp)
df_disp["x_disp"] = df_cat
print(df_disp)

g = (
    ggplot(df_disp)
    # add connected points
    + geom_path(
        mapping=aes(x="x_disp", y="y_disp_ours", colour=["DAGF"]),
        lineend="projecting",
        group=1,
        size=1,
    )
    + geom_point(
        mapping=aes(x="x_disp", y="y_disp_ours"), data=df_disp, color="blue", size=3
    )
    + scale_color_discrete(l=0.4)
    + geom_path(
        mapping=aes(x="x_disp", y="y_disp_lr", colour=["LR Net"]),
        lineend="projecting",
        group=1,
        size=1,
    )
    + geom_point(
        mapping=aes(x="x_disp", y="y_disp_lr"), data=df_disp, color="orange", size=3
    )
    + scale_colour_manual(["blue", "orange"])
    + xlab("Image resolution")
    + ylab("Memory consumption (in GB)")
    + labs(colour="Model")
    # change axis limits without removing the points
    + coord_cartesian(xlim=None, ylim=(0, 12))
    # set font style
    + theme(
        legend_position="top",
        axis_line=element_line(color="grey"),
        axis_ticks=element_line(color="grey"),
        text=element_text(family="calibri"),
    )
)

fig = g.draw()

plt.show()

g.save("../paper_figs/mem_consump.png", dpi=1000)
