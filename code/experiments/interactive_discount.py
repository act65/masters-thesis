import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider

import trl_utils as trl

initial_discount = .5
M_pis = [trl.generate_rnd_policy(2,2) for _ in range(5000)]
P, r = trl.generate_rnd_problem(2,2)
Vs = np.hstack([trl.value_functional(P, r, M_pi, initial_discount) for M_pi in M_pis])


fig, ax = plt.subplots()
ax.autoscale(enable=True)
l = plt.plot(Vs[0, :], Vs[1, :], 'b.')[0]
# ax = plt.axis([0,10,0,10])

axdisc = plt.axes([0.25, .03, 0.50, 0.02])
# Slider
samp = Slider(axdisc, 'Discount', 0, 1, valinit=initial_discount)

def update(val):
    # amp is the current value of the slider
    discount = samp.val

    # recompute the values
    Vs = np.hstack([trl.value_functional(P, r, M_pi, discount) for M_pi in M_pis])

    # update curve
    l.set_xdata(Vs[0, :])
    l.set_ydata(Vs[1, :])


    # TODO want the axes to scale with the shape
    # plt.axis([np.min(Vs[0, :]), np.max(Vs[0, :]),np.min(Vs[1, :]), np.max(Vs[1, :])])

    # redraw canvas while idle
    fig.canvas.draw_idle()

# call update function on slider value change
samp.on_changed(update)

plt.show()
# TODO want to turn into an animation!
