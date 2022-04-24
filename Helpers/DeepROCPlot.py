#!/usr/bin/env python
# -*- coding: latin-1 -*-
# DeepROCPlot.py
# Written by André Carrington
#
#    Copyright 2022 University of Ottawa
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# plotPartialArea(pfpr, ptpr, showError)
# imports are locally defined in each function

def plotPartialArea(pfpr, ptpr, showError):
    import matplotlib.pyplot as plt

    # plot partial areas in ROC plots or concordance matrix plots

    # define lines for partial area: left line (ll), right line (rl),
    #                                bottom line (bl), top line (tl)
    SWpoint_yStripe = [pfpr[0],  0]
    NWpoint_yStripe = [pfpr[0],  1]
    NEpoint_yStripe = [pfpr[-1], 1]
    SEpoint_yStripe = [pfpr[-1], 0]
    #
    SWpoint_xStripe = [0, ptpr[0] ]
    NWpoint_xStripe = [0, ptpr[-1]]
    NEpoint_xStripe = [1, ptpr[-1]]
    SEpoint_xStripe = [1, ptpr[0] ]
    #
    # take sequences of x and y (horizontal or vertical) and plot them...
    plotLine   = lambda x, y: plt.plot(x, y, '--',     color=(0.5, 0.5, 0.5),   linewidth=1.5)
    plotLine2  = lambda x, y: plt.plot(x, y, '-',      color='black',           linewidth=0.25)
    plotpAUCy  = lambda x, y: plt.fill(x, y, 'xkcd:yellow', alpha=0.5,          linewidth=None)
    plotpAUCx  = lambda x, y: plt.fill(x, y, 'b',      alpha=0.4,               linewidth=None)
    plotClear  = lambda x, y: plt.fill(x, y, 'w',                               linewidth=None)
    plotpAUCxy = lambda x, y: plt.fill(x, y, 'g',      alpha=0.4,               linewidth=None)
    plotError  = lambda x, y: plt.fill(x, y, 'r',      alpha=0.25,              linewidth=None)
    plotErrorxy= lambda x, y: plt.fill(x, y, 'r',      alpha=0.35,              linewidth=None)
    #
    # plot vertical stripe lines
    plotLine([SWpoint_yStripe[0], NWpoint_yStripe[0]], [SWpoint_yStripe[1], NWpoint_yStripe[1]])
    plotLine([SEpoint_yStripe[0], NEpoint_yStripe[0]], [SEpoint_yStripe[1], NEpoint_yStripe[1]])
    plotLine2([SWpoint_yStripe[0], NWpoint_yStripe[0]], [SWpoint_yStripe[1], NWpoint_yStripe[1]])
    plotLine2([SEpoint_yStripe[0], NEpoint_yStripe[0]], [SEpoint_yStripe[1], NEpoint_yStripe[1]])

    # plot vertical AUCy in muted yellow
    x = pfpr + [SEpoint_yStripe[0]] + [SWpoint_yStripe[0]] + [pfpr[0]]
    y = ptpr + [SEpoint_yStripe[1]] + [SWpoint_yStripe[1]] + [ptpr[0]]
    plotpAUCy(x, y)

    # plot vertical   Error in light orange
    if showError:
       x = [NWpoint_yStripe[0]] + [NEpoint_yStripe[0]] + [NEpoint_yStripe[0]] + [NWpoint_yStripe[0]]
       y = [NWpoint_xStripe[1]] + [NEpoint_xStripe[1]] + [NEpoint_yStripe[1]] + [NWpoint_yStripe[1]]
       plotError(x, y)
    #endif

    # plot horizontal stripe lines
    plotLine([SWpoint_xStripe[0], SEpoint_xStripe[0]], [SWpoint_xStripe[1], SEpoint_xStripe[1]])
    plotLine([NWpoint_xStripe[0], NEpoint_xStripe[0]], [NWpoint_xStripe[1], NEpoint_xStripe[1]])
    plotLine2([SWpoint_xStripe[0], SEpoint_xStripe[0]], [SWpoint_xStripe[1], SEpoint_xStripe[1]])
    plotLine2([NWpoint_xStripe[0], NEpoint_xStripe[0]], [NWpoint_xStripe[1], NEpoint_xStripe[1]])

    # plot horizontal AUCx in slightly muted blue
    x = pfpr + [NEpoint_xStripe[0]] + [SEpoint_xStripe[0]] + [pfpr[0]]
    y = ptpr + [NEpoint_xStripe[1]] + [SEpoint_xStripe[1]] + [ptpr[0]]
    plotpAUCx(x, y)

    # plot horizontal Error in light orange
    if showError:
       x = [SWpoint_xStripe[0]] + [SWpoint_yStripe[0]] + [NWpoint_yStripe[0]] + [NWpoint_xStripe[0]]
       y = [SWpoint_xStripe[1]] + [SEpoint_xStripe[1]] + [NEpoint_xStripe[1]] + [NWpoint_xStripe[1]]
       plotError(x, y)
    #endif

    # plot overlap in AUCxy as muted green; pAUCc = pAUCx + pAUCy + pAUCxy
    x = pfpr + [SEpoint_yStripe[0]] + [pfpr[0]]
    y = ptpr + [SEpoint_xStripe[1]] + [ptpr[0]]
    plotClear(x, y)   # first clear overlap area (using white)
    plotpAUCxy(x, y)  # then plot/fill muted green
    plt.plot(pfpr, ptpr, 'b-', linewidth=2)  # replot the partial curve
    #
    # plot overlap    Error in light red
    if showError:
        x = pfpr + [NWpoint_yStripe[0]] + [pfpr[0]]
        y = ptpr + [NWpoint_xStripe[1]] + [ptpr[0]]
        plotErrorxy(x, y)
    #endif

    return
#enddef
