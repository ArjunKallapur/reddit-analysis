#!/usr/bin/env python3

# May first need:
# In your VM: sudo apt-get install libgeos-dev (brew install on Mac)
# pip3 install https://github.com/matplotlib/basemap/archive/v1.1.0.tar.gz

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import numpy as np

from mpl_toolkits.basemap import Basemap as Basemap
from matplotlib.colors import rgb2hex
from matplotlib.patches import Polygon

"""
IMPORTANT
This is EXAMPLE code.
There are a few things missing:
1) You may need to play with the colors in the US map.
2) This code assumes you are running in Jupyter Notebook or on your own system.
   If you are using the VM, you will instead need to play with writing the images
   to PNG files with decent margins and sizes.
3) The US map only has code for the Positive case. I leave the negative case to you.
4) Alaska and Hawaii got dropped off the map, but it's late, and I want you to have this
   code. So, if you can fix Hawaii and Alask, ExTrA CrEdIt. The source contains info
   about adding them back.
"""


"""
PLOT 1: SENTIMENT OVER TIME (TIME SERIES PLOT)
"""
# Assumes a file called time_data.csv that has columns
# date, Positive, Negative. Use absolute path.

ts = pd.read_csv("time_data.csv")
# Remove erroneous row.
ts = ts[ts['date_created'] != '2018-12-31']

plt.figure(figsize=(12,5))
ts.date = pd.to_datetime(ts['date_created'], format='%Y-%m-%d')
ts.set_index(['date_created'],inplace=True)

ax = ts.plot(title="President Trump Sentiment on /r/politics Over Time",
        color=['green', 'red'],
       ylim=(0, 1.05))
ax.plot()
plt.savefig("part1.png")

"""
PLOT 2: SENTIMENT BY STATE (POSITIVE AND NEGATIVE SEPARATELY)
# This example only shows positive, I will leave negative to you.
"""

# This assumes you have a CSV file called "state_data.csv" with the columns:
# state, Positive, Negative
#
# You should use the FULL PATH to the file, just in case.

state_data = pd.read_csv("state_data.csv")

"""
You also need to download the following files. Put them somewhere convenient:
https://github.com/matplotlib/basemap/blob/master/examples/st99_d00.shp
https://github.com/matplotlib/basemap/blob/master/examples/st99_d00.dbf
https://github.com/matplotlib/basemap/blob/master/examples/st99_d00.shx

IF YOU USE WGET (CONVERT TO CURL IF YOU USE THAT) TO DOWNLOAD THE ABOVE FILES, YOU NEED TO USE 
wget "https://github.com/matplotlib/basemap/blob/master/examples/st99_d00.shp?raw=true"
wget "https://github.com/matplotlib/basemap/blob/master/examples/st99_d00.dbf?raw=true"
wget "https://github.com/matplotlib/basemap/blob/master/examples/st99_d00.shx?raw=true"

The rename the files to get rid of the ?raw=true

"""

ax.get_legend().remove()
# Lambert Conformal map of lower 48 states.
m = Basemap(llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49,
        projection='lcc', lat_1=33, lat_2=45, lon_0=-95)
shp_info = m.readshapefile('st99_d00','states',drawbounds=True)  # No extension specified in path here.
pos_data = dict(zip(state_data.state, state_data.pos))
neg_data = dict(zip(state_data.state, state_data.neg))

diff_cmap = plt.cm.Blues
statenames = []
diff_colors = {}

vmin = -0.7; vmax=-0.4
for shapedict in m.states_info:
	statename = shapedict['NAME']
	if statename not in ['District of Columbia', 'Puerto Rico']:
        	pos = pos_data[statename] - neg_data[statename]
        	diff_colors[statename] = diff_cmap(( pos - vmin )/( vmax - vmin))[:3]
	statenames.append(statename)

ax = plt.gca() # get current axes instance
for nshape, seg in enumerate(m.states):
    # skip Puerto Rico and DC
    if statenames[nshape] not in ['District of Columbia', 'Puerto Rico']:
        color = rgb2hex(diff_colors[statenames[nshape]]) 
        poly = Polygon(seg, facecolor=color, edgecolor=color)
        ax.add_patch(poly)
plt.title('Positive - Negative Trump Sentiment Across the US')
plt.savefig("mycoolmap2.png")

# choose a color for each state based on sentiment.
pos_colors = {}
statenames = []
pos_cmap = plt.cm.Greens # use 'hot' colormap

vmin = 0.33; vmax = 0.45 # set range.
for shapedict in m.states_info:
    statename = shapedict['NAME']
    # skip DC and Puerto Rico.
    if statename not in ['District of Columbia', 'Puerto Rico']:
        pos = pos_data[statename]
        pos_colors[statename] =pos_cmap(( pos - vmin )/( vmax - vmin))[:3]
    statenames.append(statename)
# cycle through state names, color each one.
# POSITIVE MAP
ax = plt.gca() # get current axes instance
for nshape, seg in enumerate(m.states):
    # skip Puerto Rico and DC
    if statenames[nshape] not in ['District of Columbia', 'Puerto Rico']:
        color = rgb2hex(pos_colors[statenames[nshape]]) 
        poly = Polygon(seg, facecolor=color, edgecolor=color)
        ax.add_patch(poly)
plt.title('Positive Trump Sentiment Across the US')
plt.savefig("mycoolmap.png")

# NEGATIVE MAP
neg_colors = {}
neg_cmap = plt.cm.Reds
statenames_neg = []
vmin = 0.85; vmax = 0.95 # set range.
for shapedict in m.states_info:
    statename = shapedict['NAME']
    # skip DC and Puerto Rico.
    if statename not in ['District of Columbia', 'Puerto Rico']:
        pos = neg_data[statename]
        neg_colors[statename] = neg_cmap(( pos - vmin )/( vmax - vmin))[:3]
    statenames.append(statename)

ax = plt.gca() # get current axes instance
for nshape, seg in enumerate(m.states):
    # skip Puerto Rico and DC
    if statenames[nshape] not in ['District of Columbia', 'Puerto Rico']:
        color = rgb2hex(neg_colors[statenames[nshape]]) 
        poly = Polygon(seg, facecolor=color, edgecolor=color)
        ax.add_patch(poly)
plt.title('Negative Trump Sentiment Across the US')
sm = plt.cm.ScalarMappable(cmap=neg_cmap, norm=matplotlib.colors.Normalize(vmin=vmin,vmax=vmax))
sm.set_array(neg_colors)
plt.colorbar(sm)
plt.savefig("mycoolmap1.png")




# SOURCE: https://stackoverflow.com/questions/39742305/how-to-use-basemap-python-to-plot-us-with-50-states
# (this misses Alaska and Hawaii. If you can get them to work, EXTRA CREDIT)

"""
PART 4 SHOULD BE DONE IN SPARK
"""

"""
PLOT 5A: SENTIMENT BY STORY SCORE
"""
# What is the purpose of this? It helps us determine if the story score
# should be a feature in the model. Remember that /r/politics is pretty
# biased.

# Assumes a CSV file called submission_score.csv with the following coluns
# submission_score, Positive, Negative

story = pd.read_csv("submission_score.csv")
plt.figure(figsize=(12,5))
fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(story['submission_score'], story['pos'], s=10, c='b', marker="s", label='Positive')
ax1.scatter(story['submission_score'], story['neg'], s=10, c='r', marker="o", label='Negative')
plt.legend(loc='lower right');

plt.xlabel('President Trump Sentiment by Submission Score')
plt.ylabel("Percent Sentiment")
plt.savefig("plot5a.png")

"""
PLOT 5B: SENTIMENT BY COMMENT SCORE
"""
# What is the purpose of this? It helps us determine if the comment score
# should be a feature in the model. Remember that /r/politics is pretty
# biased.

# Assumes a CSV file called comment_score.csv with the following columns
# comment_score, Positive, Negative

story = pd.read_csv("comment_score.csv")
plt.figure(figsize=(12,5))
fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(story['comments_score'], story['pos'], s=10, c='b', marker="s", label='Positive')
ax1.scatter(story['comments_score'], story['neg'], s=10, c='r', marker="o", label='Negative')
plt.legend(loc='lower right');

plt.xlabel('President Trump Sentiment by Comment Score')
plt.ylabel("Percent Sentiment")
plt.savefig("plot5b.png")
