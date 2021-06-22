import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import sys
import copy


# Generate histogram of a batch of utxos
def get_histogram(height, amount, xedges, yedges):
    # place batch into histogram
    x = np.array(height)
    y = np.array(amount) * 1e-8

    # take log of amounts
    y[np.where(y == 0)] = 1e-9
    y = np.log10(y)

    tmp_hist, xedges, yedges = np.histogram2d(x, y, (xedges, yedges))
    return tmp_hist, xedges, yedges


# Script start from here
if __name__ == "__main__":
    block_height = int(sys.argv[1])
    coin_count = int(sys.argv[2])

    # set histogram resolution
    max_yedge = 5
    min_yedge = -8
    yres = 800
    xres = yres * 2
    maxx = int(block_height) + 3  # adding three for padding here
    minx = 1
    yedges = np.linspace(-8, max_yedge, yres - 3)
    yedges = np.concatenate(([-10, -9, -8.5], yedges, [10]))
    xedges = np.linspace(minx, maxx, xres + 1)
    hist = np.zeros(shape=(xres, yres), dtype=float)

    # size of batch of utxos to call in
    batch_size = coin_count // 100

    coins_processed = 0
    height = []
    amount = []
    with open('utxodump.csv', 'r') as csvfile:
        lines = csvfile.readlines()

        # for row in csvreader:
        for line in lines:
            if coins_processed == 0:
                # Skipping first row of field names
                coins_processed += 1
                continue

            row = line.split(',')
            height.append(int(row[0]))
            amount.append(int(row[1]))

            if batch_size == coins_processed:
                # get histogram
                tmp_hist, xedges, yedges = get_histogram(height, amount, xedges, yedges)
                hist += tmp_hist

                # reset
                coins_processed = 0
                height = []
                amount = []

            coins_processed += 1

    # %% format the hist matrix for rendering
    phist = hist

    # non-zero, take logs and rotate hist matrix
    phist[np.where(phist == 0)] = .06874501
    phist = np.log10(phist)
    phist = np.rot90(phist)
    phist = np.flipud(phist)

    # get max values
    hmax = phist.max()
    hmin = phist.min()

    # insert nan from zero value bins
    phist[np.where(phist == hmin)] = np.nan

    # Adjust the one sat and zero sat position so easier to see in plot
    phist[10, :] = phist[1, :]
    phist[6, :] = phist[0, :]

    # get figure handles
    plt.clf()
    fig = plt.figure(figsize=(8, 6), facecolor='black')
    ax = fig.add_axes([.11, .37, .8, .55])

    # Color maps for pcolor
    my_cmap = copy.copy(cm.gnuplot2)
    my_cmap.set_bad(color='black')

    # Render scatter
    im = ax.pcolormesh(phist, vmin=-1, vmax=np.floor(hmax * .6), cmap=my_cmap, label='UTXO Histogram')

    # yaxis format
    plt.yticks(np.linspace(0, yres, num=14))
    labels = ["100k", "10k", "1k", '100', '10',
              "1", ".1", '.01', '.001', '10k sat',
              "1k sat", "100 sat", '10 sat', '0 sat', ]
    labels.reverse()
    ax.set_yticklabels(labels, fontsize=8)
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(labelright=True)

    # xaxis format
    ticks_year = ['2009', '2010', '2011', '2012',
                  '2013', '2014', '2015', '2016',
                  '2017', '2018', '2019', '2020', '2021']
    ticks_height = [1, 32500, 100400, 160400,
                    214500, 278200, 336700, 391300,
                    446200, 502100, 556500, 610800, 664100]
    ticks_x = []
    label_x = []
    for n in range(len(ticks_height)):
        th = ticks_height[n]
        ticks_x.append(np.argmin(np.abs(np.array(xedges) - th)))
        label_x.append(ticks_year[n] + "\n" + str(th))

    plt.xticks(ticks_x)
    ax.set_xticklabels(label_x, rotation=0, fontsize=6)

    # Title and labels
    tick_color = "white"
    fig_title = "  The Bitcoin Blockchain (from file " + str(block_height) + ")"
    tobj = plt.title(fig_title, fontsize=12, loc='left')
    plt.setp(tobj, color=tick_color)
    ax.set_ylabel('Amount (BTC)', fontsize=8)
    ax.spines['bottom'].set_color(tick_color)
    ax.spines['top'].set_color(tick_color)
    ax.tick_params(axis='x', colors=tick_color)
    ax.xaxis.label.set_color(tick_color)
    ax.spines['right'].set_color(tick_color)
    ax.spines['left'].set_color(tick_color)
    ax.tick_params(axis='y', colors=tick_color)
    ax.yaxis.label.set_color(tick_color)
    ax.set_xlabel("Output time (year, block height)", fontsize=8)

    #  Color bar
    cbaxes = fig.add_axes([0.75, .925, 0.2, 0.015])
    cb = plt.colorbar(im, orientation="horizontal", cax=cbaxes)
    cbaxes.set_xlim(-0.01, np.floor(hmax * .8) + .1)
    cbaxes.xaxis.set_ticks_position('top')
    cbticks = np.arange(int(np.floor(hmax * .6)) + 1)
    cb.set_ticks(cbticks)
    clabels = ['1', '10', '100', '1k', '10k', '100k', '1M']
    cbaxes.set_xticklabels(clabels[0:len(cbticks)], fontsize=6)
    cbaxes.set_ylabel("Number of \nunspent outputs", rotation=0, fontsize=6)
    cbaxes.yaxis.set_label_coords(-.25, 0)
    cbaxes.tick_params('both', length=0, width=0, which='major')
    cb.outline.set_visible(False)
    cbaxes.spines['bottom'].set_color(tick_color)
    cbaxes.tick_params(axis='x', colors=tick_color)
    cbaxes.yaxis.label.set_color(tick_color)

    # save the image
    fig_name = "./utxo_heatmap_" + str(block_height) + ".png"
    plt.savefig(fig_name, dpi=1200, bbox_inches='tight', facecolor=fig.get_facecolor(), transparent=True)
