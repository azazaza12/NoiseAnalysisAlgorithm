import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def visualise_aps_panel(frequencies, depth, aps_data, title_label='', vmin_color=-1, vmax_color=-1):
    fig1, ax1 = plt.subplots()
    if (vmin_color==-1):
        vmax_color = np.percentile(aps_data, 99)
        vmin_color = np.percentile(aps_data, 85)
        colors = ['white', 'blue', 'cyan', 'green', 'yellow', 'orange', 'red', 'maroon']
        cmap = LinearSegmentedColormap.from_list('custom_colormap', colors)
        mesh = ax1.pcolormesh(frequencies, depth,aps_data, cmap=cmap, vmin=vmin_color, vmax=vmax_color, shading='auto')
    else:
        colors = ['white', 'blue', 'cyan', 'green', 'yellow', 'orange', 'red', 'maroon']
        cmap = LinearSegmentedColormap.from_list('custom_colormap', colors)
        mesh = ax1.pcolormesh(frequencies, depth, aps_data, cmap=cmap, vmin=vmin_color, vmax=vmax_color, shading='auto')
    # Добавляем метки осей
    ax1.set_xlabel('Frequency, kHz')
    ax1.set_ylabel('Depth, m')
    plt.title(title_label)
    plt.colorbar(mesh)
    plt.gca().invert_yaxis()



def visualise_new_aps_panel(aps_data, title_label='', vmin_color=-1, vmax_color=-1):
    fig1, ax1 = plt.subplots()
    if (vmin_color==-1):
        vmax_color = np.percentile(aps_data, 99)
        vmin_color = np.percentile(aps_data, 85)
        colors = ['white', 'blue', 'cyan', 'green', 'yellow', 'orange', 'red', 'maroon']
        cmap = LinearSegmentedColormap.from_list('custom_colormap', colors)
        mesh=ax1.pcolormesh(aps_data, cmap=cmap, vmin=vmin_color, vmax=vmax_color, shading='auto')
    else:
        colors = ['white', 'blue', 'cyan', 'green', 'yellow', 'orange', 'red', 'maroon']
        cmap = LinearSegmentedColormap.from_list('custom_colormap', colors)
        mesh = ax1.pcolormesh(aps_data, cmap=cmap, vmin=vmin_color, vmax=vmax_color, shading='auto')
    # Добавляем метки осей
    ax1.set_xlabel('Frequency, kHz')
    ax1.set_ylabel('Depth (m)')
    plt.title(title_label)
    plt.colorbar(mesh)
    plt.gca().invert_yaxis()

