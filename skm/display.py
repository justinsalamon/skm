
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors

# REMOVE PYLAB DEPENDENCY
def _get_cmap(self, N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct
    RGB color.'''
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color


def visualize_clusters(self, X):
    '''
    Visualize the input data and cluster assignments
    '''
    plt.figure(figsize=(8,8))
    # colors_ = ['b','r','g', 'c', 'm', 'y', 'k', 'DeepPink', 'Lime', 'Maroon', 'b','r','g', 'c', 'm', 'y', 'k', 'DeepPink', 'Lime', 'Maroon']
    cmap = self._get_cmap(self.k)
    for n,point in enumerate(X.T):
        # point = self.pca.inverse_transform(point.T)
        if self.assignment is None:
            col = 'b'
        else:
            col = cmap(self.assignment[n])
        plt.plot(point[0],point[1], 'o', color=col)
        plt.axis([-2, 2, -2, 2])
    # plt.axhline(0, color='black')
    # plt.axvline(0, color='black')
    for n,centroid in enumerate(self.D.T):
        # centroid = self.pca.inverse_transform(centroid.T)
        if self.assignment is None:
            col = 'b'
        else:
            col = cmap(n)
        plt.plot(centroid[0],centroid[1],'o', color=col, markersize=10, markeredgewidth=1, markeredgecolor='k')
        # plt.plot(centroid[0],centroid[1],'x', color=cmap(n), markersize=10, markeredgewidth=1, markeredgecolor=cmap(n))
    plt.show()
