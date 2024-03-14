# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams["font.family"] = "Arial"
# from skimage import measure
import pyvista as pv
pv.set_jupyter_backend('trame')
from tqdm import tqdm, trange
from scipy import interpolate, ndimage
from scipy.io import savemat

# %% [markdown]
# ## Define function

# %% [markdown]
# ### Generate randomwave

# %%
def sample_k(k_mean,k_cov):
    return np.random.multivariate_normal(k_mean,k_cov)

#### Superpositioning the random wave ####
def rotation_matrix(axis, phi):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(phi / 2.0)
    b, c, d = -axis * np.sin(phi / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def sample_wave_MO_complex(r_grid, k_mean, k_cov, n_wave = 100, kappa=1e8):
    rho = np.zeros_like(r_grid[0]).astype('complex64')
    r_grid = [r.astype(np.float32) for r in r_grid]
    for i in range(n_wave):
        k_sample = sample_k(k_mean,k_cov)

        # misorientation
        """
        https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution
        https://doi.org/10.1080/03610919408813161
        """
        sigma = 1e-6
        xi = np.random.rand()
        theta = np.random.rand()*2*np.pi
        W = 1+1/kappa*(np.log(xi*(1-(xi-1)/xi*np.exp(-2*kappa))))
        phi = np.arccos(W)
        axis = [np.cos(theta),np.sin(theta),0]
        R = rotation_matrix(axis,phi)
        k_sample_rot = R@k_sample

        k_dot_r = np.sum([r_grid[x]*k_sample_rot[x] for x in range(3)],axis=0)
        phi_r = np.random.rand()*2*np.pi # random phase
        rho_i = np.exp(1j*(k_dot_r + phi_r)) # cos(k_n.r + phi_n)
        rho += rho_i.astype('complex64')

    rho = np.sqrt(2/n_wave)*rho
    
    return rho

def scale_rho(rho, xyz, scale):
    rho = ndimage.zoom(rho, scale, order=1)
    # r_grid = np.array([ndimage.zoom(r, scale, order=1) for r in r_grid]) 
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    x_zoom = ndimage.zoom(x, scale, order=1)
    y_zoom = ndimage.zoom(y, scale, order=1)
    z_zoom = ndimage.zoom(z, scale, order=1)
    r_grid = np.meshgrid(x_zoom,y_zoom,z_zoom)
    return rho, r_grid

# %% [markdown]
# ### Identify vortex lines

# %%
## identify vortex line (slice of plane)
def vortex_slice(phase_slice):
    '''
    Input
    phase_slice: 2D array of phase field (floats, -pi to pi).

    Output
    vortex_array: 2D array of vortex locations (binary)
    ----------------------------------------------------------------
     --- --- ---
    | 1 | 8 | 7 |
    --- --- --- 
    | 2 | p | 6 |
    --- --- ---
    | 3 | 4 | 5 |
    --- --- ---
    A pixel is identified as a vortex if the phase difference along 
    the encircling path exceeds pi for an odd number of times.
    '''
    # list_cells = np.array([[-1,1], [-1,0], [-1,-1], [0,-1], [1,-1], [1,0], [1,1], [0,1]])

    vortex_array = np.zeros_like(phase_slice)
    # evaluate the phase difference
    pd_u = phase_slice[:,1:]-phase_slice[:,0:-1]
    pd_d = phase_slice[:,0:-1]-phase_slice[:,1:]
    pd_l = phase_slice[0:-1,:]-phase_slice[1:,:]
    pd_r = phase_slice[1:,:]-phase_slice[0:-1,:]

    ## vectorized
    ib_i = np.arange(phase_slice.shape[0]-2)
    ib_j = np.arange(phase_slice.shape[0]-2)
    phase_diff = np.array([pd_d[np.ix_(ib_i+1-1,ib_j+1)],pd_d[np.ix_(ib_i+1-1,ib_j)],
                           pd_r[np.ix_(ib_i,ib_j+1-1)],pd_r[np.ix_(ib_i+1,ib_j+1-1)],
                           pd_u[np.ix_(ib_i+1+1,ib_j)],pd_u[np.ix_(ib_i+1+1,ib_j+1)], 
                           pd_l[np.ix_(ib_i+1,ib_j+1+1)],pd_l[np.ix_(ib_i,ib_j+1+1)]])
    index_defect = np.sum(np.abs(phase_diff)>np.pi,axis=0)%2>0
    index_defect = np.pad(index_defect, ((1,1), (1,1)))
    vortex_array[index_defect] = 1
    
    ## for loop
    # for i in range(phase_slice.shape[0]-2):
    #     for j in range(phase_slice.shape[0]-2):
    #         # center_coord = np.array([i+1,j+1])
    #         # cells_coord = center_coord + list_cells
    #         # phase_coord = [phase_slice[k] for k in zip(*cells_coord.T)]
    #         # phase_diff = np.array([(phase_coord[(i+1)%8]-phase_coord[i%8]) for i in range(8)])
    #         phase_diff = np.array([pd_d[i+1-1,j+1],pd_d[i+1-1,j],
    #                                pd_r[i,j+1-1],pd_r[i+1,j+1-1],
    #                                pd_u[i+1+1,j],pd_u[i+1+1,j+1], 
    #                                pd_l[i+1,j+1+1],pd_l[i,j+1+1]])

    #         if np.sum(np.abs(phase_diff)>np.pi)%2>0:
    #             vortex_array[i+1,j+1] = 1

    return vortex_array

def vortex_phase(rho_phase, print_size=False):
    """
    Scan over the simulation cell and identify defects.

    Input:
        rho_phase: The 3D density phase array.

    Output:
        vortex_volume: The 3D array identifying defect position.
    """

    ## scanning over the simulation cell
    n_slices = rho_phase.shape[1]
    if print_size:
        print(f'n_slices = {n_slices}')
    vortex_volume = np.ones_like(rho_phase)
    for ax in range(3):
        axis_slice = ax
        vortex_array_list = []
        for i in range(n_slices):
            rho_phase_slices = rho_phase.take(indices = i, axis=axis_slice)
            vortex_array = vortex_slice(rho_phase_slices)
            vortex_array_list.append(vortex_array)

        vortex_volume_ax = np.array(vortex_array_list)
        vortex_volume_ax = np.moveaxis(vortex_volume_ax,0,axis_slice)
        vortex_volume = vortex_volume*(1-vortex_volume_ax)

    vortex_volume = 1-vortex_volume

    return vortex_volume


# %% [markdown]
# ###  Trace line defects

# %%
### Return lists of adjacent elements
def get_adjacency(array):
    """
    Return lists of adjacent elements

    Args: 
    - array: Binary array where 1 represent the voxels containing the vertex lines
    Returns:
    - positions_tuple_list: List of positions of the voxels (expressed in tuple)
    - positions_list: List of positions of the voxels (expressed in raveled index)
    - adjacent_list: List of inices of adjacent voxels (expressed in raveled index)
    - adjacent_id_list: List of inices of adjacent voxels, 
                        which correspond to the index of positions_tuple_list and positions_list
    """
    # first part
    positions_tuple_list = []
    positions_list = []
    adjacent_list = []
    adjacent_id_list = []

    # Pad the array to simplify boundary checks
    padded_array = np.pad(array, pad_width=1, mode='constant', constant_values=0)

    # Define the kernel for directly connected neighboring elements
    kernel = np.zeros((3, 3, 3), dtype=int)
    ones_list = [[0,1,1],[2,1,1],[1,0,1],[1,2,1],[1,1,0],[1,1,2]]
    for index in ones_list:
        i,j,k = index
        kernel[i,j,k] = 1

    # # include "edge" elements
    # kernel = np.ones((3, 3, 3), dtype=int)
    # zeros_list = [[0,0,0],[2,0,0],[0,2,0],[0,0,2],[0,2,2],[2,0,2],[2,2,0],[2,2,2],[1,1,1]]
    # for index in zeros_list:
    #     i,j,k = index
    #     kernel[i,j,k] = 0
    # kernel[1, 1, 1] = 0  # Exclude the center element

    # Find positions labeled as 1
    labeled_positions = np.argwhere(array == 1)

    for index, pos in enumerate(labeled_positions):
        i, j, k = pos
        current_pos = tuple(pos)
        current_pos_ravel = np.ravel_multi_index(pos,array.shape)
        adjacent_positions = []

        # Get neighboring positions
        neighbors = padded_array[i:i+3, j:j+3, k:k+3] * kernel

        # Find adjacent positions labeled as 1
        adjacent_indices = np.argwhere(neighbors == 1)
        for index in adjacent_indices:
            x, y, z = index - 1  # Adjust indices to get relative positions
            # adjacent_positions.append((i + x, j + y, k + z))
            adjacent_positions.append(np.ravel_multi_index([i + x, j + y, k + z], array.shape))
        
        positions_tuple_list.append(current_pos)
        positions_list.append(current_pos_ravel)
        adjacent_list.append(adjacent_positions)

    # second part: map adjacent_list to indices of positions_list
    # accelerate position mapping by constructing dictionary
    ravel_index_map = {ravel_index: idx for idx, ravel_index in enumerate(positions_list)}

    for index in range(len(positions_tuple_list)):
        adjacent_id = [ravel_index_map[adjacent] for adjacent in adjacent_list[index]]
        adjacent_id_list.append(adjacent_id)

    return positions_tuple_list, positions_list, adjacent_list, adjacent_id_list

### Return connected components according to the adjacency list
def get_connected_part(adjacent_id_list):
    """
    Label connected parts (one-based) by performing a depth-first search (DFS) using a stack.
    
    Args:
    - adjacent_id_list (list): List containing adjacent IDs for each particle
    
    Returns:
    - parts_arr (numpy.ndarray): Array containing the labels of connected parts
    """

    # Initialize check set, output list, and counter
    checked = set()
    parts_arr = np.zeros(len(adjacent_id_list), dtype=int)  # Array to store part labels
    max_part = 0  # Counter for labeling connected parts

    # Iterate through each index and its adjacent IDs
    for index, adjacent_id in enumerate(adjacent_id_list):
        if index not in checked:
            max_part += 1  # Increment the part label counter
            parts_arr[index] = max_part  # Assign the current index the new part label
            checked.add(index)  # Mark the current index as checked

        stack = [index]  # Create a stack with the current index
        while stack:
            current = stack.pop()  # Get the top element from the stack
            # Iterate through each adjacent ID of the current index
            for neighbor in adjacent_id_list[current]:
                if neighbor not in checked:
                    parts_arr[neighbor] = max_part  # Assign the part label to the neighbor
                    checked.add(neighbor)  # Mark the neighbor as checked
                    stack.append(neighbor)  # Add the neighbor to the stack for further exploration

    return parts_arr

### Generate sample points representing the vortex cores
def get_core(positions_tuple_list, adjacent_id_list, cluster_size=3):
    """
    Generate sample points representing the vortex cores based on positions and adjacency information.
    
    Args:
    - positions_tuple_list (list): List containing tuples of position coordinates
    - adjacent_id_list (list): List containing adjacent IDs for each particle
    - cluster_size (int, optional): Size of the cluster to determine sample points. Defaults to 5.

    Returns:
    - sample (list): List of lists containing sample points for each identified vortex core
    - sample_adj_list (list): List of adjacency lists for the sample points
    - sample_adj_dict (list): List of adjacency dictionaries for the sample points
    """
    # Convert positions to an array
    poslist_arr = np.array(positions_tuple_list)
    
    # Determine connected parts
    parts_arr = get_connected_part(adjacent_id_list)
    max_part = np.max(parts_arr)

    # Initialize lists to store the sample points and their adjacencies
    sample = []
    sample_adj_list = []
    sample_adj_dict = []

    # Loop through each identified part
    for i in range(max_part):
        current_id = np.where(parts_arr == i + 1)[0]
        current_adjacent_id = [adjacent_id_list[i] for i in current_id]

        # Extract points within the current part
        poslist_arr_current = poslist_arr[current_id]

        # Initialize lists to store the current part's sample points and their connections
        current_sample = []
        current_sample_parent = []

        checked = set()
        for j, adjacent_id in enumerate(current_adjacent_id):
            index = current_id[j]

            if index in checked:
                continue

            stack = [index]
            stack_sample_id = [1]  # Index of the current sample

            while stack:
                current = stack.pop()
                prev_sample_id = stack_sample_id.pop()

                if current in checked:
                    continue

                # Generate a list of neighbors within the cluster size around the current vertex
                d_sq = np.sum((poslist_arr_current - poslist_arr[current, :]) ** 2, axis=1)
                cluster_id = np.where(d_sq < (cluster_size ** 2))[0]

                # Generate the sample point by averaging the points within the cluster
                cluster = poslist_arr_current[cluster_id]
                current_sample.append(np.mean(cluster, axis=0))

                # Record the connections between sample points
                current_sample_parent.append(prev_sample_id)

                # Find adjacent particles not within the current cluster
                adjacent_current = []
                for neighbor in current_id[cluster_id]:
                    checked.add(neighbor)
                    adjacent_current.extend(adjacent_id_list[neighbor])
                adjacent_current = list(set(adjacent_current) - set(current_id[cluster_id]))

                # Continue searching for the neighbors of the current cluster
                for neighbor in adjacent_current:
                    if neighbor not in checked:
                        stack.append(neighbor)
                        stack_sample_id.append(len(current_sample) - 1)  # Store the index of the current sample point

        # Build adjacency dictionary if more than one sample point exists
        if len(current_sample_parent) > 1:
            adj_dict = {idx: set([connection]) for idx, connection in enumerate(current_sample_parent)}
            for idx, connection in enumerate(current_sample_parent):
                adj_dict[connection].add(idx)
        else:
            adj_dict = {}

        # Build adjacency list if more than one sample point exists
        adj_list = []
        if len(current_sample_parent) > 1:
            adj_list = [list(adj_dict[idx]) for idx in range(len(current_sample))]

        # Append the sample points and their adjacencies to the respective lists
        sample.append(current_sample)
        sample_adj_list.append(adj_list)
        sample_adj_dict.append(adj_dict)

    return sample, sample_adj_list, sample_adj_dict

### Generate ordered list of sample points
def get_ordered_points(sample, sample_adj_list):
    """
    Generate an ordered list of sample points based on their adjacency information.

    Args:
    - sample (list): List of lists containing sample points for each identified vortex core
    - sample_adj_list (list): List of adjacency lists for the sample points

    Returns:
    - sample_ordered (list): Ordered list of sample points following the connections in the adjacency lists.
                             Each entry in the list represents a continuous branch of connected sample points.
    """
    
    sample_ordered = []  # Initialize the list to store ordered sample points
    for i, adj_list in enumerate(sample_adj_list):
        visited_nodes = set()  # Set to keep track of visited nodes within the sample
        sample_reorder = []  # Temporary list to store each branch of sample points

        # If the sample has more than one node
        if len(sample[i]) > 1:

            # Recursive function to traverse the adjacency list from a given node
            def traverse_adj_list(node, adj_list, visited, sample_reorder):
                visited.add(node)
                sample_reorder.append(sample[i][node])  # Append the current sample point to the branch
                for neighbor in adj_list[node]:
                    if neighbor in intersects:  # Stop traversal if an intersect node is reached
                        continue
                    if neighbor not in visited:
                        traverse_adj_list(neighbor, adj_list, visited, sample_reorder)

            # Calculate the connectivity of each node
            connectivity = np.array([len(adj) for adj in adj_list])

            # Identify nodes with connectivity greater than 2 (possible intersections)
            intersects = np.where(connectivity > 2)[0]

            # If there are intersect nodes, split the sample into branches
            if len(intersects) > 0:
                for intersect in intersects:
                    visited_nodes.add(intersect)
                    # Traverse the adjacency list from the intersect node for each branch
                    for neighbor in adj_list[intersect]:
                        if (neighbor not in visited_nodes) and (neighbor not in intersects):
                            branch = []  # New empty list for each branch
                            traverse_adj_list(neighbor, adj_list, visited_nodes, sample_reorder=branch)

                            if len(branch) > 1:
                                sample_ordered.append(branch)  # Append the branch to the ordered sample list
            
            # If there are no intersect nodes, find the ends of the branches
            else:
                ends = np.where(connectivity != 2)[0]
                if len(ends) == 0:
                    end = 0
                else:
                    end = ends[0]
                traverse_adj_list(end, sample_adj_list[i], visited_nodes, sample_reorder)

        # If there's only one node, append it directly
        else:
            sample_reorder.append(sample[i][0])

        sample_ordered.append(sample_reorder)  # Append the final ordered sample branch to the list

    return sample_ordered

### Line tangent
def get_tangent(points_ordered):
    if len(points_ordered)<3:
        return[]
    else:
        grad = np.gradient(points_ordered)[0]
        norm = np.linalg.norm(grad,axis=1)

        tangent = [g/n for g,n in zip(grad,norm)]

        return(tangent)

# %% [markdown]
# ### Visualization

# %%
def visualize_lamellar(r_grid,rho_real,vortex_volume,
                       lamellar=True,isometric=False,alpha=0,
                       color='#303030',
                       filename = './test_lamellar.png'):
    # pyvista
    # https://stackoverflow.com/questions/6030098
    grid = pv.StructuredGrid(r_grid[1], r_grid[0], r_grid[2])
    grid["vol"] = rho_real.flatten('F')
    mesh = grid.contour([alpha])

    grid["vol"] = vortex_volume.flatten('F')
    mesh2 = grid.contour([0.5])

    # Visualization
    pv.set_plot_theme('document')
    pl = pv.Plotter(window_size=[600, 600])
    pl.enable_anti_aliasing('msaa')

    if lamellar:
        backface_params = dict(color=color,
                            ambient=0.2, diffuse=0.8, specular=0.1, specular_power=10,
                            opacity=1
                            )
        pl.add_mesh(mesh, show_scalar_bar=False, color='#A0A0A0',  
                    ambient=0.2, diffuse=0.8, specular=0.1, specular_power=10,
                    backface_params=backface_params, 
                    smooth_shading=True, 
                    opacity=1
                    )

    pl.add_mesh(mesh, opacity=0, show_scalar_bar=False)

    # backface_params_defect = dict(color='#FF0000',
    #                     ambient=0.2, diffuse=0.8, specular=0.1, specular_power=10,
    #                     opacity=0.5
    #                     )
    # pl.add_mesh(mesh2, show_scalar_bar=False, color='#FF0000',  
    #             ambient=0.2, diffuse=0.8, specular=0.1, specular_power=10,
    #             backface_params=backface_params_defect, 
    #             smooth_shading=True, 
    #             opacity=0.5
    #             )

    if isometric:
        # camera setting
        pl.enable_parallel_projection()
        pl.camera_position = 'yz'
        pl.camera.reset_clipping_range()
    else:
        # camera setting
        pl.camera_position = 'yz'
        pl.camera.azimuth = -60.0
        pl.camera.elevation = 24.0
        pl.camera.reset_clipping_range()

    # light setting
    light = pv.Light()
    light.set_direction_angle(21, -55.0)
    light.attenuation_values = (0,0,2)
    pl.add_light(light)

    pl.add_bounding_box()
    pl.show(screenshot=filename)
    # pl.close(render=False)

def visualize_defect(r_grid,rho_real,vortex_volume,sample_ordered,
                       lamellar=False,isometric=False,alpha=0,
                       lw=1,
                       color='#303030',
                       filename = './test_defect.png'):
    # pyvista
    # https://stackoverflow.com/questions/6030098
    grid = pv.StructuredGrid(r_grid[1], r_grid[0], r_grid[2])
    grid["vol"] = rho_real.flatten('F')
    mesh = grid.contour([0])

    # Visualization
    pv.set_plot_theme('document')
    pl = pv.Plotter(window_size=[600, 600])
    pl.enable_anti_aliasing('msaa')

    if lamellar:
        backface_params = dict(color=color,
                            ambient=0.2, diffuse=0.8, specular=0.1, specular_power=10,
                            opacity=1
                            )
        pl.add_mesh(mesh, show_scalar_bar=False, color='#A0A0A0',  
                    ambient=0.2, diffuse=0.8, specular=0.1, specular_power=10,
                    backface_params=backface_params, 
                    smooth_shading=True, 
                    opacity=1
                    )

    pl.add_mesh(mesh, opacity=0, show_scalar_bar=False)

    if 1:
        gridsize = r_grid[0].shape[0]
        def polyline_from_points(points):
            poly = pv.PolyData()
            poly.points = points
            the_cell = np.arange(0, len(points), dtype=np.int_)
            the_cell = np.insert(the_cell, 0, len(points))
            poly.lines = the_cell
            return poly

        for i, points in enumerate(sample_ordered):
            if len(sample_ordered[i])>=3:
                polyline = polyline_from_points(np.array(points)/gridsize*2-1)
                tangent_i = np.array(get_tangent(points))
                # theta = np.arccos(tangent_i)
                polyline["scalars"] = tangent_i[:,2]**2
                tube = polyline.tube(radius=0.01*lw)
                pl.add_mesh(tube, show_scalar_bar=False, 
                            clim=[0,1], cmap='viridis')

    if isometric:
        # camera setting
        pl.enable_parallel_projection()
        pl.camera_position = 'yz'
        pl.camera.reset_clipping_range()
    else:
        # camera setting
        pl.camera_position = 'yz'
        pl.camera.azimuth = -60.0
        pl.camera.elevation = 24.0
        pl.camera.reset_clipping_range()

    # light setting
    light = pv.Light()
    light.set_direction_angle(21, -55.0)
    light.attenuation_values = (0,0,2)
    pl.add_light(light)

    pl.add_bounding_box()
    pl.show(screenshot=filename)
    # pl.close(render=False)

def defect_density(vortex_volume, r_grid):
    d_voxel = r_grid[0][0,1,0]-r_grid[0][0,0,0]
    d_cell = [(r_grid[0][0,-1,0]-r_grid[0][0,0,0]),
              (r_grid[1][-1,0,0]-r_grid[1][0,0,0]),
              (r_grid[2][0,0,-1]-r_grid[2][0,0,0])] 
    sum_voxel = np.sum(vortex_volume) # total volume of defective voxels (pixel^3)
    cross_voxel = 4 # crossection of defective voxels (pixel^2)
    length = sum_voxel/cross_voxel # length of defect line (pixel)

    cell_volume = d_cell[0]*d_cell[1]*d_cell[2]
    line_density = length*d_voxel/cell_volume

    return(line_density)

def scale_rho(rho, r_grid, scale):
    rho = ndimage.zoom(rho, scale, order=1)
    r_grid = np.array([ndimage.zoom(r, scale, order=1) for r in r_grid]) 
    return rho, r_grid

## Generate vertex lines
def gen_vertex_lines(r_grid, sigma_k, kappa, alpha, d, box_size, x_scale=1, n_grid_scale=128):
    #### Wave vector distribution ####
    scale = box_size/d/2 # how many layers in the box
    k_mean_z = np.array([0,0,scale])*2*np.pi # lamellar perpendicular to z axis 
    # k_mean_x = np.array([0,0,0])*np.pi # lamellar perpendicular to z axis 
    k_var  = (np.array([0,0,sigma_k*scale])*2*np.pi)**2
    k_cov  = np.diagflat(k_var)

    ### generate randomwave
    rho = sample_wave_MO_complex(r_grid,k_mean_z,k_cov,n_wave = 100, kappa=kappa)

    scale_zoom = n_grid_scale/rho.shape[0]

    rho, r_grid = scale_rho(rho, r_grid, scale_zoom)

    rho_real = rho.real
    rho_imag = rho.imag
    rho_phase = np.angle(rho)

    vortex_volume = vortex_phase(rho_phase)

    ### Trace defect lines
    array = vortex_volume
    positions_tuple_list, positions_list, adjacent_list, adjacent_id_list = get_adjacency(array)
    poslist_arr = np.array(positions_tuple_list)

    sample, sample_adj_list, sample_adj_dict = get_core(positions_tuple_list, adjacent_id_list, 
                                                    cluster_size=3)
    sample_ordered = get_ordered_points(sample, sample_adj_list)

    sample_ordered_sm = []
    tangent_list = []
    for line in sample_ordered:
        if len(line) == 0:
            continue
        if len(line)>5:
            x = np.arange(len(line))
            x_fine = np.arange(len(line)*x_scale)/x_scale
            arr_line = np.array(line)
            ll = len(line)
            spl_list = [splrep(x,arr_line[:,i],w=np.ones(len(line)),s=ll+np.sqrt(ll*2)) for i in range(3)]
            line_sm = np.array([splev(x_fine,spl) for spl in spl_list]).T
            sample_ordered_sm.append(line_sm)

            # get line tangent
            tangent = np.array(get_tangent(line_sm))
            tangent_list.append(tangent)

        # else:
        #     sample_ordered_sm.append(line)
    print('Finished generating sample lines')
    return r_grid, rho_real, vortex_volume, sample_ordered_sm, tangent_list