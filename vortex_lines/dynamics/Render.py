import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from skimage import measure
import pyvista as pv
# pv.set_jupyter_backend('trame')
from tqdm import tqdm, trange
from scipy import interpolate, ndimage
from scipy.io import savemat


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
    pl.show(screenshot=filename,jupyter_backend="none")
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
    pl.show(screenshot=filename,jupyter_backend="none")
    # pl.close(render=False)