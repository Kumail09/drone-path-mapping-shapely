import geopandas as gpd
import geopandas.io.file
import shapely.speedups
import numpy as np
from shapely.geometry import LineString, shape
import fiona
import os
import pandas as pd

shapely.speedups.enable()


def read_kml_geometry(fp):
    geopandas.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
    kml_df = gpd.read_file(fp, driver='KML')
    geometry = kml_df
    # geometry.plot()
    # plt.show()
    return geometry


def read_shp_geom(fp):
    rooftop_points_df = gpd.read_file(fp)
    rooftop_points_geom = rooftop_points_df



    return rooftop_points_geom


def travelling_salesman(all_points):
    # Calculate the euclidian distance in n-space of the route r traversing cities c, ending at the path start.
    path_distance = lambda r,c: np.sum([np.linalg.norm(c[r[p]]-c[r[p-1]]) for p in range(len(r))])
    # Reverse the order of all elements from element i to element k in array r.
    two_opt_swap = lambda r,i,k: np.concatenate((r[0:i],r[k:-len(r)+i-1:-1],r[k+1:len(r)]))

    def two_opt(cities, improvement_threshold): # 2-opt Algorithm adapted from https://en.wikipedia.org/wiki/2-opt
        route = np.arange(cities.shape[0]) # Make an array of row numbers corresponding to cities.
        improvement_factor = 1 # Initialize the improvement factor.
        best_distance = path_distance(route,cities) # Calculate the distance of the initial path.
        while improvement_factor > improvement_threshold: # If the route is still improving, keep going!
            distance_to_beat = best_distance # Record the distance at the beginning of the loop.
            for swap_first in range(1,len(route)-2): # From each city except the first and last,
                for swap_last in range(swap_first+1,len(route)): # to each of the cities following,
                    new_route = two_opt_swap(route,swap_first,swap_last) # try reversing the order of these cities
                    new_distance = path_distance(new_route,cities) # and check the total distance with this modification.
                    if new_distance < best_distance: # If the path distance is an improvement,
                        route = new_route # make this the accepted best route
                        best_distance = new_distance # and update the distance corresponding to this route.
            improvement_factor = 1 - best_distance/distance_to_beat # Calculate how much the route has improved.
        return route # When the route is no longer improving substantially, stop searching and return the route.

    # Create a matrix of cities, with each row being a location in 2-space (function works in n-dimensions).
    # cities = np.random.RandomState(42).rand(5, 2)
    # Find a good route with 2-opt ("route" gives the order in which to travel to each city by row number.)

    route = two_opt(all_points, 0.001)
    return route

def get_files(mypath, extension):
    """Read files from folder with extension specified"""
    f = []
    for root, dirs, files in os.walk(mypath):
        for file in files:
            if file.endswith(extension):
                 f.append(r'{}/{}'.format(mypath, file))
    return f

def get_points(rooftop_points):
    rooftop_points_geom = rooftop_points['geometry']
    all_single_points = []
    for multipoint in rooftop_points_geom:
        single_points = [(p.x, p.y) for p in multipoint]
        for point in single_points:
            all_single_points.append(point)
    return np.array(all_single_points)

def sort_lst(X, Y):
    sorted_list = []
    a_list = list(range(0, len(X)))
    dict_of_points = dict(zip(a_list, X))
    for val in Y:
        sorted_list.append(dict_of_points[val])
        # print(dict_of_points[val])
    return sorted_list


def main():
    all_kml_area_files = get_files("Area_Kml", ".kml")
    all_point_files = get_files("Points_Shp", ".shp")
    i = 0
    for k_file in all_kml_area_files:
        for p_file in all_point_files:
            try:
                area_poly = read_kml_geometry(k_file)
                parcel_ids_pts = []
                geometry_pts = []

                with fiona.open(p_file) as roof_source:
                    for curr_point in roof_source:
                        shp_geom = shape(curr_point['geometry'])
                        parcel_id = curr_point['properties']['PARCEL_ID']
                        pip_mask = shp_geom.within(area_poly.loc[0, 'geometry'])
                        if pip_mask is True:
                            parcel_ids_pts.append(parcel_id)
                            geometry_pts.append(shp_geom)
                try:
                    pid_df = pd.DataFrame(parcel_ids_pts, columns=["PARCEL_ID"])
                    pid_df['geometry'] = geometry_pts
                    pid_df.to_csv(f'Output_Kml/Parcel_ID_Coord_{i}.csv',index=False)
                except Exception as E:
                    print(E)

                pip_data = gpd.GeoDataFrame(pid_df, geometry='geometry', crs={'init': 'epsg:4326'})
                "Dropped Duplicates"
                pip_data.drop_duplicates(inplace=True)
                kml_points = get_points(pip_data)
                index_of_points_arr = travelling_salesman(kml_points)
                pts_list = kml_points.tolist()
                ordered_pts = sort_lst(pts_list, index_of_points_arr)
                lstr = LineString(ordered_pts)
                d = {'geometry': [lstr]}
                df = gpd.GeoDataFrame(d, crs="EPSG:4326")  # (change epsg)
                df.to_file('Output_Kml/Path_Extracted_{}.kml'.format(i), driver='KML')
                i += 1
            except Exception as E:
                print(E)
if __name__ == '__main__':
    main()