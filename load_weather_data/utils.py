# import gaussian filter
from scipy.ndimage import gaussian_filter

def load_nc_data(data_file, variable='air'):
    from netCDF4 import Dataset as dt
    f = dt(data_file)
    if variable=='air':
        air = f.variables['air']
        air_range = air.valid_range
        data = air[:].data
        # convert to degree celsius
        if air.units == 'degK':
            data -= 273
            air_range -= 273
    else:
        precip = f.variables[variable]
        data = precip[:].data

        # apply gaussian filter for every timestep
        data = gaussian_filter(data, sigma=1)

    return data