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
        precip = f.variables['precip']
        data = precip[:].data
    return data