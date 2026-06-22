import xarray as xr
import numpy as np
import pandas as pd

ds_temp = xr.open_dataset("data/CTSS_1979-2020.nc")

# Lon/Lat были переменными - берём их значения как координаты nLon/nLat для выборки по градусам
ds_temp_assigned_coords = ds_temp.assign_coords(
    nLon=(ds_temp.Lon.values),
    nLat=(ds_temp.Lat.values)
)

# Lon/Lat остались дублирующими data variables после assign_coords - убираем
ds_temp_coords = ds_temp_assigned_coords.drop_vars({'Lon', 'Lat'})

# Обрезка по границам Иркутской области
ds_temp_region = ds_temp_coords.sel(
    nLat=slice(65, 51),
    nLon=slice(95, 120)
)

time_values = ds_temp_region.Time.values

# Time типа float (например 1979.01) Разбиваем на год и месяц
years_floor = np.floor(time_values)
months_floor = np.round((time_values - years_floor) * 100)
years = years_floor.astype(int)
months = months_floor.astype(int)

# Создаем DataFrame с колонками year/month/day и собираем дату (datetime)
df = pd.DataFrame({'year': years, 'month': months, 'day': 1})
dates = pd.to_datetime(df)

# Заменяем координату Time формата float на формат datetime
ds_temp_final = ds_temp_region.assign_coords(Time = dates)
print(ds_temp_final)