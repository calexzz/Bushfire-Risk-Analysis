import xarray as xr
import numpy as np
import pandas as pd

# Пути к файлам
PATH_TMP_CORR  = "CTSS_1979-2020.nc"
PATH_PRC_CORR  = "CRSS_1979-2019.nc"
PATH_ERA5_TMP  = "data_stream-moda_stepType-avgua.nc"
PATH_ERA5_PRC  = "data_stream-moda_stepType-avgad.nc"
OUT_TMP        = "temperature_1979-2025.nc"
OUT_PRC        = "precipitation_1979-2025.nc"


# Температура. скорректированные данные 1979-2020
ds_t = xr.open_dataset(PATH_TMP_CORR)

# Назначаем Lat/Lon как координаты осей
ds_t = ds_t.assign_coords(
    Lat=("nLat", ds_t["Lat"].values),
    Lon=("nLon", ds_t["Lon"].values)
).swap_dims({"nLat": "Lat", "nLon": "Lon"})

# Декодируем время из формата 1979.01 в номоиальный 
time_vals = ds_t["Time"].values
years  = np.floor(time_vals).astype(int)
months = np.round((time_vals - years) * 100).astype(int)

# Строим MultiIndex и переводим в (Year, Month, Lat, Lon)
corr_tmp = ds_t["CorTmp"].values  # (504, 61, 241)

ds_tmp_corr = xr.Dataset(
    {"CorTmp": (["Year", "Month", "Lat", "Lon"], corr_tmp.reshape(42, 12, 61, 241))},
    coords={
        "Year":  np.arange(1979, 2021),
        "Month": np.arange(1, 13),
        "Lat":   ds_t["Lat"].values,
        "Lon":   ds_t["Lon"].values,
    }
)
print(ds_tmp_corr)


# Температура в ERA5 2020-2025, K -> C
print("Загружаем ERA5 температуру (2020-2025)...")
ds_era_t = xr.open_dataset(PATH_ERA5_TMP)

# Переводим из Кельвинов в Цельсии
t2m_c = ds_era_t["t2m"] - 273.15

# Переименовываем координаты под единый стиль
t2m_c = t2m_c.rename({"latitude": "Lat", "longitude": "Lon"})

# Извлекаем год и месяц из valid_time
years_era  = t2m_c["valid_time"].dt.year.values
months_era = t2m_c["valid_time"].dt.month.values

# Берём только 2021-2025 (2020 уже есть в скорректированных)
mask = years_era >= 2021
t2m_c_new = t2m_c.values[mask]
years_new  = years_era[mask]
months_new = months_era[mask]

# Собираем в датасет с теми же координатами Lat/Lon
# ERA5 сетка 95-120, скорректированная 60-120 — берём пересечение
lat_vals = t2m_c["Lat"].values
lon_vals = t2m_c["Lon"].values

# Создаём индекс (Year, Month) как MultiIndex
time_idx = pd.MultiIndex.from_arrays([years_new, months_new], names=["Year", "Month"])
unique_years  = sorted(set(years_new))
unique_months = list(range(1, 13))

# Reshape в (Year, Month, Lat, Lon)
n_years = len(unique_years)
arr = t2m_c_new.reshape(n_years, 12, len(lat_vals), len(lon_vals))

ds_tmp_era = xr.Dataset(
    {"CorTmp": (["Year", "Month", "Lat", "Lon"], arr)},
    coords={
        "Year":  unique_years,
        "Month": unique_months,
        "Lat":   lat_vals,
        "Lon":   lon_vals,
    }
)
print(ds_tmp_era)

# Объеденение температур из разных источников
ds_tmp_full = xr.concat([
    ds_tmp_corr.sel(Lon=slice(95, 120)),  # обрезаем корр. до bbox ERA5
    ds_tmp_era
], dim="Year")

ds_tmp_full.attrs["description"] = "Corrected ERA5 monthly mean temperature 1979-2025, Irkutsk region"
print(f"  Итоговый датасет температуры: {ds_tmp_full}")


# Осадки. скорректированные 1979-2019

ds_p = xr.open_dataset(PATH_PRC_CORR)
ds_prc_corr = ds_p.rename({"Y": "Year", "Mon": "Month"})
print(f"  Готово: {ds_prc_corr}")


ds_era_p = xr.open_dataset(PATH_ERA5_PRC)

# м -> мм, и умножаем на кол-во дней в месяце (ERA5 tp — это м/день среднее)
tp = ds_era_p["tp"].rename({"latitude": "Lat", "longitude": "Lon"})

years_era_p  = tp["valid_time"].dt.year.values
months_era_p = tp["valid_time"].dt.month.values

# Количество дней в каждом месяце для перевода м/день мм/месяц
days_in_month = np.array([
    pd.Timestamp(y, m, 1).days_in_month
    for y, m in zip(years_era_p, months_era_p)
])

tp_mm = tp.values * 1000 * days_in_month[:, None, None]

# Берём 2020-2025 (осадки скорректированы только до 2019)
mask_p = years_era_p >= 2020
tp_mm_new    = tp_mm[mask_p]
years_new_p  = years_era_p[mask_p]
months_new_p = months_era_p[mask_p]

lat_p = tp["Lat"].values
lon_p = tp["Lon"].values

unique_years_p = sorted(set(years_new_p))
# берём только до 2025
full_years_p = [y for y in unique_years_p if y <= 2025]
mask_full = np.isin(years_new_p, full_years_p)
tp_mm_new    = tp_mm_new[mask_full]
years_new_p  = years_new_p[mask_full]

n_years_p = len(full_years_p)
arr_p = tp_mm_new.reshape(n_years_p, 12, len(lat_p), len(lon_p))

ds_prc_era = xr.Dataset(
    {"MonTPrecCor": (["Year", "Month", "Lat", "Lon"], arr_p)},
    coords={
        "Year":  full_years_p,
        "Month": list(range(1, 13)),
        "Lat":   lat_p,
        "Lon":   lon_p,
    }
)
print(f"  ERA5 осадки 2020-2025: {ds_prc_era}")


# \Объеденение осадков
ds_prc_full = xr.concat([
    ds_prc_corr.sel(Lon=slice(95, 120)),
    ds_prc_era
], dim="Year")

ds_prc_full.attrs["description"] = "Corrected ERA5 monthly total precipitation 1979-2025, Irkutsk region"
print(f"  Итоговый датасет осадков: {ds_prc_full}")


#Сохранение
print(f"Сохранение {OUT_TMP}...")
ds_tmp_full.to_netcdf(OUT_TMP)

print(f"Сохранение {OUT_PRC}...")
ds_prc_full.to_netcdf(OUT_PRC)

print("Готово!")
print(f"Температура: {OUT_TMP}")
print(f"Осадки: {OUT_PRC}")
