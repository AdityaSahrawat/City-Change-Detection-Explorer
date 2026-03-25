import geopandas as gpd

gdf = gpd.read_file("data/hisar.geojson")

print(gdf.is_valid)   # should be True
print(gdf.geom_type)  # should be Polygon or MultiPolygon

gdf["geometry"] = gdf.buffer(0)