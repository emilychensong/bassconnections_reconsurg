import geopandas as gpd
from shapely.geometry import Polygon

from shapely.geometry import Polygon, MultiPolygon

def calculate_boundary_data(gdf):
    data = []
    
    for index, row in gdf.iterrows():
        # Geometry of the boundary (can be a Polygon or MultiPolygon)
        geom = row['geometry']
        
        # Calculate area (in square kilometers)
        area = geom.area / 10**6  # Convert from m² to km²
        
        # Calculate the centroid (in terms of lat, long)
        centroid = geom.centroid
        centroid_lat = centroid.y
        centroid_long = centroid.x
        
        # Initialize variables for northernmost and easternmost points
        northernmost = None
        easternmost = None
        
        # Check if the geometry is a Polygon or MultiPolygon
        if isinstance(geom, Polygon):
            # For Polygon, directly access the exterior coordinates
            coords = geom.exterior.coords
        elif isinstance(geom, MultiPolygon):
            # For MultiPolygon, access each individual polygon via 'geoms'
            coords = []
            for polygon in geom.geoms:
                coords.extend(polygon.exterior.coords)
        
        # Get the northernmost and easternmost coordinates
        if coords:
            northernmost = max(coords, key=lambda x: x[1])
            easternmost = max(coords, key=lambda x: x[0])
        
        # Prepare data for output
        data.append({
            'Name': row['shapeName'],  # Assuming 'shapeName' is the correct column for the name
            'Area (km²)': area,
            'Centroid Lat': centroid_lat,
            'Centroid Long': centroid_long,
            'Northernmost Coord': northernmost,
            'Easternmost Coord': easternmost
        })
    
    return data




# Function to read and process GeoJSON file
def process_geojson(file_path):
    # Read the GeoJSON into a GeoDataFrame
    gdf = gpd.read_file(file_path)
    
    # Print the column names to help debug
    print(gdf.columns)
    
    # Assuming the file contains the Gaza Strip and its 5 governorates,
    # and that each geometry has a 'name' column for identification
    data = calculate_boundary_data(gdf)
    
    return data


# Example usage
geojson_file = r"C:\Users\emily\OneDrive - Duke University\Bass Connections\Data\geoBoundaries-PSE-ADM2.geojson"  # Specify the path to your GeoJSON file
results = process_geojson(geojson_file)

# Print results
for result in results:
    print(result)
