# %% [markdown]
#
# [![Fixel Algorithms](https://fixelalgorithms.co/images/CCExt.png)](https://fixelalgorithms.gitlab.io)
# 
# # Generate Israel Settlements Distance Matrix
# Using Overpass (OpenStreetMap) / Wikipedia data, generate a distance matrix.
# 
# > Notebook by:
# > - Royi Avital RoyiAvital@fixelalgorithms.com
# 
# ## Revision History
# 
# | Version | Date       | User        |Content / Changes                                                                         |
# |---------|------------|-------------|------------------------------------------------------------------------------------------|
# | 0.1.000 | 20/06/2026 | Royi Avital | First version                                                                            |
# |         |            |             |                                                                                          |

# %% Packages

# Scientific Python
import numpy as np
import scipy as sp
import pandas as pd

# Geo
from geopy.distance import geodesic

# Miscellaneous
from platform import python_version
import random
import requests

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns


# %% Configuration

# %matplotlib inline

# warnings.filterwarnings("ignore")

seedNum = 512
np.random.seed(seedNum)
random.seed(seedNum)

# sns.set_theme() #>! Apply SeaBorn theme

# %% Constants


# %% Local Packages


# %% Auxiliary Functions




# %% Parameters

numSettlements = 1000


# %% Data of Settlements

lOverpassEndpoints = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]

# Query by bounding box (southwest lat/lon, northeast lat/lon)
overpassQuery = """
[out:json][timeout:120];
(
  node["place"~"^(city|town|village)$"](29.4,34.2,33.4,35.9);
  way["place"~"^(city|town|village)$"](29.4,34.2,33.4,35.9);
);
out center;
"""

# Query by country code (ISO 3166-1 alpha-2) and admin level
overpassQuery = """
[out:json][timeout:120];
area["ISO3166-1"="IL"]["admin_level"="2"]->.searchArea;
(
    node["place"~"^(city|town|village)$"](area.searchArea);
    way["place"~"^(city|town|village)$"](area.searchArea);
    relation["place"~"^(city|town|village)$"](area.searchArea);
);
out center;
"""

dHeaders = {
    "User-Agent": "OverpassQueryClient/1.0",
    "Accept": "application/json, text/plain;q=0.9, */*;q=0.8",
    "Content-Type": "text/plain; charset=utf-8",
}

dData     = None
lastError = None

for overpassUrl in lOverpassEndpoints:
    try:
        # Send raw Overpass QL as the request body.
        postResponse = requests.post(overpassUrl, data=overpassQuery, headers=dHeaders, timeout=180)
        postResponse.raise_for_status()
        dData = postResponse.json()
        print(f"Overpass query succeeded via: {overpassUrl}")
        break
    except requests.RequestException as ex:
        statusCode = getattr(ex.response, "status_code", None)
        responseText = ""
        if getattr(ex, "response", None) is not None and ex.response is not None:
            responseText = ex.response.text[:400]
        print(f"Overpass request failed via {overpassUrl} (status={statusCode}): {responseText}")
        lastError = ex

if dData is None:
    raise RuntimeError("All Overpass endpoints failed") from lastError

lSettlements = []

# Parse the geo-data
for element in dData.get('elements', []):
    tags = element.get('tags', {})
    
    # Extract name (prefer English, fallback to default name)
    settlementName = tags.get('name:en', tags.get('name'))
    
    # Extract population safely (convert to int)
    populationStr = tags.get('population', '0').replace(',', '').split(';')[0]
    try:
        population = int(populationStr)
    except ValueError:
        population = 0
        
    # Extract coordinates (ways have a center, nodes have direct lat/lon)
    coordLat = element.get('lat', element.get('center', {}).get('lat'))
    coordLon = element.get('lon', element.get('center', {}).get('lon'))
    
    if settlementName and coordLat and coordLon:
        lSettlements.append({
            'name': settlementName,
            'population': population,
            'lat': coordLat,
            'lon': coordLon
        })

# Normalize names to have only ASCII characters (remove accents, etc.)
for settlement in lSettlements:
    settlement['name'] = settlement['name'].encode('ascii', 'ignore').decode('ascii')

# Convert to DataFrame, drop any duplicates, and sort by largest population
dfSettlement = pd.DataFrame(lSettlements).drop_duplicates(subset = ['name'])
dfSettlement = dfSettlement.sort_values(by = 'population', ascending = False).head(numSettlements)
dfSettlement = dfSettlement.reset_index(drop = True)

# %% Distance Matrix

lNames  = dfSettlement['name'].tolist()
lCoords = list(zip(dfSettlement['lat'], dfSettlement['lon']))

mD = np.zeros((numSettlements, numSettlements))

for ii, coordA in enumerate(lCoords):
    for jj, coordB in enumerate(lCoords):
        if ii == jj:
            continue  # Skip distance to self
        elif jj < ii:
            mD[ii, jj] = mD[jj, ii]  # Symmetric
        else:
            # Matches Google Maps' "Measure Distance" formula
            mD[ii, jj] = geodesic(coordA, coordB).kilometers

# %% Export CSV & Parquet

dfDistance = pd.DataFrame(mD, columns = lNames, index = lNames)

dfSettlement.to_csv(f'Israel{numSettlements}Settlements.csv')
dfDistance.to_csv(f'Israel{numSettlements}SettlementsGeodesicDistances.csv')

dfSettlement.to_parquet(f'Israel{numSettlements}Settlements.parquet')
dfDistance.to_parquet(f'Israel{numSettlements}SettlementsGeodesicDistances.parquet')


# %%



# %%




# %%
