    ### Instructions:

    1. **Select Geographic Level**:
       - Choose between ZCTA (ZIP Code Tabulation Areas), STATE, or COUNTRY level mapping
       - For ZCTA level, your CSV must have a 'ZCTA' column
       - For STATE level, your CSV must have a 'geoid' column with state identifiers
       - For COUNTRY level, your CSV must have an 'ISO 3 country code' column matching ISO3 codes

    2. **Upload Data Files**:
       - Upload a CSV file containing your geographic data
       - Choose either a built-in shapefile or upload your own shapefile

    3. **Shapefile Options**:
       - **Built-in**: 
         - For ZCTA: Use the pre-loaded US ZCTA 2024 shapefile
         - For STATE: Use the pre-loaded US State Boundaries shapefile
         - For COUNTRY: Use the pre-loaded World Administrative Boundaries shapefile
       - **Upload your own**: Upload a Shapefile (.shp) and its components (.dbf, .shx, .prj) containing required boundaries

    4. **Edit Data**:
       - You can edit your CSV data directly in the app using the data editor
       - Changes made to the data will be used when generating the map
       - Download the edited CSV data for future use

    5. **Configure Map Settings**:
       - Select the data column you want to visualize
       - Set a legend title and hover label (defaults to column name)
       - Choose a color scheme using the visual color picker
       - Adjust min, max, and step values for the data range
       - Customize the color and opacity for areas with no data

    6. **Generate and Download**:
       - Click "Generate Choropleth Map" to create the visualization
       - Use the "Download Map as HTML" button to save the interactive map

    ### Data Requirements:

    - **For ZCTA Level**:
      - Your CSV file should have a 'ZCTA' column
      - Built-in shapefile contains 'ZCTA5CE20' column for joining
      - If using your own shapefile, it should contain a 'ZCTA5CE20' or 'ZCTA5CE10' column

    - **For STATE Level**:
      - Your CSV file should have a 'geoid' column with state identifiers
      - Built-in shapefile contains 'geoid' and 'name' (state name) columns
      - If using your own shapefile, it should contain 'geoid', 'GEOID', 'STATEFP', or 'stusps' column

    - **For COUNTRY Level**:
      - Your CSV file should have an 'ISO 3 country code' column with ISO3 country codes
      - Built-in shapefile contains 'iso3' column and 'english' (country name) column
      - If using your own shapefile, it should contain 'iso3', 'ISO_A3', or similar ISO country code column
    """# choropleths