import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MousePosition
import streamlit as st
import tempfile
import os
from streamlit_folium import folium_static
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
import base64
from io import BytesIO

# Initialize session state for persistent data
if 'csv_data' not in st.session_state:
    st.session_state.csv_data = None
if 'selected_color' not in st.session_state:
    st.session_state.selected_color = "YlOrBr"
# Add session state for no-data color and opacity
if 'no_data_color' not in st.session_state:
    st.session_state.no_data_color = "#ffffff"  # Default white
if 'no_data_opacity' not in st.session_state:
    st.session_state.no_data_opacity = 0.5  # Default opacity of 50%
# Add session state for geography level
if 'geography_level' not in st.session_state:
    st.session_state.geography_level = "ZCTA"  # Default to ZCTA level

# Set page configuration
st.set_page_config(
    page_title="Choropleth Map Generator",
    layout="wide"
)

# Define color options
color_options = [
    "YlOrRd", "YlOrBr", "YlGnBu", "YlGn", "Reds", "RdPu", "Purples", "PuRd",
    "PuBuGn", "PuBu", "OrRd", "Oranges", "Greys", "Greens", "GnBu", "BuPu",
    "BuGn", "Blues", "Set2", "Paired", "Dark2", "RdYlBu", "RdBu", "PuOr",
    "PRGn", "PiYG", "BrBG"
]


# Function to load built-in shapefile from local path
@st.cache_data
def get_builtin_shapefile(shapefile_option):
    shapefile_paths = {
        "US ZCTA 2024": "/Users/javad/Documents/MEGA/Workspace/PyCharm/Temple/Windows_App/6_choropleth_generator/tl_2024_us_zcta520/tl_2024_us_zcta520.shp",
        "US State Boundaries": "/Users/javad/Documents/MEGA/Workspace/PyCharm/Temple/Windows_App/6_choropleth_generator/us-state-boundaries/us-state-boundaries.shp",
        "World Administrative Boundaries": "/Users/javad/Documents/MEGA/Workspace/PyCharm/Temple/Windows_App/6_choropleth_generator/world-administrative-boundaries/world-administrative-boundaries.shp"
    }

    shapefile_path = shapefile_paths.get(shapefile_option)
    if not shapefile_path:
        return None

    try:
        # Load the shapefile from the specified path
        if os.path.exists(shapefile_path):
            shp_gdf = gpd.read_file(shapefile_path)
            return shp_gdf
        else:
            st.error(f"Shapefile not found at: {shapefile_path}")
            return None
    except Exception as e:
        st.error(f"Error loading built-in shapefile: {str(e)}")
        return None


# Load state shapefile for use as a background in ZCTA maps
@st.cache_data
def get_state_boundaries():
    shapefile_path = "/Users/javad/Documents/MEGA/Workspace/PyCharm/Temple/Windows_App/6_choropleth_generator/us-state-boundaries/us-state-boundaries.shp"
    try:
        if os.path.exists(shapefile_path):
            state_gdf = gpd.read_file(shapefile_path)
            # Simplify geometry to reduce file size
            state_gdf['geometry'] = state_gdf['geometry'].simplify(tolerance=0.01)
            return state_gdf
        else:
            return None
    except Exception:
        return None


# Modified function to generate choropleth map for ZCTA, state, and country level
def generate_choropleth_map(df, shp_gdf, selected_column, min_value, max_value, step_value, fill_color,
                            custom_legend_name, custom_hover_name, no_data_color, no_data_opacity, geography_level):
    try:
        # Convert the selected column to numeric, coercing errors to NaN
        df[selected_column] = pd.to_numeric(df[selected_column], errors='coerce')

        # Determine the correct geography ID column name based on geography level
        geo_id_column = None
        join_column = None

        if geography_level == "ZCTA":
            # For ZCTA level
            if 'ZCTA5CE20' in shp_gdf.columns:
                geo_id_column = 'ZCTA5CE20'
            elif 'ZCTA5CE10' in shp_gdf.columns:
                geo_id_column = 'ZCTA5CE10'
            join_column = 'ZCTA'
        elif geography_level == "STATE":
            # For State level - specifically use geoid as you've indicated
            geo_id_column = 'geoid'  # Use lowercase geoid from shapefile
            join_column = 'geoid'
        else:  # COUNTRY level
            # For Country level - use ISO 3 country code
            geo_id_column = 'iso3'  # Column in shapefile
            join_column = 'ISO 3 country code'  # Column in CSV

        if geo_id_column:
            # Make sure geography ID column is treated as a string for proper joining
            shp_gdf[geo_id_column] = shp_gdf[geo_id_column].astype(str)
            if join_column in df.columns:
                df[join_column] = df[join_column].astype(str)

            # Create a base map
            # Calculate the center of the shapefile for initial view
            bounds = shp_gdf.geometry.total_bounds
            center_lat = (bounds[1] + bounds[3]) / 2
            center_lon = (bounds[0] + bounds[2]) / 2
            map_center = [center_lat, center_lon]

            # Different zoom levels based on geography
            if geography_level == "COUNTRY":
                initial_zoom = 2  # Wider view for world map
            else:
                initial_zoom = 4  # Default for US maps

            choropleth_map = folium.Map(location=map_center, zoom_start=initial_zoom)

            # Add background based on geography level
            if geography_level == "ZCTA":
                # First add state boundaries as background - this is efficient
                state_gdf = get_state_boundaries()
                if state_gdf is not None:
                    # Add state boundaries as white background
                    folium.GeoJson(
                        state_gdf.to_json(),
                        style_function=lambda x: {
                            'fillColor': no_data_color,
                            'fillOpacity': no_data_opacity,
                            'color': '#333333',
                            'weight': 0.5
                        },
                        name="Background"
                    ).add_to(choropleth_map)

                # Filter shapefile to only include ZCTAs that have data in the CSV for the DATA LAYER
                zcta_values_in_csv = df[join_column].unique().tolist()
                filtered_shp_gdf = shp_gdf[shp_gdf[geo_id_column].isin(zcta_values_in_csv)]

                # Simplify the geometry to further reduce size
                filtered_shp_gdf['geometry'] = filtered_shp_gdf['geometry'].simplify(tolerance=0.001)

                # Merge using the filtered shapefile for data visualization
                merged_df = filtered_shp_gdf.merge(df, left_on=geo_id_column, right_on=join_column, how='inner')

                # For hover functionality on all ZCTAs, create a separate simplified version of the shapefile
                hover_gdf = shp_gdf.copy()
                # Simplify geometry significantly to reduce size
                hover_gdf['geometry'] = hover_gdf['geometry'].simplify(tolerance=0.005)
                # Keep only essential columns to reduce memory
                hover_gdf = hover_gdf[[geo_id_column, 'geometry']]
            else:
                # For state or country level, add white background for areas without data
                folium.GeoJson(
                    shp_gdf.to_json(),
                    style_function=lambda x: {
                        'fillColor': no_data_color,
                        'fillOpacity': no_data_opacity,
                        'color': '#333333',
                        'weight': 0.5
                    },
                    name="Background"
                ).add_to(choropleth_map)

                # For country level, simplify geometry to improve performance
                if geography_level == "COUNTRY":
                    shp_gdf['geometry'] = shp_gdf['geometry'].simplify(tolerance=0.01)

                # Regular merge for state or country level
                merged_df = shp_gdf.merge(df, left_on=geo_id_column, right_on=join_column, how='inner')
                # For hover, use the original shapefile
                hover_gdf = shp_gdf.copy()

            # Check if merge was successful
            if len(merged_df) == 0:
                st.error(f"No matching {geography_level} found after merging data. Check your {join_column} values.")
                return None

            # Set default values for min, max, and step if they are empty
            if selected_column in merged_df.columns:
                data_min = merged_df[selected_column].min()
                data_max = merged_df[selected_column].max()
            else:
                st.error(f"Column '{selected_column}' not found after merging data.")
                return None

            # Use provided values or defaults
            if min_value is None:
                min_value = 0
            if max_value is None:
                max_value = data_max
            if step_value is None:
                step_value = max_value / 10 if max_value != 0 else 1

            # Calculate the bounding box of the data for optimized zoom
            bounds = merged_df.geometry.total_bounds
            # Convert bounds from (minx, miny, maxx, maxy) to [[south, west], [north, east]]
            sw = [bounds[1], bounds[0]]  # southwest corner [lat, lon]
            ne = [bounds[3], bounds[2]]  # northeast corner [lat, lon]

            # Add MousePosition to show coordinates on hover
            MousePosition().add_to(choropleth_map)

            # Create choropleth with standard options
            choropleth = folium.Choropleth(
                geo_data=merged_df.to_json(),
                name="Data Areas",
                data=merged_df,
                columns=[join_column, selected_column],
                key_on=f"feature.properties.{join_column}",
                fill_color=fill_color,
                fill_opacity=0.7,
                line_opacity=0.2,
                legend_name=custom_legend_name,
                highlight=True,
                overlay=True
            ).add_to(choropleth_map)

            # Add hover tooltips for all geographic areas (even those without data)
            highlight_function = lambda x: {'fillColor': '#000000',
                                            'color': '#000000',
                                            'fillOpacity': 0.50,
                                            'weight': 0.1}

            # Add separate hover layer for all areas
            if geography_level == "ZCTA":
                # For ZCTA level, add hover for ALL ZCTAs
                # Create a less detailed version of the shapefile for the hover to keep file size manageable
                # We create "batches" of ZCTAs to avoid message size limits

                # Determine batch size based on total number of ZCTAs
                total_zctas = len(hover_gdf)
                batch_size = 1000  # Adjust as needed

                # Create batches
                for start_idx in range(0, total_zctas, batch_size):
                    end_idx = min(start_idx + batch_size, total_zctas)
                    hover_batch = hover_gdf.iloc[start_idx:end_idx].copy()

                    # Add tooltip for this batch
                    folium.features.GeoJson(
                        data=hover_batch.to_json(),
                        style_function=lambda x: {'fillOpacity': 0.0, 'weight': 0, 'fillColor': 'transparent'},
                        control=False,
                        tooltip=folium.features.GeoJsonTooltip(
                            fields=[geo_id_column],
                            aliases=['ZCTA:'],
                            sticky=False)
                    ).add_to(choropleth_map)
            elif geography_level == "STATE":
                # For STATE level, add hover for all states
                tooltip_fields = [geo_id_column]
                tooltip_aliases = ['GEOID:']

                # Add state name if available
                if 'name' in hover_gdf.columns:
                    tooltip_fields.insert(0, 'name')
                    tooltip_aliases.insert(0, 'State Name:')

                folium.features.GeoJson(
                    data=hover_gdf.to_json(),
                    style_function=lambda x: {'fillOpacity': 0.0, 'weight': 0},
                    control=False,
                    tooltip=folium.features.GeoJsonTooltip(
                        fields=tooltip_fields,
                        aliases=tooltip_aliases,
                        sticky=False)
                ).add_to(choropleth_map)
            else:  # COUNTRY level
                # For COUNTRY level, add hover for all countries
                tooltip_fields = [geo_id_column]
                tooltip_aliases = ['ISO3:']

                # Add country name if available (using 'english' column for country name)
                if 'english' in hover_gdf.columns:
                    tooltip_fields.insert(0, 'english')
                    tooltip_aliases.insert(0, 'Country:')

                folium.features.GeoJson(
                    data=hover_gdf.to_json(),
                    style_function=lambda x: {'fillOpacity': 0.0, 'weight': 0},
                    control=False,
                    tooltip=folium.features.GeoJsonTooltip(
                        fields=tooltip_fields,
                        aliases=tooltip_aliases,
                        sticky=False)
                ).add_to(choropleth_map)

            # Add more detailed tooltips for areas with data
            if geography_level == "STATE":
                tooltip_fields = [join_column, selected_column]
                tooltip_aliases = [f'GEOID:', f'{custom_hover_name}:']
                # Add state name to tooltip if available
                if 'name' in merged_df.columns:
                    tooltip_fields.insert(0, 'name')
                    tooltip_aliases.insert(0, 'State Name:')
            elif geography_level == "COUNTRY":
                tooltip_fields = [join_column, selected_column]
                tooltip_aliases = [f'ISO3:', f'{custom_hover_name}:']
                # Add country name to tooltip if available
                if 'english' in merged_df.columns:
                    tooltip_fields.insert(0, 'english')
                    tooltip_aliases.insert(0, 'Country:')
                if 'Country' in merged_df.columns:
                    if 'english' not in merged_df.columns:
                        tooltip_fields.insert(0, 'Country')
                        tooltip_aliases.insert(0, 'Country:')
            else:  # ZCTA
                tooltip_fields = [join_column, selected_column]
                tooltip_aliases = [f'{geography_level}:', f'{custom_hover_name}:']

            # Add tooltip for areas with data (more detailed, shows the data values)
            data_tooltip = folium.features.GeoJson(
                data=merged_df.to_json(),
                style_function=lambda x: {'fillOpacity': 0.0, 'weight': 0},
                control=False,
                highlight_function=highlight_function,
                tooltip=folium.features.GeoJsonTooltip(
                    fields=tooltip_fields,
                    aliases=tooltip_aliases,
                    sticky=True)
            )
            choropleth_map.add_child(data_tooltip)

            # Set the map bounds to fit the data exactly
            if geography_level != "COUNTRY":
                # For ZCTA and STATE, fit bounds to the data
                choropleth_map.fit_bounds([sw, ne])
            # For COUNTRY, we keep the initial world view

            # Add layer control
            folium.LayerControl().add_to(choropleth_map)

            return choropleth_map
        else:
            st.error(f"Required ID column not found in the shapefile for {geography_level} level.")
            return None

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None


# Function to generate color bar images for the table
def generate_color_bar_image(color_name, width=300, height=10):
    """Generate a base64 encoded image of a color bar with clear boundaries"""
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=300)  # Higher DPI for clarity

    # For categorical colormaps like 'Paired', use a different approach to preserve distinct colors
    cmap = get_cmap(color_name)
    if color_name in ['Paired', 'Set1', 'Set2', 'Set3', 'Dark2', 'Pastel1', 'Pastel2', 'Accent', 'tab10', 'tab20']:
        # Categorical colormaps - generate discrete color blocks
        n_colors = min(12, cmap.N)  # Use up to 12 colors (common for qualitative colormaps)
        gradient = np.linspace(0, 1, n_colors, endpoint=False).reshape(1, -1)
        gradient = np.repeat(gradient, height, axis=0)  # Make it the desired height
        # Use nearest interpolation to avoid blending colors
        ax.imshow(gradient, aspect='auto', cmap=cmap, interpolation='nearest')
    else:
        # Continuous colormaps - generate smooth gradient
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        gradient = np.repeat(gradient, height, axis=0)  # Make it the desired height
        ax.imshow(gradient, aspect='auto', cmap=cmap)

    ax.set_axis_off()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Save to BytesIO buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=300)  # Higher DPI for sharper edges
    plt.close(fig)
    buf.seek(0)

    # Convert to base64
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return f'data:image/png;base64,{img_str}'


# Function to handle color selection from the grid
def select_color(color_name):
    st.session_state.selected_color = color_name
    return None


# Streamlit app layout
st.title("Choropleth Map Generator")
st.subheader("Create ZCTA, State, or Country Level Maps")

# Selection for geographic level
geography_level = st.radio(
    "Select Geographic Level",
    ["ZCTA", "STATE", "COUNTRY"],
    index=0 if st.session_state.geography_level == "ZCTA" else (
        1 if st.session_state.geography_level == "STATE" else 2),
    help="Choose the geographic level for your map"
)

# Update session state with the selected geography level
st.session_state.geography_level = geography_level

# Display guidance based on geography level
if st.session_state.geography_level == "ZCTA":
    st.info("For ZCTA level, your CSV should have a 'ZCTA' column with ZIP codes.")
elif st.session_state.geography_level == "STATE":
    st.info("For STATE level, your CSV should have a 'geoid' column matching the state geoid values in the shapefile.")
else:  # COUNTRY level
    st.info(
        "For COUNTRY level, your CSV should have an 'ISO 3 country code' column matching the ISO3 codes in the shapefile.")

# Single column layout
st.subheader("Upload CSV File")
csv_file = st.file_uploader("", type="csv",
                            help=f"Select a CSV file containing {st.session_state.geography_level} data")

if csv_file is not None:
    try:
        # Store the initial DataFrame in session state if not already loaded
        if st.session_state.csv_data is None or csv_file.name != getattr(st.session_state, 'last_uploaded_file', None):
            df = pd.read_csv(csv_file)

            # Convert ID column to string
            if st.session_state.geography_level == "ZCTA":
                id_column = 'ZCTA'
            elif st.session_state.geography_level == "STATE":
                id_column = 'geoid'
            else:  # COUNTRY level
                id_column = 'ISO 3 country code'

            if id_column in df.columns:
                df[id_column] = df[id_column].astype(str)

            st.session_state.csv_data = df
            st.session_state.last_uploaded_file = csv_file.name
            st.success("CSV file uploaded successfully")
        else:
            # Use the stored DataFrame
            df = st.session_state.csv_data

        # Show a preview of the data with editing capabilities
        st.write("Data Preview (editable):")
        edited_df = st.data_editor(
            df,
            num_rows="dynamic",
            height=300,
            use_container_width=True,
            key="csv_data_editor"
        )

        # If changes were made, update the session state
        if not edited_df.equals(st.session_state.csv_data):
            st.session_state.csv_data = edited_df
            df = edited_df
            st.info("CSV data has been modified. These changes will be used when generating the map.")

            # Add a download button for the edited CSV
            edited_csv = edited_df.to_csv(index=False)
            st.download_button(
                label="Download Edited CSV Data",
                data=edited_csv,
                file_name="edited_data.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"Error reading CSV file: {str(e)}")
        df = None

st.subheader("Shapefile Source")
shapefile_source = st.radio("Choose a shapefile source", ["Built-in", "Upload your own"])

if shapefile_source == "Built-in":
    if st.session_state.geography_level == "ZCTA":
        shapefile_option = st.selectbox(
            "Select built-in shapefile",
            ["US ZCTA 2024"]
        )
        st.markdown("""
        Source: [U.S. Census Bureau TIGER/Line® Shapefiles](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html)
        """)
    elif st.session_state.geography_level == "STATE":
        shapefile_option = st.selectbox(
            "Select built-in shapefile",
            ["US State Boundaries"]
        )
        st.markdown("""
        Source: [OpenDataSoft - US State Boundaries](https://public.opendatasoft.com/explore/dataset/us-state-boundaries/export/?flg=en-us)
        """)
    else:  # COUNTRY level
        shapefile_option = st.selectbox(
            "Select built-in shapefile",
            ["World Administrative Boundaries"]
        )
        st.markdown("""
        Source: [OpenDataSoft - World Administrative Boundaries](https://public.opendatasoft.com/explore/dataset/world-administrative-boundaries/information/?flg=en-us)
        """)

    if shapefile_option:
        with st.spinner(f"Loading {shapefile_option} shapefile..."):
            shp_gdf = get_builtin_shapefile(shapefile_option)

            if shp_gdf is not None:
                st.success(f"{shapefile_option} shapefile loaded successfully")

                # Add scrollable preview of complete shapefile data (editable)
                st.write("Shapefile Data Preview:")
                edited_shp_df = st.data_editor(
                    shp_gdf.drop(columns=['geometry']),
                    num_rows="dynamic",
                    height=400,
                    use_container_width=True,
                    disabled=False,
                    key="builtin_shapefile_editor"
                )

                # Add a download button for the edited shapefile data
                if not edited_shp_df.equals(shp_gdf.drop(columns=['geometry'])):
                    st.info(
                        "You've made changes to the shapefile data. Use the button below to download the edited version.")

                    # Convert the edited DataFrame to CSV
                    edited_shp_csv = edited_shp_df.to_csv(index=False)

                    # Create a download button for the edited CSV
                    st.download_button(
                        label="Download Edited Shapefile Data",
                        data=edited_shp_csv,
                        file_name="edited_shapefile_data.csv",
                        mime="text/csv"
                    )

                    # Show a note that original data is still used
                    st.caption(
                        "Note: The map will be generated using the original shapefile data, not the edited version.")
else:
    st.subheader("Upload Shapefile")
    shp_file = st.file_uploader("Choose a Shapefile", type="shp", help="Select a Shapefile containing boundaries")

    if shp_file is not None:
        # Create a temporary directory to save the uploaded shapefile
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save the shapefile to the temporary directory
            temp_shapefile_path = os.path.join(temp_dir, "temp.shp")
            with open(temp_shapefile_path, 'wb') as f:
                f.write(shp_file.getbuffer())

            # Check for additional shapefile components
            # Users need to upload all parts of the shapefile (.dbf, .shx, etc.)
            shp_components = st.file_uploader("Upload additional shapefile components (.dbf, .shx, .prj, .cpg, .xml)",
                                              type=["dbf", "shx", "prj", "cpg", "xml"],
                                              accept_multiple_files=True)

            if shp_components:
                for component in shp_components:
                    component_path = os.path.join(temp_dir, component.name)
                    with open(component_path, 'wb') as f:
                        f.write(component.getbuffer())

                try:
                    # Read the shapefile
                    shp_gdf = gpd.read_file(temp_shapefile_path)
                    st.success("Shapefile uploaded successfully")

                    # Add scrollable preview of complete shapefile data (editable)
                    st.write("Shapefile Data Preview (editable):")
                    edited_shp_df = st.data_editor(
                        shp_gdf.drop(columns=['geometry']),
                        num_rows="dynamic",
                        height=400,
                        use_container_width=True,
                        disabled=False,
                        key="uploaded_shapefile_editor"
                    )

                    # Add a download button for the edited shapefile data
                    if not edited_shp_df.equals(shp_gdf.drop(columns=['geometry'])):
                        st.info(
                            "You've made changes to the shapefile data. Use the button below to download the edited version.")

                        # Convert the edited DataFrame to CSV
                        edited_shp_csv = edited_shp_df.to_csv(index=False)

                        # Create a download button for the edited CSV
                        st.download_button(
                            label="Download Edited Shapefile Data",
                            data=edited_shp_csv,
                            file_name="edited_shapefile_data.csv",
                            mime="text/csv"
                        )

                        # Show a note that original data is still used
                        st.caption(
                            "Note: The map will be generated using the original shapefile data, not the edited version.")

                    # Check for appropriate ID columns based on geography level
                    if st.session_state.geography_level == "ZCTA":
                        zcta_cols = [col for col in shp_gdf.columns if 'ZCTA' in col.upper()]
                        if not zcta_cols:
                            st.warning(
                                "Warning: No ZCTA column found in the shapefile. Mapping may not work correctly.")
                    elif st.session_state.geography_level == "STATE":
                        state_cols = [col for col in shp_gdf.columns if
                                      col.upper() in ['GEOID', 'STATEFP', 'STUSPS', 'STATE',
                                                      'NAME'] or col.lower() == 'geoid']
                        if not state_cols:
                            st.warning(
                                "Warning: No standard state identifier columns found in the shapefile. Mapping may not work correctly.")
                    else:  # COUNTRY level
                        country_cols = [col for col in shp_gdf.columns if
                                        col.upper() in ['ISO3', 'ISO_A3', 'ISO', 'ISO 3 CODE'] or col.lower() == 'iso3']
                        if not country_cols:
                            st.warning(
                                "Warning: No standard country identifier columns found in the shapefile. Mapping may not work correctly.")
                except Exception as e:
                    st.error(f"Error reading shapefile: {str(e)}")
                    shp_gdf = None
            else:
                st.warning("Please upload the additional shapefile components (.dbf, .shx, .prj files)")
                shp_gdf = None

# Continue only if both files are uploaded
if ('df' in locals() and 'shp_gdf' in locals() and df is not None and shp_gdf is not None) or (
        'st.session_state.csv_data' in globals() and 'shp_gdf' in locals() and st.session_state.csv_data is not None and shp_gdf is not None):
    # Use the dataframe from session state if available
    if 'df' not in locals() or df is None:
        df = st.session_state.csv_data

    st.subheader("Map Configuration")

    # Check for required ID column based on geography level
    if st.session_state.geography_level == 'ZCTA':
        required_columns = ['ZCTA']
        id_column = 'ZCTA'
    elif st.session_state.geography_level == 'STATE':
        required_columns = ['geoid']
        id_column = 'geoid'
    else:  # COUNTRY level
        required_columns = ['ISO 3 country code']
        id_column = 'ISO 3 country code'

    if id_column not in df.columns:
        st.error(
            f"Required column '{id_column}' not found in your CSV data. For {st.session_state.geography_level} level, please ensure your data has this column.")
        # Show the available columns to help the user
        st.write("Available columns in your CSV:", ", ".join(df.columns))
    else:
        # Column selection - exclude ID column and default to the column after ID
        column_options = [col for col in df.columns if col != id_column]

        # Find the index of ID in the original columns and get the next column
        if id_column in df.columns:
            id_index = df.columns.get_loc(id_column)
            if id_index + 1 < len(df.columns):
                default_column = df.columns[id_index + 1]
            else:
                default_column = column_options[0] if column_options else None
        else:
            default_column = column_options[0] if column_options else None

        # Use index parameter to set the default selection
        default_index = column_options.index(default_column) if default_column in column_options else 0
        selected_column = st.selectbox("Select Data Column", column_options, index=default_index)

        # Get min, max values for the selected column
        if selected_column:
            numeric_values = pd.to_numeric(df[selected_column], errors='coerce')
            data_min = numeric_values.min()
            data_max = numeric_values.max()
            default_step = data_max / 10 if data_max != 0 else 1

        # Legend and hover text
        custom_legend_name = st.text_input("Legend Title",
                                           value=selected_column if 'selected_column' in locals() else "")
        custom_hover_name = st.text_input("Hover Label", value=selected_column if 'selected_column' in locals() else "")

        # Color selection
        st.subheader("Color Scheme Selection")

        # Create an expander for the color selection
        with st.expander("Click to select a color scheme"):
            # Display current selection at the top of the expander
            st.write(f"Currently selected: **{st.session_state.selected_color}**")

            # First get all color bar images
            color_bar_images = {}
            for color in color_options:
                color_bar_images[color] = generate_color_bar_image(color,
                                                                   height=20)  # Increased height for better visibility

            # Calculate number of columns for grid layout
            num_columns = 6

            # Create columns for the color grid
            cols = st.columns(num_columns)

            # Display color options in grid with clickable buttons
            for i, color_name in enumerate(color_options):
                col_idx = i % num_columns
                with cols[col_idx]:
                    # Create a button with the color name directly
                    # Use a different style for the selected button (bold text instead of underline)
                    if st.session_state.selected_color == color_name:
                        button_text = f"**{color_name}**"  # Bold text for selected color
                    else:
                        button_text = color_name

                    if st.button(
                            button_text,  # Use color name as the button text instead of "Select"
                            key=f"color_btn_{color_name}",
                            help=f"Select {color_name} color scheme",
                            use_container_width=True
                    ):
                        st.session_state.selected_color = color_name

                    # Display color bar image (without the color name above it)
                    st.image(
                        color_bar_images[color_name],
                        caption="",
                        use_container_width=True
                    )

                    # Add a small space instead of the blue line
                    st.markdown("<div style='height: 5px;'></div>", unsafe_allow_html=True)

        # Display the currently selected color scheme outside the expander
        st.write(f"Selected color scheme: **{st.session_state.selected_color}**")
        st.image(
            generate_color_bar_image(st.session_state.selected_color, width=1000, height=30),
        )

        # Range settings
        st.subheader("Value Range")

        min_value = st.number_input("Minimum Value", value=0.0)
        max_value = st.number_input("Maximum Value",
                                    value=float(data_max) if 'data_max' in locals() else 100.0)
        step_value = st.number_input("Step Value",
                                     value=float(default_step) if 'default_step' in locals() else 10.0)

        # No-data area settings
        st.subheader("Areas without Data")

        # Add custom CSS to make the color picker box wider
        st.markdown("""
        <style>
        div[data-testid="stColorPicker"] {
            width: 100%;
        }
        div[data-testid="stColorPicker"] > div {
            width: 100%;
        }
        div[data-testid="stColorPicker"] > div > div {
            width: 100% !important;
        }
        input[type="color"] {
            width: 100% !important;
            height: 50px !important;
        }
        </style>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            no_data_color = st.color_picker(
                "Color",
                value=st.session_state.no_data_color,
                help="Color for areas with no data"
            )
            # Update session state when changed
            if no_data_color != st.session_state.no_data_color:
                st.session_state.no_data_color = no_data_color

        with col2:
            # Replace slider with a number input for percentage
            opacity_percentage = st.number_input(
                "Opacity (%)",
                min_value=0,
                max_value=100,
                value=int(st.session_state.no_data_opacity * 100),
                help="Opacity percentage for areas with no data (0 = transparent, 100 = opaque)"
            )

            # Convert percentage to decimal value (0-1) and update session state
            opacity_decimal = opacity_percentage / 100.0
            if opacity_decimal != st.session_state.no_data_opacity:
                st.session_state.no_data_opacity = opacity_decimal

        # Generate map button
        if st.button("Generate Choropleth Map"):
            with st.spinner("Generating map..."):
                choropleth_map = generate_choropleth_map(
                    df, shp_gdf, selected_column, min_value, max_value, step_value,
                    st.session_state.selected_color, custom_legend_name, custom_hover_name,
                    st.session_state.no_data_color, st.session_state.no_data_opacity,
                    st.session_state.geography_level
                )

                if choropleth_map:
                    st.subheader("Choropleth Map")

                    # Custom CSS to ensure minimal spacing
                    st.markdown("""
                    <style>
                    .download-button-container {
                        margin-top: -15px;
                    }
                    </style>
                    """, unsafe_allow_html=True)

                    # Use the container_width parameter in folium_static
                    folium_static(choropleth_map, width=None, height=520)

                    # Generate HTML for download
                    map_html = choropleth_map._repr_html_()

                    # Apply the custom class to minimize space
                    st.markdown('<div class="download-button-container">', unsafe_allow_html=True)

                    # Generate appropriate filename based on geography level
                    level_text = "Country_Level" if st.session_state.geography_level == "COUNTRY" else (
                        "State_Level" if st.session_state.geography_level == "STATE" else "ZCTA_Level")
                    column_text = selected_column.replace(' ', '_')
                    map_filename = f"Choropleth_{level_text}_{column_text}.html"

                    # Download button with minimal space above
                    st.download_button(
                        label="Download Map as HTML",
                        data=map_html,
                        file_name=map_filename,
                        mime="text/html"
                    )

                    # Close the container
                    st.markdown('</div>', unsafe_allow_html=True)

                    st.success("Map generated successfully!")

else:
    st.info("Please upload a CSV file and select a shapefile option to generate a map.")

# Add some helpful instructions
with st.expander("How to Use This App"):
    st.markdown("""
    ### Instructions:

    1. **Select Geographic Level**:
       - Choose between ZCTA (ZIP Code Tabulation Areas), STATE, or COUNTRY level mapping
       - For ZCTA level, your CSV must have a 'ZCTA' column with ZIP codes
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
      - Your CSV file should have a 'ZCTA' column with ZIP codes
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
    """)

# Footer
st.markdown("---")
st.caption("Choropleth Map Generator | Copyright © 2025 [The Informatics Lab](https://theinformaticslab.com)")
