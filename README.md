# MaybeMap - Professional Regional Mapping Tool

A Python application that creates professional-quality regional maps from Excel data for business presentations and internal reference.

## Overview

MaybeMap transforms Excel spreadsheets containing city and group information into polished, executive-ready maps with:
- **Geographic-aware smart labeling** (labels placed toward empty space)
- **Numbered dense area system** (organized spiral placement for clustered cities)
- **Professional cartographic labeling** (zero overlapping text guaranteed)
- **Hierarchical typography** (major cities emphasized)
- **Interactive HTML maps** for digital exploration
- **High-resolution PNG exports** for print/presentations

## Quick Start

### Prerequisites
- Python 3.8 or higher
- [Geoapify API key](https://www.geoapify.com/) (free tier available)

### Installation
1. Clone or download this project
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your API key:
   ```
   GEOAPIFY_KEY=your_api_key_here
   ```
4. Prepare your Excel file (see Data Format below)

### Running
```bash
python maybeMap.py
```

**First run takes 10-15 minutes** for geocoding and boundary fetching. Subsequent runs are much faster due to caching.

## Data Format

Your Excel file (`Socket-Regions.xlsx`) must contain these columns:
- **City**: City name (e.g., "Saint Louis")
- **State**: Two-letter state code (e.g., "MO") 
- **Group**: Regional grouping (e.g., "Group 1: Metro East")

Example:
```
City          | State | Group
Saint Louis   | MO    | Group 1: Metro East
Chicago       | IL    | Group 2: Illinois
Springfield   | MO    | Group 1: Metro East
```

## Output Files

### 🖼️ Static Map (`grouped_regions_map.png`)
- High-resolution PNG (220 DPI)
- **Geographic-aware label placement** toward empty spaces
- **Numbered dense areas** with spiral arrangement and reference legend
- **Zero overlapping labels** guaranteed for professional appearance
- **Manual positioning overrides** for specific cities
- Perfect for presentations, reports, and printing

### 🌐 Interactive Map (`grouped_regions_map.html`)
- Click cities for detailed information popups
- Toggle layers on/off
- Fullscreen presentation mode
- Professional legend and controls
- Perfect for digital exploration and meetings

### 📊 Data Export (`grouped_regions.geojson`)
- Geographic data in standard GeoJSON format
- Compatible with GIS software and web applications

## Features

### 🎯 Advanced Label Placement System
- **Geographic-aware positioning** - Labels placed toward empty spaces using 8-directional density analysis
- **Edge city intelligence** - Cities near map edges get inward-facing labels
- **Interior city optimization** - Labels directed toward least dense areas
- **Numbered dense areas** - Cities with 16+ neighbors get organized numbered placement
- **Spiral arrangement** - Numbered labels arranged in professional spiral pattern
- **Zero overlapping labels** - Guaranteed readable text placement

### 🏗️ Professional Cartography
- **Hierarchical typography** - Major cities in bold, larger fonts
- **Adaptive collision detection** - Relaxed separation for very dense areas
- **Manual positioning overrides** - Fine-tune specific city placements
- **Clean leader lines** - Professional connecting lines with optimal routing
- **Density-first processing** - Dense area cities get priority placement

### 🏙️ City Classification System
- **Primary Cities**: Major metropolitan areas (largest markers/fonts)
- **Secondary Cities**: State capitals and significant cities  
- **Tertiary Cities**: All other cities
- **Dense Area Cities**: Automatically numbered when 16+ neighbors detected

### 🔄 Smart Data Processing
- **Automatic geocoding** via Geoapify API
- **OSM boundary fetching** for accurate city shapes
- **Intelligent caching** for fast subsequent runs
- **Fallback systems** for missing data
- **Density-based sorting** for optimal placement order

## Configuration

Key settings in `maybeMap.py`:

```python
# Map styling
BASEMAP = "voyager"              # "voyager", "dark", or "positron"
EXPORT_INTERACTIVE_HTML = True

# Professional typography
LABEL_FONTSIZE_PRIMARY = 11      # Major cities
LABEL_FONTSIZE_SECONDARY = 9     # Medium cities  
LABEL_FONTSIZE_TERTIARY = 8      # Small cities

# Advanced label placement
LABEL_MIN_SEPARATION_M = 8000    # Base minimum separation (8km)
MAX_LABEL_DISTANCE_M = 80000     # Maximum leader line length (80km)
dense_area_threshold = 16        # Cities with 16+ neighbors get numbered

# Spiral placement for dense areas
angle_step = 0.5                 # Radians between spiral positions
radius_step = 8000               # 8km radius increase per spiral turn
```

## File Structure

```
MaybeMap-1/
├── maybeMap.py              # Main application
├── requirements.txt         # Python dependencies
├── Socket-Regions.xlsx      # Your input data
├── .env                     # API keys (create this)
├── README.md               # This file
├── CLAUDE.md               # Developer guidance
├── city_geocodes_cache.json     # Geocoding cache (auto-generated)
├── osm_boundaries_cache.json    # Boundary cache (auto-generated)
├── grouped_regions_map.png      # Output: Static map
├── grouped_regions_map.html     # Output: Interactive map
└── grouped_regions.geojson      # Output: Geographic data
```

## CLI Progress Output

The application provides real-time progress feedback:

```
Starting geocoding process for 61 unique cities...
   Found coordinates: 38.6270, -90.1994
...
Geocoding complete! Processed 61 cities.

Starting OSM boundary fetching for 61 cities...
   Found OSM boundary polygon (area: 162.45 km²)
...
OSM boundary fetching complete! Processed 61 cities.

Generating professional cartographic map...
Applying professional cartographic labeling system...
   Map analysis: 58 cities analyzed
      East Alton: Found preferred position at 8000m after 1 attempts
   Numbered Cahokia Heights: #1 pin(-10040052, 4660327) -> label(-10018112, 4672313)
   Placing Columbia: pin(-10040995, 4642170) -> label(-10065995, 4617170)
   Professional labeling complete - no overlapping labels!
```

## Troubleshooting

### Common Issues

**"Missing GEOAPIFY_KEY" error**
- Create a `.env` file with your API key
- Get a free key at [geoapify.com](https://www.geoapify.com/)

**Slow first run**
- Normal behavior - building geocoding and boundary caches
- Subsequent runs will be much faster
- You can monitor progress in the CLI output

**Cities not found**
- Check city name spelling in Excel file
- Verify state abbreviations are correct (two letters)
- Some small towns may not be in the geocoding database

**Empty or missing boundaries**
- Script automatically falls back to circular buffers around points
- This is normal for smaller cities without detailed OSM data

## Use Cases

### Business Applications
- **Regional sales territories** mapping
- **Market analysis** visualization  
- **Operations planning** reference maps
- **Executive presentations** with professional appearance
- **Internal documentation** with interactive exploration

### Technical Applications
- **Geographic data visualization** from spreadsheets
- **Quick prototyping** of regional analysis
- **Data validation** through visual inspection

## API Dependencies

- **Geoapify**: City geocoding (latitude/longitude lookup)
- **OpenStreetMap**: City boundary polygons via OSMnx
- **CartoDB**: Base map tiles for visualization

## Development

See `CLAUDE.md` for detailed development guidance and architecture notes.

## License

Internal business tool - modify as needed for your organization's requirements.