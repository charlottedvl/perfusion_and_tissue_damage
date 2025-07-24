from paraview.simple import *

# Load the .xdmf file
xdmf_file = OpenDataFile("results/p0000/perfusion_LMCAo/perfusion.xdmf")
xdmf_file.UpdatePipeline()

# Create slice filter
sliceFilter = Slice(Input=xdmf_file)
sliceFilter.SliceType = 'Plane'

# Set view
renderView = GetActiveViewOrCreate('RenderView')

# Set the background color to black
renderView.Background = [0.0, 0.0, 0.0] # RGB for black

rep = Show(sliceFilter, renderView)

# The Xdmf indicates "perfusion" is a scalar attribute with ElementDegree="0",
# which implies it's cell-wise constant data.
color_array_name = "perfusion"

# Explicitly color by "perfusion" as CELL_DATA
ColorBy(rep, ('CELL_DATA', color_array_name))

# Get the color transfer function (LUT) for the specific data array
lut = GetColorTransferFunction(color_array_name)

# Set the color space to RGB for a smooth gradient
lut.ColorSpace = 'RGB'

# --- CORRECTED WAY TO GET DATA RANGE ---
# Ensure the slice filter is updated to get the latest data information
sliceFilter.UpdatePipeline()
data_info = sliceFilter.GetDataInformation()
cell_arrays_info = data_info.GetCellDataInformation()

data_min = None
data_max = None

# Get the array information directly by name
array_info = cell_arrays_info.GetArrayInformation(color_array_name)

if array_info: # Check if the array info was successfully retrieved
    data_min, data_max = array_info.GetComponentRange(0)

if data_min is not None and data_max is not None:
    # Set RGB points: map data_min to black (0,0,0) and data_max to white (1,1,1)
    # The format is [data_value1, R1, G1, B1, data_value2, R2, G2, B2, ...]
    lut.RGBPoints = [
        data_min, 0.0, 0.0, 0.0,  # Map minimum data value to black
        data_max, 1.0, 1.0, 1.0   # Map maximum data value to white
    ]

    # Apply the configured lookup table to the representation
    rep.LookupTable = lut
    # Ensure the LUT is scaled to the data range
    rep.RescaleTransferFunctionToDataRange(True)

    # You can optionally hide the color bar if it's not desired for an MRI-like appearance
    # scalarBar = GetScalarBar(lut, renderView)
    # scalarBar.Visibility = 0 # Set to 0 to hide, 1 to show

else:
    print(f"Warning: Data range for '{color_array_name}' not found or is empty in cell data. Cannot set grayscale colormap accurately.")

# Function to save slice image
def save_slice(normal, origin, filename):
    # Update the slice plane
    sliceFilter.SliceType.Origin = origin
    sliceFilter.SliceType.Normal = normal
    sliceFilter.UpdatePipeline()

    # Set camera position to be some distance away from the slice along the normal
    global x_range, y_range, z_range
    distance = max(x_range, y_range, z_range) * 1.5

    cam_pos = [origin[i] + normal[i] * distance for i in range(3)]
    cam_focal = origin

    # Determine a view-up vector perpendicular to the normal
    if abs(normal[2]) < 0.9:
        view_up = [0, 0, 1]
    else:
        view_up = [0, 1, 0]

    # Adjust the render view camera
    renderView.CameraPosition = cam_pos
    renderView.CameraFocalPoint = cam_focal
    renderView.CameraViewUp = view_up

    # Use ResetCamera() to ensure clipping range is correctly set
    renderView.ResetCamera()

    # Save the screenshot
    SaveScreenshot(filename, renderView, ImageResolution=[800, 800])

# Get data bounds (these are geometry bounds, not data value bounds)
bounds = xdmf_file.GetDataInformation().GetBounds()
x_range = bounds[1] - bounds[0]
y_range = bounds[3] - bounds[2]
z_range = bounds[5] - bounds[4]

# Initial camera setup for the overall view
renderView.ResetCamera() # Ensure initial camera is also reset

# Take 4 slices in each direction
for i in range(1, 5):
    save_slice([1,0,0], [bounds[0] + i*x_range/5, 0, 0], f"x{i:02d}.png")
    save_slice([0,1,0], [0, bounds[2] + i*y_range/5, 0], f"y{i:02d}.png")
    save_slice([0,0,1], [0, 0, bounds[4] + i*z_range/5], f"z{i:02d}.png")
