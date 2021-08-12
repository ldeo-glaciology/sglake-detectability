This submission contains two datasets:

1. "H_beta.zarr" contains ice thickness (H) and basal sliding coefficient (beta)
maps for the Antarctic Ice Sheet. Spatial x-y coordinates are also included.
These can be loaded by running the python code: 'import xarray' and then 'H_beta
= xarray.open_zarr('data/H_beta.zarr')'. The data can then be manipulated with
xarray and numpy.

2. "active_lake_statistics.dat" contains subglacial lake locations and estimated
lengths for the Antarctic Ice Sheet in ASCII format. Cols 1 and 2 are the PS71 x
and y centroid of the lake outline. Cols 3, 4, and 5 are different potential
lake “lengths” to use. Cols 3 and 4 are the width and length of the minimum area
Feret diameter rectangles (where width is always the short side, length is
always the long side). Col. 5 is the diameter of an area-equivalent circle
(i.e., if the lake had the same area, but was a circle). Column 6 is the ice
thickness from BedMachine.
