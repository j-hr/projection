#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# LoadState must be here even though I later overwrite all settings manually
# without it it won't split windows properly
servermanager.LoadState("$DIR$/empty.pvsm")

# set active view
SetActiveView(None)

# Create a new 'Render View'
rv1 = CreateView('RenderView')
rv1.ViewSize = [1101, 656]
rv1.AxesGrid = 'GridAxes3DActor'
rv1.StereoType = 0
rv1.Background = [1.0, 1.0, 1.0]
rv1.OrientationAxesLabelColor = [0.0, 0.0, 0.0]

# get layout
viewLayout1 = GetLayout()

# place view in the layout
viewLayout1.AssignView(0, rv1)

load = [
    ('$FILENAME1$','$VECTORNAME1$',rv1),
]

for (path, vector_name, rv) in load:
    # set active view
    SetActiveView(rv)
    # create a new 'XDMF Reader'
    wss = XDMFReader(FileNames=[path])
    wss.PointArrayStatus = [vector_name]
    # get animation scene
    animationScene1 = GetAnimationScene()
    # update animation scene based on data timesteps
    animationScene1.UpdateAnimationUsingDataTimeSteps()
    animationScene1.GoToNext()
    # show data in view
    wssDisplay = Show(wss, rv)
    # trace defaults for the display properties.
    wssDisplay.ColorArrayName = [None, '']
    # wssDisplay.ScalarOpacityUnitDistance = 0.6518746966631972  # nevím, co dělá
    wssDisplay.Opacity = 1.0

    # reset view to fit data
    rv.ResetCamera()
    
    # scalar coloring
    ColorBy(wssDisplay, ('POINTS', vector_name))

    # get color transfer function/color map and define old 'Cool to Warm' scheme
    color = GetColorTransferFunction(vector_name)
    color.RGBPoints = [0.0, 0.2196078431372549, 0.3058823529411765, 0.7568627450980392, 1.0, 0.7058823529411765, 0.01568627450980392, 0.11764705882352941]
    # get opacity transfer function/opacity map for 'IBCI'
    opacity = GetOpacityTransferFunction(vector_name)
    opacity.Points = [0.0, 1.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
    color.ColorSpace = 'Diverging'
    # Rescale transfer function
    color.RescaleTransferFunctionToDataRange(False)

    # get color transfer function/color map for 'Vector'
    legend = GetScalarBar(color, rv)
    # set legend text color to black
    legend.TitleColor = [0.0, 0.0, 0.0]
    legend.LabelColor = [0.0, 0.0, 0.0]
    legend.Title = vector_name

    animationScene1.GoToFirst()
