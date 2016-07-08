#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# LoadState must be here even though I later overwrite all settings manually
# without it it won't split windows properly
servermanager.LoadState("paraview_scripts/empty_state_velocity.pvsm")

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

# split cell
viewLayout1.SplitHorizontal(0, 0.5)

# set active view
SetActiveView(None)

# Create a new 'Render View'
rv2 = CreateView('RenderView')
rv2.ViewSize = [546, 656]
rv2.AxesGrid = 'GridAxes3DActor'
rv2.StereoType = 0
rv2.Background = [1.0, 1.0, 1.0]
rv2.OrientationAxesLabelColor = [0.0, 0.0, 0.0]

# place view in the layout
viewLayout1.AssignView(2, rv2)

load = [
    ('$FILENAME1$','$VECTORNAME1$',rv1),
    ('$FILENAME2$','$VECTORNAME2$',rv2),
]

for (path, vector_name, rv) in load:
    # set active view
    SetActiveView(rv)
    # create a new 'XDMF Reader'
    velocity = XDMFReader(FileNames=[path])
    velocity.PointArrayStatus = [vector_name]
    # get animation scene
    animationScene1 = GetAnimationScene()
    # update animation scene based on data timesteps
    animationScene1.UpdateAnimationUsingDataTimeSteps()
    animationScene1.GoToNext()
    # show data in view
    velocityDisplay = Show(velocity, rv)
    # trace defaults for the display properties.
    velocityDisplay.ColorArrayName = [None, '']
    # velocityDisplay.ScalarOpacityUnitDistance = 0.6518746966631972  # nevím, co dělá
    velocityDisplay.Opacity = 0.2

    # reset view to fit data
    rv.ResetCamera()
    
    # turn off scalar coloring
    ColorBy(velocityDisplay, None)

    # create a new 'Glyph'
    glyph1 = Glyph(Input=velocity, GlyphType='Arrow')
    glyph1.Scalars = [None, '']
    glyph1.Vectors = ['POINTS', vector_name]
    glyph1.ScaleFactor = $FACTOR$
    glyph1.GlyphTransform = 'Transform2'
    glyph1.GlyphMode = 'All Points'
    glyph1.ScaleMode = 'vector'

    # show data in view
    glyph1Display = Show(glyph1, rv)
    # trace defaults for the display properties.
    glyph1Display.ColorArrayName = [None, '']

    # set scalar coloring
    ColorBy(glyph1Display, ('POINTS', 'GlyphVector'))

    # show color bar/color legend
    glyph1Display.SetScalarBarVisibility(rv, True)

    # get color transfer function/color map and define old 'Cool to Warm' scheme
    color_v = GetColorTransferFunction('GlyphVector')
    color_v.RGBPoints = [0.0, 0.2196078431372549, 0.3058823529411765, 0.7568627450980392, 1.0, 0.7058823529411765, 0.01568627450980392, 0.11764705882352941]
    # get opacity transfer function/opacity map for 'IBCI'
    opacity_v = GetOpacityTransferFunction('GlyphVector')
    opacity_v.Points = [0.0, 1.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
    color_v.ColorSpace = 'Diverging'
    # Rescale transfer function
    color_v.RescaleTransferFunctionToDataRange(False)

    # get color transfer function/color map for 'GlyphVector'
    glyph_legend = GetScalarBar(color_v, rv)
    # set legend text color to black
    glyph_legend.TitleColor = [0.0, 0.0, 0.0]
    glyph_legend.LabelColor = [0.0, 0.0, 0.0]
    glyph_legend.Title = vector_name



    animationScene1.GoToFirst()

AddCameraLink(rv1,rv2,'link1')
