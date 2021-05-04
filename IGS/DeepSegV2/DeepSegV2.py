import os
import unittest
import logging
import vtk, qt, ctk, slicer

# DeepSegV2 imports
try:
  import matplotlib.pyplot as plt
except:
  slicer.util.pip_install('matplotlib')
  import matplotlib.pyplot as plt
# TODO: add other imports
try:
  import numpy as np
except:
  slicer.util.pip_install('numpy~=1.19.2')
  import numpy as np
try:
  import nibabel as nib
except:
  slicer.util.pip_install('nibabel')
  import nibabel as nib
try:
  from nilearn.image import crop_img as crop_image
except:
  slicer.util.pip_install('nilearn')
  from nilearn.image import crop_img as crop_image

import sys
sys.argv = ['pdm']
import tensorflow.python

try:
  import tensorflow as tf
except:
  slicer.util.pip_install('tensorflow')
  import tensorflow as tf


#import numpy as np
#import nibabel as nib
#import tensorflow as tf

# utlity functions imports
#import matplotlib.pyplot as plt
#from nilearn.image import crop_img as crop_image


from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin


config = dict()

# define input
config["image_shape"] = (192, 224, 160) # the input to the pre-trained model

config["input_dir"] = 'BraTS20_sample_case' # directory of the input image(s)
config["preprocess_dir"] = 'BraTS20_sample_case_preprocess' # directory of the pre-processed image(s)
config["predict_dir"] = 'BraTS20_sample_case_predict' # directory of the predicted segmentation
config["predict_name"] = 'BraTS20_sample_case_pred.nii.gz' # name of the predicted segmentation

#config["image_path"] = 'BraTS20_sample_case'
#config["image_path_preprocess"] = 'BraTS20_sample_case_preprocess'
#config['image_path_predict'] = os.path.join('BraTS20_sample_case_predict','BraTS20_sample_case_pred.nii.gz')

# define used MRI modalities 
config["all_modalities"] = ["t1", "t1ce", "flair", "t2"]
config["image_modalities"] = config["all_modalities"]
# one variable for each MRI modality
config["image_1"] = 'BraTS20_sample_case_flair.nii.gz'
config["image_2"] = 'BraTS20_sample_case_t1.nii.gz'
config["image_3"] = 'BraTS20_sample_case_t1ce.nii.gz'
config["image_4"] = 'BraTS20_sample_case_t2.nii.gz'

# OR one variable for all MRI modalities
config["images"] = ['BraTS20_sample_case_flair.nii.gz', 'BraTS20_sample_case_t1.nii.gz', 
                     'BraTS20_sample_case_t1ce.nii.gz', 'BraTS20_sample_case_t2.nii.gz']

# model parameters
config["input_shape"] = (config["image_shape"][0], config["image_shape"][1], 
                         config["image_shape"][2], len(config["images"]))

config['model_path'] = os.path.join('weights', 'model-238.h5')
config['tumor_type'] = "all" # "all", "whole", "core", "enhancing"

#
# DeepSegV2
#

class DeepSegV2(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "DeepSegV2"  # TODO: make this more human readable by adding spaces
    self.parent.categories=["Examples"]
    self.parent.dependencies = []  # TODO: add here list of module names that this module requires
    self.parent.contributors=["Ramy Zeineldin"]
    # TODO: update with short description of the module and a link to online module documentation
    self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#DeepSegV2">module documentation</a>.
"""
    # TODO: replace with organization, grant and thanks
    self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
"""

    # Additional initialization step after application startup is complete
    slicer.app.connect("startupCompleted()", registerSampleData)

#
# Register sample data sets in Sample Data module
#

def registerSampleData():
  """
  Add data sets to Sample Data module.
  """
  # It is always recommended to provide sample data for users to make it easy to try the module,
  # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

  import SampleData
  iconsPath = os.path.join(os.path.dirname(__file__), 'Resources/Icons')

  # To ensure that the source code repository remains small (can be downloaded and installed quickly)
  # it is recommended to store data sets that are larger than a few MB in a Github release.

  # DeepSegV21
  SampleData.SampleDataLogic.registerCustomSampleDataSource(
    # Category and sample name displayed in Sample Data module
    category='DeepSegV2',
    sampleName='DeepSegV21',
    # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
    # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
    thumbnailFileName=os.path.join(iconsPath, 'DeepSegV21.png'),
    # Download URL and target file name
    uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
    fileNames='DeepSegV21.nrrd',
    # Checksum to ensure file integrity. Can be computed by this command:
    #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
    checksums = 'SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95',
    # This node name will be used when the data set is loaded
    nodeNames='DeepSegV21'
  )

  # DeepSegV22
  SampleData.SampleDataLogic.registerCustomSampleDataSource(
    # Category and sample name displayed in Sample Data module
    category='DeepSegV2',
    sampleName='DeepSegV22',
    thumbnailFileName=os.path.join(iconsPath, 'DeepSegV22.png'),
    # Download URL and target file name
    uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
    fileNames='DeepSegV22.nrrd',
    checksums = 'SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97',
    # This node name will be used when the data set is loaded
    nodeNames='DeepSegV22'
  )

#
# DeepSegV2Widget
#

class DeepSegV2Widget(ScriptedLoadableModuleWidget, VTKObservationMixin):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent=None):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.__init__(self, parent)
    VTKObservationMixin.__init__(self)  # needed for parameter node observation
    self.logic = None
    self._parameterNode = None
    self._updatingGUIFromParameterNode = False

  def setup(self):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.setup(self)

    # Load widget from .ui file (created by Qt Designer).
    # Additional widgets can be instantiated manually and added to self.layout.
    uiWidget = slicer.util.loadUI(self.resourcePath('UI/DeepSegV2.ui'))
    self.layout.addWidget(uiWidget)
    self.ui = slicer.util.childWidgetVariables(uiWidget)

    # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
    # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
    # "setMRMLScene(vtkMRMLScene*)" slot.
    uiWidget.setMRMLScene(slicer.mrmlScene)

    # Create logic class. Logic implements all computations that should be possible to run
    # in batch mode, without a graphical user interface.
    self.logic = DeepSegV2Logic()

    # Connections

    # These connections ensure that we update parameter node when scene is closed
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

    # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
    # (in the selected parameter node).
    #self.ui.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    self.ui.FLAIRSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    self.ui.T1Selector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    self.ui.T1ceSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    self.ui.T2Selector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)    
    self.ui.outputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)

    # Buttons
    self.ui.applyButton.connect('clicked(bool)', self.onApplyButton)

    # Make sure parameter node is initialized (needed for module reload)
    self.initializeParameterNode()

  def cleanup(self):
    """
    Called when the application closes and the module widget is destroyed.
    """
    self.removeObservers()

  def enter(self):
    """
    Called each time the user opens this module.
    """
    # Make sure parameter node exists and observed
    self.initializeParameterNode()

  def exit(self):
    """
    Called each time the user opens a different module.
    """
    # Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)
    self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

  def onSceneStartClose(self, caller, event):
    """
    Called just before the scene is closed.
    """
    # Parameter node will be reset, do not use it anymore
    self.setParameterNode(None)

  def onSceneEndClose(self, caller, event):
    """
    Called just after the scene is closed.
    """
    # If this module is shown while the scene is closed then recreate a new parameter node immediately
    if self.parent.isEntered:
      self.initializeParameterNode()

  def initializeParameterNode(self):
    """
    Ensure parameter node exists and observed.
    """
    # Parameter node stores all user choices in parameter values, node selections, etc.
    # so that when the scene is saved and reloaded, these settings are restored.

    self.setParameterNode(self.logic.getParameterNode())

    # Select default input nodes if nothing is selected yet to save a few clicks for the user
    if not self._parameterNode.GetNodeReference("InputVolume1"):
      firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
      if firstVolumeNode:
        self._parameterNode.SetNodeReferenceID("InputVolume1", firstVolumeNode.GetID())

  def setParameterNode(self, inputParameterNode):
    """
    Set and observe parameter node.
    Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
    """

    if inputParameterNode:
      self.logic.setDefaultParameters(inputParameterNode)

    # Unobserve previously selected parameter node and add an observer to the newly selected.
    # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
    # those are reflected immediately in the GUI.
    if self._parameterNode is not None:
      self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
    self._parameterNode = inputParameterNode
    if self._parameterNode is not None:
      self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    # Initial GUI update
    self.updateGUIFromParameterNode()

  def updateGUIFromParameterNode(self, caller=None, event=None):
    """
    This method is called whenever parameter node is changed.
    The module GUI is updated to show the current state of the parameter node.
    """

    if self._parameterNode is None or self._updatingGUIFromParameterNode:
      return

    # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
    self._updatingGUIFromParameterNode = True

    # Update node selectors and sliders
    self.ui.FLAIRSelector.setCurrentNode(self._parameterNode.GetNodeReference("InputVolume1"))
    self.ui.T1Selector.setCurrentNode(self._parameterNode.GetNodeReference("InputVolume2"))
    self.ui.T1ceSelector.setCurrentNode(self._parameterNode.GetNodeReference("InputVolume3"))
    self.ui.T2Selector.setCurrentNode(self._parameterNode.GetNodeReference("InputVolume4"))
    self.ui.outputSelector.setCurrentNode(self._parameterNode.GetNodeReference("OutputVolume"))

    # Update buttons states and tooltips
    if self._parameterNode.GetNodeReference("InputVolume1") and self._parameterNode.GetNodeReference("OutputVolume"):
      self.ui.applyButton.toolTip = "Compute output segmentation"
      self.ui.applyButton.enabled = True
    else:
      self.ui.applyButton.toolTip = "Select input and output volume nodes"
      self.ui.applyButton.enabled = False

    # All the GUI updates are done
    self._updatingGUIFromParameterNode = False

  def updateParameterNodeFromGUI(self, caller=None, event=None):
    """
    This method is called when the user makes any change in the GUI.
    The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
    """

    if self._parameterNode is None or self._updatingGUIFromParameterNode:
      return

    wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

    #self._parameterNode.SetNodeReferenceID("InputVolume", self.ui.inputSelector.currentNodeID)
    self._parameterNode.SetNodeReferenceID("InputVolume1", self.ui.FLAIRSelector.currentNodeID)
    self._parameterNode.SetNodeReferenceID("InputVolume2", self.ui.T1Selector.currentNodeID)
    self._parameterNode.SetNodeReferenceID("InputVolume3", self.ui.T1ceSelector.currentNodeID)
    self._parameterNode.SetNodeReferenceID("InputVolume4", self.ui.T2Selector.currentNodeID)
    self._parameterNode.SetNodeReferenceID("OutputVolume", self.ui.outputSelector.currentNodeID)

    self._parameterNode.EndModify(wasModified)

  def onApplyButton(self):
    """
    Run processing when user clicks "Apply" button.
    """

    try:
      # Compute output
      #print("Hello DeepSegV2")
      import time
      startTime = time.time()
      logging.info('Processing started')

      #slicer.util.updateVolumeFromArray(outputVolume, img)

      img_norm = self.logic.norm_image(self.ui.FLAIRSelector.currentNode())
      slicer.util.updateVolumeFromArray(self.ui.outputSelector.currentNode(), img_norm)
      #self.logic.norm_image(self.ui.FLAIRSelector.currentNode(), self.ui.outputSelector.currentNode())
      #self.logic.preprocess_images(input_dir=config['input_dir'], preprocess_dir=config['preprocess_dir'], images=config["images"], dim=config["image_shape"])


      #self.logic.process(self.ui.FLAIRSelector.currentNode(), self.ui.outputSelector.currentNode(),
      #  self.ui.imageThresholdSliderWidget.value, self.ui.invertOutputCheckBox.checked)


      stopTime = time.time()
      logging.info('Processing completed in {0:.2f} seconds'.format(stopTime-startTime))

    except Exception as e:
      slicer.util.errorDisplay("Failed to compute results: "+str(e))
      import traceback
      traceback.print_exc()


#
# DeepSegV2Logic
#

class DeepSegV2Logic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self):
    """
    Called when the logic class is instantiated. Can be used for initializing member variables.
    """
    ScriptedLoadableModuleLogic.__init__(self)

  def setDefaultParameters(self, parameterNode):
    """
    Initialize parameter node with default settings.
    """
    if not parameterNode.GetParameter("Threshold"):
      parameterNode.SetParameter("Threshold", "100.0")
    if not parameterNode.GetParameter("Invert"):
      parameterNode.SetParameter("Invert", "false")

  """def norm_image(self, img, norm_type = "norm"):
    if norm_type == "standard_norm": # standarization, same dataset
        img_mean = img.mean()
        img_std = img.std()
        img_std = 1 if img.std()==0 else img.std()
        img = (img - img_mean) / img_std
    elif norm_type == "norm": # different datasets
        img = (img - np.min(img))/(np.ptp(img)) # (np.max(img) - np.min(img))
    elif norm_type == "norm_slow": # different datasets
#         img = (img - np.min(img))/(np.max(img) - np.min(img))
        img_ptp = 1 if np.ptp(img)== 0 else np.ptp(img) 
        img = (img - np.min(img))/img_ptp
    return img"""

  def norm_image(self, inputVolume, norm_type = "norm"):
    if not inputVolume: # or not outputVolume:
      raise ValueError("Input volume is invalid")

    # Compute the thresholded output volume using the "Threshold Scalar Volume" CLI module
    #img = slicer.util.addVolumeFromArray(inputVolume)
    img = slicer.util.arrayFromVolume(inputVolume)

    if norm_type == "standard_norm": # standarization, same dataset
        img_mean = img.mean()
        img_std = img.std()
        img_std = 1 if img.std()==0 else img.std()
        img = (img - img_mean) / img_std
    elif norm_type == "norm": # different datasets
        img = (img - np.min(img))/(np.ptp(img)) # (np.max(img) - np.min(img))
    elif norm_type == "norm_slow": # different datasets
#         img = (img - np.min(img))/(np.max(img) - np.min(img))
        img_ptp = 1 if np.ptp(img)== 0 else np.ptp(img) 
        img = (img - np.min(img))/img_ptp

    return img

  def preprocess_images(self, input_dir, preprocess_dir, images, dim=config["image_shape"]):
    for img in images:
        #print("Preprocessing: ", img)

        # load the MRI imaging modalities (flair, t1, t1ce, t2)
        img_nifti = nib.load(os.path.join(input_dir, img)) #.get_fdata(dtype='float32')

        # crop the input image
        img_preprocess = crop_image(img_nifti)
    
        # convert into numpy array
        img_array = np.array(img_preprocess.get_fdata(dtype='float32'))

        # pad the preprocessed image
        padded_image = np.zeros((dim[0],dim[1],dim[2]))
        padded_image[:img_array.shape[0],:img_array.shape[1],:img_array.shape[2]] = img_array
        
        # save nifti images
        img_preprocess_nifti = nib.Nifti1Image(self.norm_image(padded_image), img_nifti.affine, img_nifti.header) 
        if not os.path.exists(preprocess_dir):
            os.makedirs(preprocess_dir)
        nib.save(img_preprocess_nifti, os.path.join(preprocess_dir, img))


"""
  def process(self, inputVolume, outputVolume, imageThreshold, showResult=True):

    if not inputVolume or not outputVolume:
      raise ValueError("Input or output volume is invalid")

    import time
    startTime = time.time()
    logging.info('Processing started')

    # Compute the thresholded output volume using the "Threshold Scalar Volume" CLI module
    cliParams = {
      'InputVolume': inputVolume.GetID(),
      'OutputVolume': outputVolume.GetID(),
      'ThresholdValue' : imageThreshold,
      'ThresholdType' : 'Above' if invert else 'Below'
      }
    cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True, update_display=showResult)
    # We don't need the CLI module node anymore, remove it to not clutter the scene with it
    slicer.mrmlScene.RemoveNode(cliNode)

    stopTime = time.time()
    logging.info('Processing completed in {0:.2f} seconds'.format(stopTime-startTime))
"""

#
# DeepSegV2Test
#

class DeepSegV2Test(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear()

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    self.test_DeepSegV21()

  def test_DeepSegV21(self):
    """ Ideally you should have several levels of tests.  At the lowest level
    tests should exercise the functionality of the logic with different inputs
    (both valid and invalid).  At higher levels your tests should emulate the
    way the user would interact with your code and confirm that it still works
    the way you intended.
    One of the most important features of the tests is that it should alert other
    developers when their changes will have an impact on the behavior of your
    module.  For example, if a developer removes a feature that you depend on,
    your test should break so they know that the feature is needed.
    """

    self.delayDisplay("Starting the test")

    # Get/create input data

    import SampleData
    registerSampleData()
    inputVolume = SampleData.downloadSample('DeepSegV21')
    self.delayDisplay('Loaded test data set')

    inputScalarRange = inputVolume.GetImageData().GetScalarRange()
    self.assertEqual(inputScalarRange[0], 0)
    self.assertEqual(inputScalarRange[1], 695)

    outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
    threshold = 100

    # Test the module logic

    logic = DeepSegV2Logic()

    # Test algorithm with threshold
    logic.process(inputVolume, outputVolume, threshold, True)
    outputScalarRange = outputVolume.GetImageData().GetScalarRange()
    self.assertEqual(outputScalarRange[0], inputScalarRange[0])
    self.assertEqual(outputScalarRange[1], threshold)

    self.delayDisplay('Test passed')
