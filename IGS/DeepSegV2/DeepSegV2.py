import os
import unittest
import logging
import vtk, qt, ctk, slicer

from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

####################### add your imports here #######################
"""
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
"""
import numpy as np
import nibabel as nib

import sys
sys.argv = ['pdm']
import tensorflow.python
import tensorflow as tf

# utlity functions imports
import matplotlib.pyplot as plt
from tensorflow.keras.utils import get_file

#from nilearn.image import crop_img as crop_image

# import functions from models module
from DeepSegV2Lib.models import *
from DeepSegV2Lib.predict import *
import DeepSegV2Lib
#from DeepSegV2Lib import *
#####################################################################

####################### add your variables here #######################
config = dict()

# define input
config["image_shape"] = (192, 224, 160) # the input to the pre-trained model
#config["image_shape"] = (160, 224, 192) # the input to the pre-trained model

# OR one variable for all MRI modalities
config["images"] = ['BraTS20_sample_case_flair.nii.gz', 'BraTS20_sample_case_t1.nii.gz', 
                     'BraTS20_sample_case_t1ce.nii.gz', 'BraTS20_sample_case_t2.nii.gz']

# model parameters
config["input_shape"] = (config["image_shape"][0], config["image_shape"][1], 
                         config["image_shape"][2], len(config["images"]))

config['model_path'] = os.path.join('DeepSegV2Lib', 'weights', 'model-238.h5')
config['tumor_type'] = "all" # "all", "whole", "core", "enhancing"

config["preprocess_dir"] = 'BraTS20_sample_case_preprocess' # directory of the pre-processed image(s)
config["predict_dir"] = 'BraTS20_sample_case_predict' # directory of the predicted segmentation

#######################################################################

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
    self.parent.contributors=["Ramy Zeineldin (Reutlingen University)"]
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

    ####################### add your connections here #######################
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
    #########################################################################

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

    ####################### add your ui connected components here #######################
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
    #####################################################################################

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

    ####################### add your ui connected components here #######################
    #self._parameterNode.SetNodeReferenceID("InputVolume", self.ui.inputSelector.currentNodeID)
    self._parameterNode.SetNodeReferenceID("InputVolume1", self.ui.FLAIRSelector.currentNodeID)
    self._parameterNode.SetNodeReferenceID("InputVolume2", self.ui.T1Selector.currentNodeID)
    self._parameterNode.SetNodeReferenceID("InputVolume3", self.ui.T1ceSelector.currentNodeID)
    self._parameterNode.SetNodeReferenceID("InputVolume4", self.ui.T2Selector.currentNodeID)
    self._parameterNode.SetNodeReferenceID("OutputVolume", self.ui.outputSelector.currentNodeID)
    #####################################################################################

    self._parameterNode.EndModify(wasModified)

  def onApplyButton(self):
    """
    Run processing when user clicks "Apply" button.
    """

    try:
      ####################### add your main code here #######################
      # Compute output
      import time
      startTime = time.time()
      logging.info('Processing started')

      inputVolume1 = self.ui.FLAIRSelector.currentNode()
      inputVolume2 = self.ui.T1Selector.currentNode()
      inputVolume3 = self.ui.T1ceSelector.currentNode()
      inputVolume4 = self.ui.T2Selector.currentNode()
      segVolumeNode = self.ui.outputSelector.currentNode()

      # get the numpy array(s)
      img1 = slicer.util.arrayFromVolume(inputVolume1)
      img2 = slicer.util.arrayFromVolume(inputVolume2)
      img3 = slicer.util.arrayFromVolume(inputVolume3)
      img4 = slicer.util.arrayFromVolume(inputVolume4)

      imgs = np.stack([img1, img2, img3, img4], axis=3)
      # fix the data structure of nrrd (x,y,z) and numpy (z,y,x)
      imgs = np.swapaxes(imgs, 0, 2)

      # preprocess image(s)
      img_preprocess = self.logic.preprocess_images(imgs)
      #print("img_preprocess:", img_preprocess.shape)
      #print("img_preprocess 0:", img_preprocess[:,:,:,0].shape)
      #img_preprocess1 = np.swapaxes(img_preprocess[:,:,:,0], 0, 2)
      #slicer.util.updateVolumeFromArray(segVolumeNode, img_preprocess1)

      # predict tumor boundaries
      # create the residual U-Net model
      trained_model = DeepSegV2Lib.models.get_model(input_shape=config["input_shape"])

      # load weights of the pre-trained model
      pretrainedURL = "https://github.com/razeineldin/Test_Data/raw/main/model-238.h5"
      modelPath = get_file(pretrainedURL.split("/")[-1], pretrainedURL,
                  file_hash="b12111e871aa04436f2e19e79d24a77c39c22d301d651be842cd711d1ac391b8",
                  hash_algorithm="sha256")
      trained_model.load_weights(modelPath)#, by_name=True)

      # predict the tumor boundries
      tumor_pred = DeepSegV2Lib.predict.predict_segmentations(trained_model, img_preprocess, 
                  tumor_type = config['tumor_type'], output_shape=(imgs.shape[0], imgs.shape[1], imgs.shape[2]))

      # casting to unsigned int and reshape
      tumor_pred = np.array(tumor_pred).astype(np.uintc)
      tumor_pred = np.swapaxes(tumor_pred, 0, 2)
      slicer.util.updateVolumeFromArray(segVolumeNode, tumor_pred)


      # fix the orientation problem
      segVolumeNode.SetOrigin(inputVolume1.GetOrigin())
      segVolumeNode.SetSpacing(inputVolume1.GetSpacing())
      ijkToRasDirections = vtk.vtkMatrix4x4()
      inputVolume1.GetIJKToRASDirectionMatrix(ijkToRasDirections)
      segVolumeNode.SetIJKToRASDirectionMatrix(ijkToRasDirections)

      # view the segmentation output in slicer
      slicer.util.setSliceViewerLayers(background=inputVolume1)
      slicer.util.setSliceViewerLayers(foreground=segVolumeNode)
      slicer.util.setSliceViewerLayers(foregroundOpacity=0.5)

      # change the tumor colo space
      displayNode = segVolumeNode.GetDisplayNode()
      displayNode.SetAndObserveColorNodeID('vtkMRMLColorTableNodeLabels') #vtkMRMLColorTableNodeRainbow

      stopTime = time.time()
      logging.info('Processing completed in {0:.2f} seconds'.format(stopTime-startTime))
      #######################################################################

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

  ####################### add your functions here #######################
  def norm_image(self, img, norm_type = "norm"):
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

  def crop_image(self, img, output_shape=np.array(config["image_shape"])):
    # manual cropping to config["image_shape"] = (160, 224, 192)
    input_shape = np.array(img.shape)
    # center the cropped image
    offset = np.array((input_shape - output_shape)/2).astype(np.int)
    offset[offset<0] = 0
    x, y, z = offset
    crop_img = img[x:x+output_shape[0], y:y+output_shape[1], z:z+output_shape[2]]

    # pad the preprocessed image
    padded_img = np.zeros(output_shape)
    x, y, z = np.array((output_shape - np.array(crop_img.shape))/2).astype(np.int)
    padded_img[x:x+crop_img.shape[0],y:y+crop_img.shape[1],z:z+crop_img.shape[2]] = crop_img

    return padded_img

  def preprocess_images(self, imgs, dim=config["input_shape"]):
    # TODO: automatic cropping using img[~np.all(img == 0, axis=1)]
    img_preprocess = np.zeros(dim)
    for i in range(dim[-1]):
      img_preprocess[:,:,:,i] = self.crop_image(imgs[:,:,:,i])
      img_preprocess[:,:,:,i] = self.norm_image(img_preprocess[:,:,:,i])

    return img_preprocess
  #######################################################################

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
    inputVolume = SampleData.downloadSample('IGSSampleDataFlair')
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
