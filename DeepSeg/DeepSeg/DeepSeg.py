import os
import unittest
import logging
import vtk, qt, ctk, slicer

from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

####################### add your imports here #######################

try:
  import matplotlib.pyplot as plt
except:
  slicer.util.pip_install("matplotlib")
  import matplotlib.pyplot as plt
# TODO: add other imports
try:
  import numpy as np
except:
  slicer.util.pip_install("numpy~=1.19.2")
  import numpy as np
try:
  import nibabel as nib
except:
  slicer.util.pip_install("nibabel")
  import nibabel as nib
try:
  from nilearn.image import crop_img as crop_image
except:
  slicer.util.pip_install("nilearn")
  from nilearn.image import crop_img as crop_image

import sys
import time
sys.argv = ["pdm"]
import tensorflow.python

try:
  import tensorflow as tf
except:
  slicer.util.pip_install("tensorflow")
  import tensorflow as tf

import numpy as np
import nibabel as nib

import sys
sys.argv = ["pdm"]
import tensorflow.python
import tensorflow as tf

# utlity functions imports
import matplotlib.pyplot as plt
from tensorflow.keras.utils import get_file

#from nilearn.image import crop_img as crop_image

# import functions from models module
from DeepSegLib.models import *
#from DeepSegLib.models_nnUNet import *
from DeepSegLib.predict import *
import DeepSegLib
#from DeepSegLib import *

# Tensorflow 2.XX\n",
if float(tf.__version__[:3]) >= 2.0:
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"] = "0" # "0,1"

  gpus = tf.config.experimental.list_physical_devices("GPU")
  #print("Num GPUs Available:", len(gpus))
  if gpus:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
      logical_gpus = tf.config.experimental.list_logical_devices("GPU")
      print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs\n")

ICON_DIR = os.path.dirname(os.path.realpath(__file__)) + "/Resources/Icons/"

#####################################################################


#
# DeepSeg
#

class DeepSeg(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "DeepSeg"  # TODO: make this more human readable by adding spaces
    self.parent.categories=["Examples"]
    self.parent.dependencies = []  # TODO: add here list of module names that this module requires
    self.parent.contributors=["Ramy Zeineldin, Pauline Weimann (Reutlingen University)"]
    # TODO: update with short description of the module and a link to online module documentation
    self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#DeepSeg">module documentation</a>.
"""
    # TODO: replace with organization, grant and thanks
    self.parent.acknowledgementText = """
This module has been done within the Research Group Computer Assisted Medicine (CaMed), Reutlingen University and the Health Robotics and Automation (HERA), Institute for Anthropomatics and Robotics (IAR), Karlsruhe Institute of Technology (KIT), Germany. The authors acknowledge support by the state of Baden-Württemberg through bwHPC. This work is partialy funded by the German Academic Exchange Service (DAAD) under Scholarship No. 91705803.

"""

#
# DeepSegWidget
#

class DeepSegWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
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

    self.modelParameters = None

  def setup(self):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.setup(self)

    # Load widget from .ui file (created by Qt Designer).
    # Additional widgets can be instantiated manually and added to self.layout.
    uiWidget = slicer.util.loadUI(self.resourcePath("UI/DeepSeg.ui"))
    self.layout.addWidget(uiWidget)
    self.ui = slicer.util.childWidgetVariables(uiWidget)

    # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget"s
    # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget"s.
    # "setMRMLScene(vtkMRMLScene*)" slot.
    uiWidget.setMRMLScene(slicer.mrmlScene)

    # Create logic class. Logic implements all computations that should be possible to run
    # in batch mode, without a graphical user interface.
    self.logic = DeepSegLogic()

    # Status and Progress
    statusLabel = qt.QLabel("Status: ")
    self.currentStatusLabel = qt.QLabel("Idle")
    hlayout = qt.QHBoxLayout()
    hlayout.addStretch(1)
    hlayout.addWidget(statusLabel)
    hlayout.addWidget(self.currentStatusLabel)
    self.layout.addLayout(hlayout)

    self.progress = qt.QProgressBar()
    self.progress.setRange(0, 1000)
    self.progress.setValue(0)
    self.layout.addWidget(self.progress)
    self.progress.hide()

    # Cancel/Restore Defaults/Apply row
    self.restoreDefaultsButton = qt.QPushButton("Restore Defaults")
    self.restoreDefaultsButton.toolTip = "Restore the default parameters."
    self.restoreDefaultsButton.enabled = True

    self.cancelButton = qt.QPushButton("Cancel")
    self.cancelButton.toolTip = "Abort the algorithm."
    self.cancelButton.enabled = False

    self.applyButton = qt.QPushButton("Apply")
    self.applyButton.toolTip = "Run the algorithm."
    self.applyButton.enabled = False

    hlayout = qt.QHBoxLayout()
    hlayout.addWidget(self.restoreDefaultsButton)
    hlayout.addStretch(1)
    hlayout.addWidget(self.cancelButton)
    hlayout.addWidget(self.applyButton)
    self.layout.addLayout(hlayout)

    self.onBackgroundSelector() # Change 3D View Background

    # Connections

    # These connections ensure that we update parameter node when scene is closed
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

    ####################### add your connections here #######################
    # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
    # (in the selected parameter node).

    self.ui.FLAIRSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    self.ui.T1Selector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    self.ui.T1ceSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    self.ui.T2Selector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)    
    self.ui.outputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)

    # Advanced parameters
    self.ui.modalitySelector.currentIndexChanged.connect(self.onModalitySelector)
    self.ui.imageShapeSelector.currentIndexChanged.connect(self.onImageShapeSelector)
    self.ui.tumourTypeSelector.currentIndexChanged.connect(self.onTumourTypeSelector)
    self.ui.backgroundSelector.currentIndexChanged.connect(self.onBackgroundSelector)

    # Buttons
    self.ui.show3DButton.connect("clicked(bool)", self.onShow3DButton)
    self.ui.editSegButton.connect("clicked(bool)", self.onEditSegButton)

    self.restoreDefaultsButton.connect("clicked(bool)", self.onRestoreDefaultsButton)
    self.cancelButton.connect("clicked(bool)", self.onCancelButton)
    self.applyButton.connect("clicked(bool)", self.onApplyButton)

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
    # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
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
      firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode") # vtkMRMLSegmentationNode

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
      # show 3D Button
      self.ui.show3DButton.toolTip = "Create 3D Model"
      self.ui.show3DButton.enabled = True
      # edit seg Button
      self.ui.editSegButton.toolTip = "Switch to Segment Editor"
      self.ui.editSegButton.enabled = True
      # Cancel Button
      self.cancelButton.toolTip = "Cancel the execution of the module"
      self.cancelButton.enabled = True
      # apply Button
      self.applyButton.toolTip = "Compute output segmentation"
      self.applyButton.enabled = True

    else:
      # show 3D Button
      self.ui.show3DButton.toolTip = "Apply the algorithm first!"
      self.ui.show3DButton.enabled = False
      # edit seg Button
      self.ui.editSegButton.toolTip = "Apply the algorithm first!"
      self.ui.editSegButton.enabled = False
      # Cancel Button
      self.cancelButton.toolTip = "Cancel the execution of the module"
      self.cancelButton.enabled = False
      # apply Button
      self.applyButton.toolTip = "Select input and output volume nodes"
      self.applyButton.enabled = False

    # restore defaults Button
    self.restoreDefaultsButton.toolTip = "Reset parameters to default"
    self.restoreDefaultsButton.enabled = True

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
    self._parameterNode.SetNodeReferenceID("InputVolume1", self.ui.FLAIRSelector.currentNodeID)
    self._parameterNode.SetNodeReferenceID("InputVolume2", self.ui.T1Selector.currentNodeID)
    self._parameterNode.SetNodeReferenceID("InputVolume3", self.ui.T1ceSelector.currentNodeID)
    self._parameterNode.SetNodeReferenceID("InputVolume4", self.ui.T2Selector.currentNodeID)
    self._parameterNode.SetNodeReferenceID("OutputVolume", self.ui.outputSelector.currentNodeID)

    #####################################################################################

    self._parameterNode.EndModify(wasModified)

  def onModalitySelector(self):
    modality = self.ui.modalitySelector.currentIndex
    #print("modality:", modality)

    if modality == 0: # FLAIR
      self._parameterNode.SetParameter("images_num", "1")
    elif modality == 1: # FLAIR, T1, T1ce, T2
      self._parameterNode.SetParameter("images_num", "4")

  def onImageShapeSelector(self):
    imageShape = self.ui.imageShapeSelector.currentIndex

    if imageShape == 0: # 240,240,155
      self._parameterNode.SetParameter("image_shape", "(240, 240, 155)")
    elif imageShape == 1: # 192,224,160
      self._parameterNode.SetParameter("image_shape", "(192, 224, 160)")

  def onTumourTypeSelector(self):
    tumourType = self.ui.tumourTypeSelector.currentIndex

    if tumourType == 0:
      self._parameterNode.SetParameter("tumor_type", "whole")
    elif tumourType == 1:
      self._parameterNode.SetParameter("tumor_type", "core")
    elif tumourType == 2:
      self._parameterNode.SetParameter("tumor_type", "enhancing")
    elif tumourType == 3:
      self._parameterNode.SetParameter("tumor_type", "all")

  def onBackgroundSelector(self):
    viewNode = slicer.app.layoutManager().threeDWidget(0).mrmlViewNode()
    backgroundColor = self.ui.backgroundSelector.currentIndex

    if backgroundColor == 0: # black
      viewNode.SetBackgroundColor(0, 0, 0)
      viewNode.SetBackgroundColor2(0, 0, 0)
    elif backgroundColor == 1: # white
      viewNode.SetBackgroundColor(1, 1, 1)
      viewNode.SetBackgroundColor2(1, 1, 1)
    elif backgroundColor == 2: # light blue
      viewNode.SetBackgroundColor(140. / 255, 140. / 255, 165. / 255) # RGB
      viewNode.SetBackgroundColor2(85. / 255, 85. / 255, 140. / 255)


  # TODO: Show 3D
  def showVolumeRenderingMIP(self, volumeNode, useSliceViewColors=True):
    """Render volume using maximum intensity projection
    :param useSliceViewColors: use the same colors as in slice views.
    """
    # Get/create volume rendering display node
    volRenLogic = slicer.modules.volumerendering.logic()
    displayNode = volRenLogic.GetFirstVolumeRenderingDisplayNode(volumeNode)
    if not displayNode:
      displayNode = volRenLogic.CreateDefaultVolumeRenderingNodes(volumeNode)
    # Choose MIP volume rendering preset
    if useSliceViewColors:
      volRenLogic.CopyDisplayToVolumeRenderingDisplayNode(displayNode)
    else:
      displayNode.GetVolumePropertyNode().Copy(volRenLogic.GetPresetByName("MR-MIP"))
      #displayNode.GetVolumePropertyNode().Copy(volRenLogic.GetPresetByName("MR-Default")) #"MR-MIP"

    # Switch views to MIP mode
    #for viewNode in slicer.util.getNodesByClass("vtkMRMLViewNode"):
    #  viewNode.SetRaycastTechnique(slicer.vtkMRMLViewNode.MaximumIntensityProjection)
    # Show volume rendering
    displayNode.SetVisibility(True)

  def showTransparentRendering(self, volumeNode, maxOpacity=0.2, gradientThreshold=30.0):
    """Make constant regions transparent and the entire volume somewhat transparent
    :param maxOpacity: lower value makes the volume more transparent overall
      (value is between 0.0 and 1.0)
    :param gradientThreshold: regions that has gradient value below this threshold will be made transparent
      (minimum value is 0.0, higher values make more tissues transparent, starting with soft tissues)
    """
    # Get/create volume rendering display node
    volRenLogic = slicer.modules.volumerendering.logic()
    displayNode = volRenLogic.GetFirstVolumeRenderingDisplayNode(volumeNode)
    if not displayNode:
      displayNode = volRenLogic.CreateDefaultVolumeRenderingNodes(volumeNode)
    # Set up gradient vs opacity transfer function
    gradientOpacityTransferFunction = displayNode.GetVolumePropertyNode().GetVolumeProperty().GetGradientOpacity()
    gradientOpacityTransferFunction.RemoveAllPoints()
    gradientOpacityTransferFunction.AddPoint(0, 0.0)
    gradientOpacityTransferFunction.AddPoint(gradientThreshold-1, 0.0)
    gradientOpacityTransferFunction.AddPoint(gradientThreshold+1, maxOpacity)
    # Show volume rendering
    displayNode.SetVisibility(True)
  
  def onShow3DButton(self):
    """ labelmapVolumeNode = self._parameterNode.GetNodeReference("InputVolume1")
    seg = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
    slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelmapVolumeNode, seg)
    seg.CreateClosedSurfaceRepresentation()
    slicer.mrmlScene.RemoveNode(labelmapVolumeNode)"""

    brainVolumeNode = self._parameterNode.GetNodeReference("InputVolume1")
    tumorVolumeNode = self._parameterNode.GetNodeReference("OutputVolume")

    self.showVolumeRenderingMIP(tumorVolumeNode)
    #self.showTransparentRendering(tumorVolumeNode, 0.2, 30.0)
    self.showTransparentRendering(brainVolumeNode, 0.3, 60.0)
    #self.showVolumeRenderingMIP(brainVolumeNode, useSliceViewColors=False)

    # Center the 3D View on the Scene
    layoutManager = slicer.app.layoutManager()
    threeDWidget = layoutManager.threeDWidget(0)
    threeDView = threeDWidget.threeDView()
    threeDView.resetFocalPoint()

  def onEditSegButton(self):
    logging.info("Switching to Segment Editor")

    # switch to Segment Editor in order to edit the segments manually
    slicer.util.selectModule("SegmentEditor")

  def onRestoreDefaultsButton(self):
    logging.info("Restoring Defaults")

    if not self._parameterNode.GetNodeReference("InputVolume1"):
      firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode") # vtkMRMLSegmentationNode

      if firstVolumeNode:
        self._parameterNode.SetNodeReferenceID("InputVolume1", firstVolumeNode.GetID())

    self._parameterNode.SetNodeReferenceID("InputVolume2", None)
    self._parameterNode.SetNodeReferenceID("InputVolume3", None)
    self._parameterNode.SetNodeReferenceID("InputVolume4", None)
    self._parameterNode.SetNodeReferenceID("OutputVolume", None)

    # Advanced parameters
    self.ui.modalitySelector.setCurrentIndex(0)
    self.ui.imageShapeSelector.setCurrentIndex(0)
    self.ui.tumourTypeSelector.setCurrentIndex(0)
    self.ui.backgroundSelector.setCurrentIndex(0)

    slicer.util.resetSliceViews() # Reset field of view to show background volume maximized
    self.currentStatusLabel.text = "Idle"

  # TODO: Cancel
  def onCancelButton(self):
    self.currentStatusLabel.text = "Aborting"
    if self.logic:
      self.logic.abort = True

  def onApplyButton(self):
    """
    Run processing when user clicks "Apply" button.
    """

    try:
      ####################### add your main code here #######################
      # Compute output
      startTime = time.time()
      logging.info("Pre-processing")
      self.currentStatusLabel.text = "Pre-processing"
      self.progress.setValue(0)
      self.progress.show()

      inputVolume1 = self.ui.FLAIRSelector.currentNode()
      inputVolume2 = self.ui.T1Selector.currentNode()
      inputVolume3 = self.ui.T1ceSelector.currentNode()
      inputVolume4 = self.ui.T2Selector.currentNode()
      segVolumeNode = self.ui.outputSelector.currentNode()

      # model variables
      modalityNum = int(self._parameterNode.GetParameter("images_num"))
      imageShape = np.asarray(self._parameterNode.GetParameter("image_shape").strip(")(").split(", "))
      tumorType = self._parameterNode.GetParameter("tumor_type")
      inputShape = (int(imageShape[0]), int(imageShape[1]), int(imageShape[2]), modalityNum)

      if modalityNum ==1: # DeepSeg
        img1 = slicer.util.arrayFromVolume(inputVolume1)
        imgs = img1[..., np.newaxis]

      else: # 4 modalities (nnUNet)
        # get the numpy array(s)
        img1 = slicer.util.arrayFromVolume(inputVolume1)
        img2 = slicer.util.arrayFromVolume(inputVolume2)
        img3 = slicer.util.arrayFromVolume(inputVolume3)
        img4 = slicer.util.arrayFromVolume(inputVolume4)

        imgs = np.stack([img1, img2, img3, img4], axis=3)

      # fix the data structure of nrrd (x,y,z) and numpy (z,y,x)
      imgs = np.swapaxes(imgs, 0, 2)

      stopTime = time.time()
      logging.info("Loadind data completed in {0:.2f} seconds".format(stopTime-startTime))
      startTime = time.time()

      # preprocess image(s)
      img_preprocess = self.logic.preprocess_images(imgs, dim=inputShape)

      ### debuging ###
      #print("img_preprocess:", img_preprocess.shape)
      #print("img_preprocess 0:", img_preprocess[:,:,:,0].shape)
      #img_preprocess1 = np.swapaxes(img_preprocess[:,:,:,0], 0, 2)
      #slicer.util.updateVolumeFromArray(segVolumeNode, img_preprocess1)
      stopTime = time.time()
      logging.info("Pre-processing data completed in {0:.2f} seconds".format(stopTime-startTime))
      self.progress.setValue(100)
      startTime = time.time()

      # predict tumor boundaries
      self.currentStatusLabel.text = "Downloading pre-trained model"


      if modalityNum == 1: # DeepSeg
        # Model 1: DeepSeg model
        logging.info("Getting DeepSeg Model")
        trained_model = DeepSegLib.models.get_deepSeg(input_shape=inputShape)

        # load weights of the pre-trained model
        # sha1sum model_DeepSeg.h5
        pretrainedURL = "https://github.com/razeineldin/Test_Data/raw/main/model_DeepSeg.h5"
        modelPath = get_file(pretrainedURL.split("/")[-1], pretrainedURL,
                    file_hash="88d0a665a6faa08140c70f9bec915fc53ec39687",
                    hash_algorithm="sha256")

        #output_shape=(imgs.shape[0], imgs.shape[1], imgs.shape[2])

      else: # nnUNet
        # Model 2: nnU-Net model
        logging.info("Getting nnU-Net Model")
        trained_model = DeepSegLib.models.get_nnUNet(input_shape=inputShape)

        # load weights of the pre-trained model
        pretrainedURL = "https://github.com/razeineldin/Test_Data/raw/main/model_nnU-Net.h5"
        modelPath = get_file(pretrainedURL.split("/")[-1], pretrainedURL,
                    file_hash="1a1990e9cfcd806231c3bd54aee62240594fee41",
                    hash_algorithm="sha256")

      output_shape=(imgs.shape[0], imgs.shape[1], imgs.shape[2])

      #trained_model.summary(line_length=150)
      trained_model.load_weights(modelPath) #, by_name=True) 
      stopTime = time.time()
      logging.info("Getting pre-trained model completed in {0:.2f} seconds".format(stopTime-startTime))
      self.progress.setValue(400)

      """if self.logic.abort:
        self.progress.setValue(0)
        self.progress.hide()
        self.currentStatusLabel.text = "Idle"
        return 0"""

      startTime = time.time()

      # predict the tumor boundries
      self.currentStatusLabel.text = "Predicting tumor segmentation"
      tumor_pred = DeepSegLib.predict.predict_segmentations(trained_model, img_preprocess, 
                  tumor_type = tumorType, output_shape = output_shape)

      # casting to unsigned int and reshape
      tumor_pred = np.array(tumor_pred).astype(np.uintc)
      tumor_pred = np.swapaxes(tumor_pred, 0, 2)
      
      stopTime = time.time()
      logging.info("Prediction completed in {0:.2f} seconds".format(stopTime-startTime))
      self.currentStatusLabel.text = "Completed"
      self.progress.setValue(1000)

      startTime = time.time()

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

      # change the tumor color space
      displayNode = segVolumeNode.GetDisplayNode()
      displayNode.SetAndObserveColorNodeID("vtkMRMLColorTableNodeLabels") #vtkMRMLColorTableNodeRainbow

      # show 3D segmentation
      self.onShow3DButton()

      stopTime = time.time()
      logging.info("Visualization completed in {0:.2f} seconds".format(stopTime-startTime))
      self.progress.setValue(0)
      self.progress.hide()
      #######################################################################

    except Exception as e:
      self.currentStatusLabel.text = "Exception"
      slicer.util.errorDisplay("Failed to compute results: "+str(e))
      import traceback
      traceback.print_exc()

#
# DeepSegLogic
#

class DeepSegLogic(ScriptedLoadableModuleLogic):
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
    self.abort = False

  def setDefaultParameters(self, parameterNode):
    """
    Initialize parameter node with default settings.
    """
    if not parameterNode.GetParameter("images_num"):
      #images_list = '["case_flair.nii.gz", "case_t1.nii.gz", "case_t1ce.nii.gz", "case_t2.nii.gz"]'
      #images_list = ["case_flair.nii.gz"]
      parameterNode.SetParameter("images_num", str(1))
    if not parameterNode.GetParameter("image_shape"):
      parameterNode.SetParameter("image_shape", "(192, 224, 160)") # 240, 240, 155
    if not parameterNode.GetParameter("tumor_type"):
      parameterNode.SetParameter("tumor_type", "whole") # all, whole, core, enhancing

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

  def crop_image(self, img, output_shape=np.array((192, 224, 160))):
    # manual cropping to (160, 224, 192)
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

  def preprocess_images(self, imgs, dim):
    # TODO: automatic cropping using img[~np.all(img == 0, axis=1)]
    img_preprocess = np.zeros(dim)
    print("Shape img_preprocess", img_preprocess.shape)
    for i in range(dim[-1]):
      img_preprocess[:,:,:,i] = self.crop_image(imgs[:,:,:,i])
      img_preprocess[:,:,:,i] = self.norm_image(img_preprocess[:,:,:,i])

    return img_preprocess
  #######################################################################

#
# DeepSegTest
#

class DeepSegTest(ScriptedLoadableModuleTest):
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
    self.test_DeepSeg1()

  def test_DeepSeg1(self):
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
    inputVolume = SampleData.downloadSample("IGSSampleDataFlair")
    self.delayDisplay("Loaded test data set")

    inputScalarRange = inputVolume.GetImageData().GetScalarRange()
    self.assertEqual(inputScalarRange[0], 0)
    self.assertEqual(inputScalarRange[1], 695)

    outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
    threshold = 100

    # Test the module logic

    logic = DeepSegLogic()

    # Test algorithm with threshold
    logic.process(inputVolume, outputVolume, threshold, True)
    outputScalarRange = outputVolume.GetImageData().GetScalarRange()
    self.assertEqual(outputScalarRange[0], inputScalarRange[0])
    self.assertEqual(outputScalarRange[1], threshold)

    self.delayDisplay("Test passed")
