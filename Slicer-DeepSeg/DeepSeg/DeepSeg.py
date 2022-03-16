import os
import sys
import time
import unittest
import logging
import vtk, qt, ctk, slicer

# 3D Slicer imports
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

# DeepSeg dependencies
sys.argv = ["pdm"] # important for tensorflow

try:
  import nibabel as nib
except:
  slicer.util.pip_install("nibabel") # --upgrade --force-reinstall")
  import nibabel as nib
try:
  from nilearn.image import crop_img as crop_image
except:
  slicer.util.pip_install("nilearn")
  from nilearn.image import crop_img as crop_image
try:
  import numpy as np
except:
  slicer.util.pip_install("numpy~=1.19.2")
  import numpy as np
try:
  import tensorflow as tf
except:
  try:
    if sys.version_info >= (3, 7): # Python version
      slicer.util.pip_install("tensorflow==2.5")
    else:
      slicer.util.pip_install("tensorflow==2.4")
    import tensorflow as tf
  except:
    slicer.util.restart()
    slicer.util.exit() # stop from installing other packages untill restart
try:
  import tensorflow_addons
except:
  slicer.util.pip_install("tensorflow_addons")
try:
  import skimage
except:
  try:
    slicer.util.pip_install("scikit-image")
    import skimage
  except:
    slicer.util.restart()
    slicer.util.exit() # stop from installing other packages untill restart
try:
  import h5py
  if h5py.__version__ != "2.10.0" or "3.6.0" or "3.1.0":
    slicer.util.pip_install("h5py")
except:
  slicer.util.pip_install("h5py")

# Deep learning imports
from tensorflow.keras.utils import get_file

# DeepSeg imports
import DeepSegLib

# GPU handling (TF 2.X)
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

# TODO: Add local icons to the buttons
#ICON_DIR = os.path.dirname(os.path.realpath(__file__)) + "/Resources/Icons/"

#
# DeepSeg
#

class DeepSeg(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "DeepSeg"
    self.parent.categories = ["Machine Learning", "Segmentation"]

    self.parent.dependencies = ["SegmentEditor"]
    self.parent.contributors = ["Ramy Zeineldin (Reutlingen University, Karlsruhe Institute of Technology), Pauline Weimann (Reutlingen University)"]
    self.parent.helpText = """
This modules provides a basic interface for brain tumour segmentation using deep learning-based methods
See more information in <a href="https://github.com/razeineldin/Slicer-DeepSeg">module repository</a>.
"""
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

    # TODO: Convert into .ui file
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
    self.ui.toggle3DButton.connect("clicked(bool)", self.onToggle3DButton)
    self.ui.editSegButton.connect("clicked(bool)", self.onEditSegButton)

    self.restoreDefaultsButton.connect("clicked(bool)", self.onRestoreDefaultsButton)
    self.cancelButton.connect("clicked(bool)", self.onCancelButton)
    self.applyButton.connect("clicked(bool)", self.onApplyButton)

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

    # Update node selectors and sliders
    self.ui.FLAIRSelector.setCurrentNode(self._parameterNode.GetNodeReference("InputVolume1"))
    self.ui.T1Selector.setCurrentNode(self._parameterNode.GetNodeReference("InputVolume2"))
    self.ui.T1ceSelector.setCurrentNode(self._parameterNode.GetNodeReference("InputVolume3"))
    self.ui.T2Selector.setCurrentNode(self._parameterNode.GetNodeReference("InputVolume4"))
    self.ui.outputSelector.setCurrentNode(self._parameterNode.GetNodeReference("OutputVolume"))

    # Update processing buttons once the input MRI and tumor output volume exist
    if self._parameterNode.GetNodeReference("InputVolume1") and self._parameterNode.GetNodeReference("OutputVolume"):
      self.cancelButton.toolTip = "Cancel the execution of the module"
      self.cancelButton.enabled = True

      self.applyButton.toolTip = "Compute output segmentation"
      self.applyButton.enabled = True

    else:
      self.cancelButton.toolTip = "Cancel the execution of the module"
      self.cancelButton.enabled = False

      self.applyButton.toolTip = "Select input and output volume nodes"
      self.applyButton.enabled = False

    # Update the output buttons once the model finshes
    if self._parameterNode.GetParameter("Completed") == "True":
      self.ui.toggle3DButton.toolTip = "Toggle the 3D brain tumor model"
      self.ui.toggle3DButton.enabled = True

      self.ui.editSegButton.toolTip = "Switch to manual segmentation module"
      self.ui.editSegButton.enabled = True
    else:
      self.ui.toggle3DButton.toolTip = "Apply the algorithm first before showing the 3D brain tumor model!"
      self.ui.toggle3DButton.enabled = False

      self.ui.editSegButton.toolTip = "Apply the algorithm first before manual segmentation!"
      self.ui.editSegButton.enabled = False

    # Restore defaults Button
    self.restoreDefaultsButton.toolTip = "Reset to default parameters"
    self.restoreDefaultsButton.enabled = True

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

    self._parameterNode.SetNodeReferenceID("InputVolume1", self.ui.FLAIRSelector.currentNodeID)
    self._parameterNode.SetNodeReferenceID("InputVolume2", self.ui.T1Selector.currentNodeID)
    self._parameterNode.SetNodeReferenceID("InputVolume3", self.ui.T1ceSelector.currentNodeID)
    self._parameterNode.SetNodeReferenceID("InputVolume4", self.ui.T2Selector.currentNodeID)
    self._parameterNode.SetNodeReferenceID("OutputVolume", self.ui.outputSelector.currentNodeID)

    self._parameterNode.EndModify(wasModified)

  def onModalitySelector(self):
    modality = self.ui.modalitySelector.currentIndex

    if modality == 0: # FLAIR
      self._parameterNode.SetParameter("MRIsCount", "1")
    elif modality == 1: # FLAIR, T1, T1ce, T2
      self._parameterNode.SetParameter("MRIsCount", "4")

  def onImageShapeSelector(self):
    imageShape = self.ui.imageShapeSelector.currentIndex

    if imageShape == 0: # 240,240,155
      self._parameterNode.SetParameter("ImageShape", "(240, 240, 155)")
    elif imageShape == 1: # 192,224,160
      self._parameterNode.SetParameter("ImageShape", "(192, 224, 160)")

  def onTumourTypeSelector(self):
    tumourType = self.ui.tumourTypeSelector.currentIndex

    if tumourType == 0:
      self._parameterNode.SetParameter("TumorType", "whole")
    elif tumourType == 1:
      self._parameterNode.SetParameter("TumorType", "core")
    elif tumourType == 2:
      self._parameterNode.SetParameter("TumorType", "enhancing")
    elif tumourType == 3:
      self._parameterNode.SetParameter("TumorType", "all")

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

  def hideVolumeRendering(self, volumeNode):
    # Get/create volume rendering display node
    volRenLogic = slicer.modules.volumerendering.logic()
    displayNode = volRenLogic.GetFirstVolumeRenderingDisplayNode(volumeNode)
    if not displayNode:
      displayNode = volRenLogic.CreateDefaultVolumeRenderingNodes(volumeNode)

    displayNode.SetVisibility(False)

  def show3DView(self):
    # Get the input MRI and the tumor nodes
    brainVolumeNode = self._parameterNode.GetNodeReference("InputVolume1")
    tumorVolumeNode = self._parameterNode.GetNodeReference("OutputVolume")

    # Show the nodes in the 3D view
    logging.info("Showing 3D View")
    self.showVolumeRenderingMIP(tumorVolumeNode)
    self.showTransparentRendering(brainVolumeNode, 0.3, 60.0)
    self._parameterNode.SetParameter("3DView", "on")

    # Center the 3D View on the Scene
    layoutManager = slicer.app.layoutManager()
    threeDWidget = layoutManager.threeDWidget(0)
    threeDView = threeDWidget.threeDView()
    threeDView.resetFocalPoint()

    # TODO: Turn off interpolation for tumor


  def hide3DView(self):
    # Get the input MRI and the tumor nodes
    brainVolumeNode = self._parameterNode.GetNodeReference("InputVolume1")
    tumorVolumeNode = self._parameterNode.GetNodeReference("OutputVolume")

    # Hide the nodes in the 3D view
    logging.info("Clearing 3D View")
    self.hideVolumeRendering(brainVolumeNode)
    self.hideVolumeRendering(tumorVolumeNode)
    # Trigger the 3DView parameter
    self._parameterNode.SetParameter("3DView", "off")

  def onToggle3DButton(self):
    if self._parameterNode.GetParameter("3DView") == "off":
        # Show the nodes in the 3D view
        self.show3DView()

    elif self._parameterNode.GetParameter("3DView") == "on":
        # Hide the nodes in the 3D view
        self.hide3DView()

  def onEditSegButton(self):
    logging.info("Switching to Segment Editor")

    # Switch to Segment Editor in order to edit the segments manually
    slicer.util.selectModule("SegmentEditor")

  def onRestoreDefaultsButton(self):
    logging.info("Restoring Defaults")

    #self.setParameterNode(self.logic.getParameterNode())
    self.logic.resetParameters(self._parameterNode)

    if not self._parameterNode.GetNodeReference("InputVolume1"):
      firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode") # vtkMRMLSegmentationNode

      if firstVolumeNode:
        self._parameterNode.SetNodeReferenceID("InputVolume1", firstVolumeNode.GetID())

    self._parameterNode.SetNodeReferenceID("InputVolume2", None)
    self._parameterNode.SetNodeReferenceID("InputVolume3", None)
    self._parameterNode.SetNodeReferenceID("InputVolume4", None)
    self._parameterNode.SetNodeReferenceID("OutputVolume", None)

    # Advanced segmentation parameters
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
      # Compute output
      startTime = time.time()
      logging.info("Pre-processing")
      self.currentStatusLabel.text = "Pre-processing"
      self._parameterNode.SetParameter("Status", "pre-processing")
      self.progress.setValue(0)
      self.progress.show()

      inputVolume1 = self.ui.FLAIRSelector.currentNode()
      inputVolume2 = self.ui.T1Selector.currentNode()
      inputVolume3 = self.ui.T1ceSelector.currentNode()
      inputVolume4 = self.ui.T2Selector.currentNode()
      outputVolume = self.ui.outputSelector.currentNode()

      # Model variables
      modalityCount = int(self._parameterNode.GetParameter("MRIsCount"))
      imageShape = np.asarray(self._parameterNode.GetParameter("ImageShape").strip(")(").split(", "))
      tumorType = self._parameterNode.GetParameter("TumorType")
      inputShape = (int(imageShape[0]), int(imageShape[1]), int(imageShape[2]), modalityCount)

      if modalityCount ==1: # DeepSeg
        image1 = slicer.util.arrayFromVolume(inputVolume1)
        images = image1[..., np.newaxis]

      else: # 4 modalities (nnUNet)
        # Get the numpy array(s) from the input node(s)
        image1 = slicer.util.arrayFromVolume(inputVolume1)
        image2 = slicer.util.arrayFromVolume(inputVolume2)
        image3 = slicer.util.arrayFromVolume(inputVolume3)
        image4 = slicer.util.arrayFromVolume(inputVolume4)

        images = np.stack([image1, image2, image3, image4], axis=3)

      # Fix the data structure of nrrd (x,y,z) and numpy (z,y,x)
      images = np.swapaxes(images, 0, 2)

      stopTime = time.time()
      logging.info("Loadind data completed in {0:.2f} seconds".format(stopTime-startTime))
      startTime = time.time()

      # Preprocess the MRI image(s)
      imagePreprocessed = self.logic.preprocessImages(images, dim=inputShape)

      ### Debuging ###
      #print("imagePreprocessed:", imagePreprocessed.shape)
      #print("imagePreprocessed 0:", imagePreprocessed[:,:,:,0].shape)
      #imagePreprocessed1 = np.swapaxes(imagePreprocessed[:,:,:,0], 0, 2)
      #slicer.util.updateVolumeFromArray(outputVolume, imagePreprocessed1)
      stopTime = time.time()
      logging.info("Pre-processing data completed in {0:.2f} seconds".format(stopTime-startTime))
      self.progress.setValue(100)
      startTime = time.time()

      self.currentStatusLabel.text = "Downloading"
      self._parameterNode.SetParameter("Status", "downloading")

      if modalityCount == 1:
        # Model 1: DeepSeg model
        logging.info("Getting DeepSeg Model")
        trainedModel = DeepSegLib.models.get_deepSeg(input_shape=inputShape)

        # Load weights of the pre-trained model
        # sha1sum model_DeepSeg.h5
        pretrainedURL = "https://github.com/razeineldin/Test_Data/raw/main/model_DeepSeg_21.h5"
        modelPath = get_file(pretrainedURL.split("/")[-1], pretrainedURL,
                    file_hash="941eb4b2c7da98310a95176e7adabe8f84d2e3df",
                    hash_algorithm="sha256")

      else:
        # Model 2: nnU-Net model
        logging.info("Getting nnU-Net Model")
        trainedModel = DeepSegLib.models.get_nnUNet(input_shape=inputShape)

        # Load weights of the pre-trained model
        pretrainedURL = "https://github.com/razeineldin/Test_Data/raw/main/model_nnU-Net.h5"
        modelPath = get_file(pretrainedURL.split("/")[-1], pretrainedURL,
                    file_hash="1a1990e9cfcd806231c3bd54aee62240594fee41",
                    hash_algorithm="sha256")

      output_shape=(images.shape[0], images.shape[1], images.shape[2])

      #trainedModel.summary(line_length=150)
      trainedModel.load_weights(modelPath)
      stopTime = time.time()
      logging.info("Getting pre-trained model completed in {0:.2f} seconds".format(stopTime-startTime))
      self.progress.setValue(400)

      """if self.logic.abort:
        self.progress.setValue(0)
        self.progress.hide()
        self.currentStatusLabel.text = "Idle"
        return 0"""

      startTime = time.time()

      # Predict the brain tumor boundries
      self.currentStatusLabel.text = "Predicting"
      self._parameterNode.SetParameter("Status", "predicting")
      tumorPrediction = DeepSegLib.predict.predict_segmentations(trainedModel, imagePreprocessed, 
                  tumor_type = tumorType, output_shape = output_shape)

      # Casting to unsigned int and reshape
      tumorPrediction = np.array(tumorPrediction).astype(np.uintc)
      tumorPrediction = np.swapaxes(tumorPrediction, 0, 2)
      
      stopTime = time.time()
      logging.info("Prediction completed in {0:.2f} seconds".format(stopTime-startTime))
      self.currentStatusLabel.text = "Completed"
      self._parameterNode.SetParameter("Completed", "True")
      self.progress.setValue(1000)

      startTime = time.time()

      slicer.util.updateVolumeFromArray(outputVolume, tumorPrediction)

      # Fix the orientation problem
      outputVolume.SetOrigin(inputVolume1.GetOrigin())
      outputVolume.SetSpacing(inputVolume1.GetSpacing())
      ijkToRasDirections = vtk.vtkMatrix4x4()
      inputVolume1.GetIJKToRASDirectionMatrix(ijkToRasDirections)
      outputVolume.SetIJKToRASDirectionMatrix(ijkToRasDirections)

      # View the segmentation output in slicer
      slicer.util.setSliceViewerLayers(background=inputVolume1)
      slicer.util.setSliceViewerLayers(foreground=outputVolume)
      slicer.util.setSliceViewerLayers(foregroundOpacity=0.5)

      # Change the tumor color space
      displayNode = outputVolume.GetDisplayNode()
      displayNode.SetAndObserveColorNodeID("vtkMRMLColorTableNodeLabels") #vtkMRMLColorTableNodeRainbow

      # Make sure that all labels are displayed
      displayNode.SetAutoWindowLevel(0)
      displayNode.SetAutoWindowLevel(1)

      # Show 3D segmentation
      self.show3DView()

      stopTime = time.time()
      logging.info("Visualization completed in {0:.2f} seconds".format(stopTime-startTime))
      self.progress.setValue(0)
      self.progress.hide()

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
  computation done by DeepSeg. 
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
    if not parameterNode.GetParameter("MRIsCount"):
      #images_list = '["case_flair.nii.gz", "case_t1.nii.gz", "case_t1ce.nii.gz", "case_t2.nii.gz"]'
      #images_list = ["case_flair.nii.gz"]
      parameterNode.SetParameter("MRIsCount", str(1))
    if not parameterNode.GetParameter("ImageShape"):
      parameterNode.SetParameter("ImageShape", "(192, 224, 160)") # 240, 240, 155
    if not parameterNode.GetParameter("TumorType"):
      parameterNode.SetParameter("TumorType", "whole") # all, whole, core, enhancing

    if not parameterNode.GetParameter("Completed"):
      parameterNode.SetParameter("Completed", "False")
    if not parameterNode.GetParameter("Status"):
      parameterNode.SetParameter("Status", "idle")
    if not parameterNode.GetParameter("3DView"):
      parameterNode.SetParameter("3DView", "off")

  def resetParameters(self, parameterNode):
    """
    Resest parameter node with default settings.
    """
    parameterNode.SetParameter("MRIsCount", str(1))
    parameterNode.SetParameter("ImageShape", "(192, 224, 160)") # 240, 240, 155
    parameterNode.SetParameter("TumorType", "whole") # all, whole, core, enhancing
    parameterNode.SetParameter("Completed", "False")
    parameterNode.SetParameter("Status", "idle")
    parameterNode.SetParameter("3DView", "off")

  def normalizeImage(self, img, norm_type = "norm"):
    if norm_type == "standard_norm": # standarization, same dataset
        imageMean = img.mean()
        imageStandadDeviation = img.std()
        imageStandadDeviation = 1 if img.std()==0 else img.std()
        img = (img - imageMean) / imageStandadDeviation
    elif norm_type == "norm": # different datasets
        img = (img - np.min(img))/(np.ptp(img)) # (np.max(img) - np.min(img))
    elif norm_type == "norm_slow": # different datasets
#         img = (img - np.min(img))/(np.max(img) - np.min(img))
        imagePeak = 1 if np.ptp(img)== 0 else np.ptp(img) 
        img = (img - np.min(img))/imagePeak

    return img

  def cropImage(self, img, output_shape=np.array((192, 224, 160))):
    # Manual cropping to (160, 224, 192)
    input_shape = np.array(img.shape)
    # Center the cropped image
    offset = np.array((input_shape - output_shape)/2).astype(np.int)
    offset[offset<0] = 0
    x, y, z = offset
    croppedImage = img[x:x+output_shape[0], y:y+output_shape[1], z:z+output_shape[2]]

    # Pad the preprocessed image
    paddedImage = np.zeros(output_shape)
    x, y, z = np.array((output_shape - np.array(croppedImage.shape))/2).astype(np.int)
    paddedImage[x:x+croppedImage.shape[0],y:y+croppedImage.shape[1],z:z+croppedImage.shape[2]] = croppedImage

    return paddedImage

  def preprocessImages(self, images, dim):
    # TODO: Automatic cropping using img[~np.all(img == 0, axis=1)]
    imagePreprocessed = np.zeros(dim)
    #print("Shape imagePreprocessed", imagePreprocessed.shape)
    for i in range(dim[-1]):
      imagePreprocessed[:,:,:,i] = self.cropImage(images[:,:,:,i])
      imagePreprocessed[:,:,:,i] = self.normalizeImage(imagePreprocessed[:,:,:,i])

    return imagePreprocessed

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
