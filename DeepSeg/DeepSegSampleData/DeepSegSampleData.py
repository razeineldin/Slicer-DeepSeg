import os
import unittest
import logging
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

#
# DeepSegSampleData
#

class DeepSegSampleData(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "DeepSegSampleData"  # TODO: make this more human readable by adding spaces
    self.parent.categories = ["Examples"]  # TODO: set categories (folders where the module shows up in the module selector)
    self.parent.dependencies = ["SampleData"]  # TODO: add here list of module names that this module requires
    self.parent.contributors = ["Ramy Zeineldin (Reutlingen University)"]  # TODO: replace with "Firstname Lastname (Organization)"
    # TODO: update with short description of the module and a link to online module documentation
    self.parent.helpText = """This module adds sample MRI volumes and model weights to SampleData module.
See more information in <a href="https://github.com/organization/projectname#DeepSegSampleData">module documentation</a>.
"""
    # TODO: replace with organization, grant and thanks
    self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
"""

    # don't show this module - additional data will be shown in SampleData module
    parent.hidden = True

    # Add data sets to Sample Data module.
    import SampleData
    iconsPath = os.path.join(os.path.dirname(__file__), 'Resources/Icons')

    SampleData.SampleDataLogic.registerCustomSampleDataSource(
      category='DeepSegSampleData',
      sampleName='DeepSeg MRI FLAIR',
      thumbnailFileName=os.path.join(iconsPath, 'DeepSegSampleDataFLAIR.png'),
      uris="https://github.com/razeineldin/Test_Data/raw/main/sample_case_flair.nii.gz",
      fileNames='DeepSegSampleDataFLAIR.nii.gz',
      checksums = 'SHA256:ed7f08979b1a1e8208f39ae9ac0b968ce7ee9c61464ddbbe5f7c77713ff06485',
      nodeNames='DeepSegSampleDataFLAIR'
    )
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
      category='DeepSegSampleData',
      sampleName='DeepSeg MRI T1',
      thumbnailFileName=os.path.join(iconsPath, 'DeepSegSampleDataT1.png'),
      uris="https://github.com/razeineldin/Test_Data/raw/main/sample_case_t1.nii.gz",
      fileNames='DeepSegSampleDataT1.nii.gz',
      checksums = 'SHA256:047920e2a9748fba2db4a9d059c6eb7b260ab11e61afd22b43fb92f36ca904ac',
      nodeNames='DeepSegSampleDataT1'
    )
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
      category='DeepSegSampleData',
      sampleName='DeepSeg MRI T1ce',
      thumbnailFileName=os.path.join(iconsPath, 'DeepSegSampleDataT1ce.png'),
      uris="https://github.com/razeineldin/Test_Data/raw/main/sample_case_t1ce.nii.gz",
      fileNames='DeepSegSampleDataT1ce.nii.gz',
      checksums = 'SHA256:e341143e42aa951f2ec38311ca01ef4e1bde1b9e38a666b2fd0bbb8ee0331f93',
      nodeNames='DeepSegSampleDataT1ce'
    )
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
      category='DeepSegSampleData',
      sampleName='DeepSeg MRI T2',
      thumbnailFileName=os.path.join(iconsPath, 'DeepSegSampleDataT2.png'),
      uris="https://github.com/razeineldin/Test_Data/raw/main/sample_case_t2.nii.gz",
      fileNames='DeepSegSampleDataT2.nii.gz',
      checksums = 'SHA256:feda26200c9667c5250f461cede46426b93a6d1973373969250a73df14e037b1',
      nodeNames='DeepSegSampleDataT2'
    )
