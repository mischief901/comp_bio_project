import xml.etree.ElementTree as ET
from copy import deepcopy
from os import listdir
import numpy as np
from decimal import *

class Spectra :
  """The Spectra class holds the information and functions for loading and
  parsing a MS-MS Spectra from an XML datasheet. 

  Usage: 
  spectra = Spectra(file_name)
  """

  def __init__(self, file_name) :
    tree = ET.parse(file_name).getroot()
    self.file_name = file_name
    try :
      self._parse_tree(tree)
    except :
      return None
    
  def _parse_tree(self, tree) :
    self.voltage = None
    voltage = tree.find('collision-energy-voltage')
    if voltage.text :
      self.voltage = int(voltage.text)
    
    polarity = tree.find('ionization-mode')
    #print(polarity.text)
    self.polarity = polarity.text
    
    database_id = tree.find('database-id')
    #print(database_id.text)
    self.database_id = database_id.text
    
    predicted = tree.find('predicted')
    if predicted.text == "false" :
      self.predicted = False
    else :
      self.predicted = True
    peaks = []
    raw_peaks = tree.find('ms-ms-peaks').findall('ms-ms-peak')
    for peak in raw_peaks :
      peak_id = peak.find('id').text
      peak_mass_charge = peak.find('mass-charge').text
      peak_intensity = peak.find('intensity').text
      #print(peak_id, peak_mass_charge, peak_intensity)
      peaks.append((float(peak_mass_charge), float(peak_intensity)))
    self.peaks = peaks

  def __str__(self) :
    string = """{}
    Predicted: {}
    Ionization Mode: {}
    Voltage (V): {}
    Peaks:
    m/z\t\tintensity
    ------------------------""".format(self.database_id, self.predicted, self.polarity,
                                       self.voltage)
    
    for mass_charge, intensity in self.peaks :
      string = """{}
    {:.3f}\t\t{:.3f}""".format(string, mass_charge, intensity)
    return string + "\n"
    
    
def get_all(folder, filter_funs = []) :
  """Returns a dictionary of all parsable spectra. Key is the HMDB ID of the 
  compound and the value is a list of all corresponding objects. 
  A list of filter functions can be provided in order to only return spectra 
  that pass all the filters.
  """

  def apply_funs(x, funs) :
    """Applies the filter functions."""
    res = True
    for f in funs :
      res = f(x)
      if not res :
        break
    return res
  
  final = {}
  files = listdir(folder)
  print("Loading Spectras")
  for f in files :
    try :
      spectra = Spectra(folder + "/" + f)
      print(".", end="")
    except:
      continue
    if spectra == None :
      continue
    if not apply_funs(spectra, filter_funs) :
      continue
    pot_spectra = final.get(spectra.database_id, None)
    if not pot_spectra :
      final[spectra.database_id] = [deepcopy(spectra)]
    else :
      pot_spectra.append(deepcopy(spectra))
  return final


def voltage_filter(spectra) :
  if not spectra.voltage == None :
    return True
  return False

def experimental_filter(spectra) :
  if spectra.predicted :
    return False
  return True

def voltage_level(spectra, volts) :
  if spectra.voltage == volts :
    return True
  return False

def voltage_range(spectra, volt_low, volt_high) :
  if volt_low <= spectra.voltage and spectra.voltage <= volt_high :
    return True
  return False


class Embedded_Spectra :
  """The Embedded_Spectra Class is used to create a dictionary to embed the mass_peaks
  and intensities into a fixed array size where the index of the array corresponds to
  the mass_peak and the value is the intensity. This is common to vectorize words for
  machine learning. Words2Vec is a common implementation. A class variable contains
  a dictionary that is expanded for each converted spectra and there are corresponding
  functions to unembed the peaks and intensities of the spectra.

  Peaks are rounded to the nearest tenth to reduce the size of the output array.
  When more than one peak is present in the same array position, the intensities are
  summed to represent one peak instead. There is a loss of precision here, but not a
  large amount.

  translator = Embedded_Spectra(peaks)
  array = np.zeros(peaks.size)
  ## New array to hold the results of each translation.
  spectras = [list(zip(*s)) for s in list(zip(spectra, intensities))]
  for i, spectra in enumerate(spectras) :
    array[i]=translator.to_array(spectra)
  """
  
  translation_dict = {}
  pre_translation = []
  post_translation_dict = {}

  def __init__(self, peak_array) :
    for peaks in peak_array :
      for peak in peaks :
        peak = round(peak)
        #peak -= (peak % 2)
        #peak = round(Decimal(peak), 1)
        if peak in self.pre_translation :
          continue
        self.pre_translation.append(peak)

  def set_translation(self) :
    
    self.pre_translation.sort()
    for i, item in enumerate(self.pre_translation) :
      self.translation_dict[item] = i
      self.post_translation_dict[i] = item

  def get_size(self) :
    return len(self.pre_translation)

  def to_array(self, spectra, with_intensity) :
    array = np.zeros(len(self.translation_dict))
    if with_intensity :
      for peak, intensity in spectra :
        peak = round(peak)
        #peak -= (peak % 2)
        #peak = round(Decimal(peak), 1)
        index = self.translation_dict[peak]
        array[index] += intensity
      return array
    else :
      for peak in spectra :
        peak = round(peak)
        #peak -= (peak % 2)
        #peak = round(Decimal(peak), 1)
        index = self.translation_dict[peak]
        array[index] = 1
      return array

  def from_array(self, array1, array2) :
    spectra = []
    for index, intensity in enumerate(array2) :
      if intensity > 0 and array1[index] > 0.5 :
        peak = self.post_translation_dict[index]
        spectra.append((peak, intensity))
    return spectra
      
      
