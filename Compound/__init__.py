from rdkit import Chem
from rdkit.Chem.rdmolfiles import SDMolSupplier
from rdkit.Chem.rdchem import Mol
from rdkit.Chem import AllChem
from rdkit import DataStructs

from copy import deepcopy

class Reader(SDMolSupplier) :
  """The Reader class is a wrapper class for the SDMolSupplier class in rdkit.
  This allows for additional fields and methods to be added if necessary in the 
  future. Currently next and reset functionality added since it does not seem to 
  work per the SDMolSupplier docs.
  
  Usage: 
  x = Reader(file_name)
  mol1 = x.next()
  mol2 = x.next()
  x.reset()
  mol1 == x.next()
  
  Note: The supplied file_name is saved as a class variable so only one sdf file
  may be open at a time.
  """
  file_name = None
  
  def __init__(self, file_name = None) :
    if file_name == None and self.file_name == None :
      return None
    self.file_name = file_name
    self.index = 0
    SDMolSupplier.__init__(self, file_name)

  def next(self) :
    try :
      mol = self[deepcopy(self.index)]
      self.index += 1
      return mol
    except IndexError :
      raise IndexError

  def reset(self) :
    self.index = 0
    super().reset()

      
class Compound :
  """The Compound class imports one entry from the sdf file specified in the
  supplied Reader class. The entry is then filtered so only pertinent information
  is retained.

  Usage:
  sdf = Reader(file_name)
  mol1 = Compound(sdf)
  mol2 = Compound(sdf)
  print(mol1 == mol2)
  # prints False
  """

  def __init__(self, reader) :
    mol = None
    while mol == None :
      ## Only grab entries that do not throw an error.
      try :
        mol = reader.next()
      except KeyError :
        return None
      
    mol_dict = mol.GetPropsAsDict()
    try :
      self.hmdb_id = mol_dict['HMDB_ID']
      self.mol_weight = mol_dict['MOLECULAR_WEIGHT']
      self.formula = mol_dict['FORMULA']
      self.smiles = mol_dict['SMILES']
      self.mol = Chem.MolFromSmiles(self.smiles)
    except KeyError :
      return None
  
  def to_bitvector(self, fingerprint_type, options = {}) :
    """Creates an explicit bitvector of the given fingerprint_type.
    The only valid fingerprint_type currently is 'morgan'.
    Options is a dict of key, values. The options for morgan are only 'radius'
    Default value is 2.
    """

    if fingerprint_type == 'morgan' :
      self.radius = options.get('radius', 2)
      self.bitvect = AllChem.GetMorganFingerprintAsBitVect(self.mol, self.radius)
      return self.bitvect
    else :
      print("Only 'morgan' fingerprint can be performed at the moment.")
      return None

  def bitvect_as_np_array(self) :
    """Transforms the calculated 2048-bit bitvector into a np array.
    Returns a 1 x 2048 array.
    """
    import numpy as np
    arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(self.bitvect, arr)
    return arr
    

