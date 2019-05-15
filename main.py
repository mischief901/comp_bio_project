import sys
import Compound
import Spectra
import Model
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from keras.preprocessing.sequence import pad_sequences


def get_data_as_arrays(compound_dict, save=False) :
  """Splits the values of the compound_dict into 3 lists of np arrays
  [input_data, output_data, auxiliary_output_data]

  The arrays are not padded.

  If save=True is given, the same arrays are saved as .npy files.
  i.e. "input_data.npy"
  """
  input_data = []
  output_data = []
  auxiliary_output_data = []
  for c in compound_dict.values() :
    try :
      #if c.mol_weight > 1000 :
      #  continue
      [mass_peak, intensity] = list(zip(*c.spectras[0].peaks))
      if not c.to_bitvector('morgan', {'radius': 2}) :
        continue
      bitvect_array = c.bitvect_as_np_array()
      input_data.append(bitvect_array)
      output_data.append(mass_peak)
      auxiliary_output_data.append(intensity)
    except :
      continue
  input_array = np.asarray(input_data)
  output_array = np.asarray(output_data)
  auxiliary_output_array = np.asarray(auxiliary_output_data)
  print(input_array.shape)
  if save :
    np.save("pred_input_data", input_array)
    np.save("pred_output_data", output_array)
    np.save("pred_auxiliary_output_data", auxiliary_output_array)
    
  return(input_array,
         output_array,
         auxiliary_output_array)


def combine_data(spectras, compounds) :
  """This function combines the spectras and corresponding compounds together.
  Returns a dictionary of hmdb_id => Compound object with an assigned Spectra object.
  """
  final = {}
  for hmdb_id, spec_objs in spectras.items() :
    c = compounds.pop(hmdb_id, None)
    if not c :
      continue
    c.spectras = spec_objs
    final[hmdb_id] = c
  return final


def load_data() :
  try :
    input_data = np.load("pred_input_data.npy")
    output_data = np.load("pred_output_data.npy", allow_pickle=True)
    aux_output_data = np.load("pred_auxiliary_output_data.npy", allow_pickle=True)
    return input_data, output_data, aux_output_data
  except :
    print("Failed to load data")
    return None, None, None

def load_new(testing, save) :
  ## Getting structures from the sdf file.
  reader = Compound.Reader('structures.sdf')
  ## Store all the compounds in a dict of HMDB => object
  compounds = {}
  for _ in range(len(reader)) :
    if testing and len(compounds) > 1000 :
      break
    try :
      c = Compound.Compound(reader)
      ## Stop when the reader is finished. (should return none)
      compounds[c.hmdb_id] = deepcopy(c)
    except KeyboardInterrupt :
      raise KeyboardInterrupt
    except :
      continue
      
  ## Getting spectra.
  folder = 'predicted_ms_ms_spectra'
  filter_funs = [Spectra.voltage_filter,
                 lambda x : Spectra.voltage_level(x, 40)]
  spectras = Spectra.get_all(folder, filter_funs)
    
  combined = combine_data(spectras, compounds)
    
  [bitvects, mass_peaks, intensities] = get_data_as_arrays(combined, save)
  print(len(mass_peaks), " samples collected.")
  return bitvects, mass_peaks, intensities

def normalize_data(data) :
  for i, row in enumerate(data) :
    int_sum = int(np.sum(row))
    data[i] = np.true_divide(row, int_sum)
    #data[i] = np.multiply(np.true_divide(row, int_sum), 100)  

def train(testing, load_existing, save, with_intensity, embedder, input_data,
          output_data1, output_data2) :
  m = Model.Model()
  build_options = {'conv_layers': 1,
                   'rec_layers_intensity': True,
                   'rec_intensity_layers': 5,
                   'rec_layers': 3,
                   'pool_stride': 5,
                   'conv_units': 2**4,
                   'lstm_units': 2**6,
                   'lstm_dropout': 0.2,
                   'dense_layers': 3,
                   'dense_units': 2**10,
                   'conv_type': 64,
                   'conv_dropout': 0.2,
                   'with_intensity': with_intensity}

  m.build(embedder.get_size(), options=build_options)

  outputs = [output_data1]
  if with_intensity :
    outputs.append(output_data2)
    
  history, x_test, y_tests = m.run(input_data, outputs, epochs=100, batch=1024,
                                   intensity=with_intensity)

  m.model.save("pred_model.h5")
  
  #scores = m.evaluate(x_test, y_tests)
  #features = m.predict(x_test)

  file_name = "model"
  
  m.plot_model("keras_pred_" + file_name + ".png")
  
  for key in history.history.keys() :
    plt.plot(history.history[key])
  plt.legend(history.history.keys(), loc='upper right')
  plt.savefig("pred_" + file_name + ".png")
      
  return history, y_tests

def main(testing=False, load_existing=True, save=True, with_intensity=False,
         retrain_model=False, load_model = None) :

  input_data = None
  output_data = None
  aux_output_data = None
  if load_existing :
    print("Loading existing data")
    input_data, output_data, aux_output_data = load_data()
    
  try :
    ## Stupid and lazy hack since apparently I cannot use 'input_data == None' or
    ## 'not input_data' to check if it's loaded.
    loaded = input_data.size > 0
  except :
    loaded = False
  if not loaded :
    input_data, output_data, aux_output_data = load_new(testing, save)

  embedder = Spectra.Embedded_Spectra(output_data)
  embedder.set_translation()  
  
  array1 = np.zeros((output_data.size, embedder.get_size()))
  array2 = np.zeros((output_data.size, embedder.get_size()))
  
  normalize_data(aux_output_data)

  spectras1 = list(output_data)
  spectras2 = [list(zip(*s)) for s in list(zip(output_data, aux_output_data))]

  for i, spectra in enumerate(spectras1) :
    array1[i] = embedder.to_array(spectra, False)
  #Model.test_varied("wo_i", input_data, array1)
  
  for i, spectra in enumerate(spectras2) :
    array2[i] = embedder.to_array(spectra, True)
  #Model.test_varied("w_i", input_data, array2)

  if retrain_model :
    train(testing, load_existing, save, with_intensity,
          embedder, input_data, array1, array2)
    return
  m = Model.Model(load_model)

  ## Set the random seed so that it is the same for all choices.
  np.random.seed(901)
  (r, _) = input_data.shape
  for i in range(10) :
    index = np.random.randint(r)
    print(index)
    input_choice = input_data[index]
    output_choice1 = array1[index]
    output_choice2 = array2[index]

    [p1, p2] = m.predict(input_choice)
    print(output_data[index])
    plt.plot(np.multiply(output_choice1, -1))
    plt.plot(p1[0])
    plt.show()
    print(aux_output_data[index])
    plt.plot(np.multiply(output_choice2, -1))
    plt.plot(p2[0][0])
    plt.show()
    
    
  return p1, p2, output_choice1, output_choice2, output_data[index], aux_output_data[index]


if __name__ == "__main__" :

  #main(with_intensity=True, retrain_model=True, load_existing=False, save=True)
  #main(with_intensity=True, retrain_model=True)
  main(with_intensity=True, load_model="pred_model.h5")

  
def to_msms() :
  pass
  if with_intensity :
    for i, (a1, a2) in enumerate(zip(*features)) :
      predicted = embedder.from_array(a1, a2)
      fp = open("spectra_w_i_"+str(i), "w")
      real = embedder.from_array(y1_test[i], y2_test[i])
      for entry in predicted :
        fp.write("Peak: {}\t{}".format(*entry))
      for entry in real :
        fp.write("Peak: {}\t{}".format(*entry))

  else :
    for i, array in enumerate(features) :
      predicted = embedder.from_array(array)
      fp = open("spectra_wo_i_"+str(i), "w")
      real = embedder.from_arrray(y_tests[i])
      fp.write("Predicted:")
      for entry in predicted :
        fp.write(" {}\n".format(entry))
      fp.write("Real:")
      for entry in real :
        fp.write(" {}\n".format(entry))
