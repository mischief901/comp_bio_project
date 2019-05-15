import matplotlib.pyplot as plt
import numpy as np
from keras.models import Model as KModel
from keras.models import load_model
from keras.layers import Input, Dense, LSTM, Bidirectional, Activation, add
from keras.layers import MaxPooling1D, Conv1D, Reshape, Dropout
from keras.backend import clear_session
from sklearn.model_selection import train_test_split


class Model :
  """The Model class sets up a Convolutional Recurrent Neural Net.
  Usage:
  model = Model.Model()
  model.build()
  model.run()
  """

  def __init__(self, model_file=None) :
    if not model_file :
      self.model = None
    else :
      self.model = load_model(model_file, custom_objects={'rounded_cosine':rounded_cosine})
    

  def build(self, exit_size, options={}) :

    inputs = Input(shape=(1,2048))
    outputs = options.get('with_intensity', False)
    conv_num = options.get('conv_layers', 2)
    conv_units = options.get('conv_units', 16)
    conv_type = options.get('conv_type', 16)
    conv_dropout = options.get('conv_dropout', 0.1)
    pooling_stride = options.get('pool_stride', 2)
    pooling_size = options.get('pool_size', 1)
    dense_units = options.get('dense_units', 64)
    dense_layers = options.get('dense_layers', 3)
    
    conv_layer = inputs
    if conv_num > 0 :
      for _ in range(conv_num) :
        conv_layer = Conv1D(conv_units, conv_type, padding='same')(conv_layer)
        conv_layer = Activation('relu')(conv_layer)
        conv_layer = MaxPooling1D(pool_size=pooling_size,
                                  strides=pooling_stride)(conv_layer)
        conv_layer = Dropout(conv_dropout)(conv_layer)
        
    conv_layer = Dense(dense_units)(conv_layer)

    rec_num = options.get('rec_layers', 2)
    rec_intensity = options.get('rec_layers_intensity', False)
    rec_num_intensity = options.get('rec_intensity_layers', 0)
    lstm_units = options.get('lstm_units', 16)
    lstm_activation = options.get('lstm_activation', 'relu')
    lstm_dropout = options.get('lstm_dropout', 0.1)

    int_rec_layer = conv_layer
    if rec_intensity :
      for _ in range(rec_num_intensity-1) :
        int_rec_layer = Bidirectional(
          LSTM(lstm_units, activation=lstm_activation, return_sequences=True,
               dropout=lstm_dropout))(int_rec_layer)
      int_rec_layer = Bidirectional(LSTM(lstm_units, return_sequences=False))(int_rec_layer)
    
    rec_layer = inputs
    if rec_num > 0 :
      for _ in range(rec_num-1) :
        rec_layer = Bidirectional(
          LSTM(lstm_units, activation=lstm_activation, return_sequences=True,
               dropout=lstm_dropout))(rec_layer)
      rec_layer = Bidirectional(LSTM(lstm_units, return_sequences=False))(rec_layer)

    out = rec_layer
    if dense_layers > 0 :
      for _ in range(dense_layers-1) :
        out = Dense(dense_units)(out)
        
    out = Dense(exit_size, name='peaks', activation='relu')(out)

    if outputs :
      out2 = conv_layer
      for _ in range(dense_layers) :
        out2 = Dense(dense_units, activation='relu')(out2)
      out2 = Dense(exit_size, activation='relu', name='intensity')(out2)

      self.model = KModel(inputs=inputs, output=[out,out2])
      self.model.compile(optimizer='adam',
                         metrics=['accuracy', rounded_cosine],
                         loss=['cosine',
                               'cosine'])
    else :
      self.model = KModel(inputs=inputs, outputs=out)
      self.model.compile(optimizer='adam',
                         metrics=['accuracy', rounded_cosine],
                         loss='cosine')
      
    print(self.model.summary())
    return

    
  def run(self, input_1, outputs, epochs=25, batch=None, intensity=False) :
    
    x = input_1.reshape(len(input_1), 1, len(input_1[0]))
    if len(outputs) == 2 :
      [input_2, input_3] = outputs
      input_3 = input_3.reshape(len(input_3), 1, len(input_3[0]))
      (x_train, x_test, y1_train,
       y1_test, y2_train, y2_test) = train_test_split(x, input_2, input_3, test_size=0.1)
      if not batch :
        (batch, _) = y1_test.shape
      
      history = self.model.fit(x_train, [y1_train, y2_train], batch_size=batch,
                               epochs=epochs, shuffle=True, validation_split=0.2,
                               validation_freq=1)
      return history, x_test, [y1_test, y2_test]
    
    x_train, x_test, y_train, y_test = train_test_split(x, *outputs, test_size=0.2)
    if not batch :
      (batch, _) = y_test.shape
      
    history = self.model.fit(x_train, y_train, batch_size=batch, epochs=epochs,
                             shuffle=True, validation_split=0.2, validation_freq=1)
    return history, x_test, y_test

  def evaluate(self, x_test, y_tests) :
    scores = self.model.evaluate(x_test, y_tests, verbose=1)
    print(scores)
    return scores

  def predict(self, x) :
    x = x.reshape(1, 1, 2048)
    features = self.model.predict(x)
    return features

  def plot_model(self, file_name) :
    from keras.utils import plot_model as plot
    plot(self.model, to_file=file_name)
  

def rounded_cosine(ytrue, ypred) :
  """Rounds the predicted values before taking the cosine difference.
  """
  from keras.losses import cosine_proximity
  from tensorflow.keras.backend import round as Kround
  ypred = Kround(ypred)
  return -cosine_proximity(ytrue, ypred)
  

    
def test_varied(file_name, x, y) :
  options = {}
  f = file_name
  (_, exit_size) = y.shape
  print(exit_size)
  for p in range(2, 5) :
    options['pool_stride'] = p
    for u in range(5, 9) :
      options['conv_units'] = 2**u
      for k in range(5, 10, 2) :
        options['dense_units'] = 2**k
        #for i in range(1, 4, 2) :
        options['conv_layers'] = 3
        #for j in range(2, 4) :
        options['rec_layers'] = 3
        clear_session()
        m = Model()
        m.build(exit_size, options)
        hist = m.run(x, y)
        file_name = "{}_{}_{}_{}".format(f, p, u, k)
        fp = open(file_name + ".txt", "w")
        m.model.summary(print_fn=lambda x: fp.write(x + '\n'))
        fp.write(str(hist.history))
        fp.close()
        plt.plot(hist.history['acc'])
        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_acc'])
        plt.plot(hist.history['val_loss'])
        plt.legend(['train_acc', 'train_loss', 'val_acc', 'val_loss'])
        plt.savefig(file_name + ".png")
        plt.clf()
      
      

  
