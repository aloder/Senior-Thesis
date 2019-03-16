from tensorflow import keras

def build_model(shape):
    model = keras.Sequential()
    model.add(keras.layers.CuDNNLSTM(256, input_shape=shape, return_sequences=True))
    model.add(keras.layers.CuDNNLSTM(256))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(1))

    model.compile(loss='mse',
                optimizer='adam',
                metrics=['mape'])
    return model

def build_model2(shape):
    model = keras.Sequential()
    model.add(keras.layers.CuDNNLSTM(256, input_shape=shape, return_sequences=True))
    model.add(keras.layers.CuDNNLSTM(256))
    model.add(keras.layers.Dense(1024, activation='relu'))
    model.add(keras.layers.Dense(1024, activation='relu'))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(1))

    model.compile(loss='mse',
                optimizer='adam',
                metrics=['mape'])
    return model

testModels2 = [
    #[build_model4, "48seqrelu2", 48],
    [build_model, "3seq", 3],
    [build_model2, "5seqrelu", 5],
    [build_model, "5seq", 5],
    [build_model, "48seq", 48],

]

def run(args):
    for mt in testModels2:
        import LSTM
        model = mt[0]
        test = mt[1]
        print(test)
        if len(mt) == 3:
            args.sequence_size = mt[2]
        args.test = test
        LSTM.main(args, model)