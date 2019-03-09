
from tensorflow import keras
def build_model1(x_shape, z_shape):
    model = keras.Sequential()
    input1 = keras.layers.Input(shape=x_shape)
    input2 = keras.layers.Input(shape=z_shape)
    model1_out = keras.layers.CuDNNLSTM(256)(keras.layers.CuDNNLSTM(256, return_sequences=True)(input1))

    concat = keras.layers.concatenate([model1_out, input2])
    dense = keras.layers.Dense(2048, activation='tanh')(concat)
    drop = keras.layers.Dropout(0.025)(dense)
    out = keras.layers.Dense(1)(drop)
    model = keras.models.Model(inputs=[input1, input2], outputs=out)

    model.compile(loss='mse',
                optimizer='adam',
                metrics=['mape'])
    return model

def build_model2(x_shape, z_shape):
    model = keras.Sequential()
    input1 = keras.layers.Input(shape=x_shape)
    input2 = keras.layers.Input(shape=z_shape)
    model1_out = keras.layers.CuDNNLSTM(256)(keras.layers.CuDNNLSTM(256, return_sequences=True)(input1))

    concat = keras.layers.concatenate([model1_out, input2])
    dense1 = keras.layers.Dense(2048, activation='tanh')(concat)
    dense2 = keras.layers.Dense(2048, activation='tanh')(dense1)
    drop = keras.layers.Dropout(0.025)(dense2)
    out = keras.layers.Dense(1)(drop)
    model = keras.models.Model(inputs=[input1, input2], outputs=out)

    model.compile(loss='mse',
                optimizer='adam',
                metrics=['mape'])
    return model

def build_model3(x_shape, z_shape):
    model = keras.Sequential()
    input1 = keras.layers.Input(shape=x_shape)
    input2 = keras.layers.Input(shape=z_shape)
    model1_out = keras.layers.CuDNNLSTM(256)(keras.layers.CuDNNLSTM(256, return_sequences=True)(input1))

    concat = keras.layers.concatenate([model1_out, input2])
    dense1 = keras.layers.Dense(2048, activation='tanh')(concat)
    dense2 = keras.layers.Dense(2048, activation='tanh')(dense1)
    drop = keras.layers.Dropout(0.3)(dense2)
    out = keras.layers.Dense(1)(drop)
    model = keras.models.Model(inputs=[input1, input2], outputs=out)

    model.compile(loss='mse',
                optimizer='adam',
                metrics=['mape'])
    return model

def build_model4(x_shape, z_shape):
    model = keras.Sequential()
    input1 = keras.layers.Input(shape=x_shape)
    input2 = keras.layers.Input(shape=z_shape)
    model1_out = keras.layers.CuDNNLSTM(256)(keras.layers.CuDNNLSTM(256, return_sequences=True)(input1))
    model2_out = keras.layers.Dense(2048, activation='relu')(input2)

    concat = keras.layers.concatenate([model1_out, model2_out])
    dense = keras.layers.Dense(2048, activation='relu')(concat)
    drop = keras.layers.Dropout(0.3)(dense)
    out = keras.layers.Dense(1)(drop)
    model = keras.models.Model(inputs=[input1, input2], outputs=out)

    model.compile(loss='mse',
                optimizer='adam',
                metrics=['mape'])
    return model

def build_model5(x_shape, z_shape):
    model = keras.Sequential()
    input1 = keras.layers.Input(shape=x_shape)
    input2 = keras.layers.Input(shape=z_shape)
    model1_out = keras.layers.CuDNNLSTM(256)(input1)
    model2_out = keras.layers.Dense(2048, activation='relu')(input2)

    concat = keras.layers.concatenate([model1_out, model2_out])
    dense = keras.layers.Dense(2048, activation='relu')(concat)
    drop = keras.layers.Dropout(0.025)(dense)
    out = keras.layers.Dense(1)(drop)
    model = keras.models.Model(inputs=[input1, input2], outputs=out)

    model.compile(loss='mse',
                optimizer='adam',
                metrics=['mape'])
    return model

def build_model6(x_shape, z_shape):
    model = keras.Sequential()
    input1 = keras.layers.Input(shape=x_shape)
    input2 = keras.layers.Input(shape=z_shape)
    model1_out = keras.layers.CuDNNLSTM(256)(keras.layers.CuDNNLSTM(256, return_sequences=True)(keras.layers.CuDNNLSTM(256, return_sequences=True)(input1)))
    model2_out = keras.layers.Dense(2048, activation='relu')(input2)

    concat = keras.layers.concatenate([model1_out, model2_out])
    dense = keras.layers.Dense(2048, activation='relu')(concat)
    drop = keras.layers.Dropout(0.025)(dense)
    out = keras.layers.Dense(1)(drop)
    model = keras.models.Model(inputs=[input1, input2], outputs=out)

    model.compile(loss='mse',
                optimizer='adam',
                metrics=['mape'])
    return model

def build_model7(x_shape, z_shape):
    model = keras.Sequential()
    input1 = keras.layers.Input(shape=x_shape)
    input2 = keras.layers.Input(shape=z_shape)
    model1_out = keras.layers.CuDNNLSTM(256)(keras.layers.CuDNNLSTM(256, return_sequences=True)(input1))
    model2_out = keras.layers.Dense(2048, activation='relu')(input2)

    concat = keras.layers.concatenate([model1_out, model2_out])
    dense = keras.layers.Dense(2048, activation='relu')(concat)
    drop = keras.layers.Dropout(0.5)(dense)
    out = keras.layers.Dense(1)(drop)
    model = keras.models.Model(inputs=[input1, input2], outputs=out)

    model.compile(loss='mse',
                optimizer='adam',
                metrics=['mape'])
    return model

def build_model8(x_shape, z_shape):
    model = keras.Sequential()
    input1 = keras.layers.Input(shape=x_shape)
    input2 = keras.layers.Input(shape=z_shape)
    model1_out = keras.layers.CuDNNLSTM(512)(keras.layers.CuDNNLSTM(512, return_sequences=True)(input1))
    model2_out = keras.layers.Dense(2048, activation='relu')(input2)

    concat = keras.layers.concatenate([model1_out, model2_out])
    dense = keras.layers.Dense(2048, activation='relu')(concat)
    drop = keras.layers.Dropout(0.5)(dense)
    out = keras.layers.Dense(1)(drop)
    model = keras.models.Model(inputs=[input1, input2], outputs=out)

    model.compile(loss='mse',
                optimizer='adam',
                metrics=['mape'])
    return model

def build_model9(x_shape, z_shape):
    model = keras.Sequential()
    input1 = keras.layers.Input(shape=x_shape)
    input2 = keras.layers.Input(shape=z_shape)
    model1_out = keras.layers.CuDNNLSTM(256)(keras.layers.CuDNNLSTM(256, return_sequences=True)(input1))
    model2_out = keras.layers.Dense(2048, activation='relu')(input2)

    concat = keras.layers.concatenate([model1_out, model2_out])
    dense1 = keras.layers.Dense(2048, activation='relu')(concat)
    dense2 = keras.layers.Dense(2048, activation='relu')(dense1)
    drop = keras.layers.Dropout(0.5)(dense2)
    out = keras.layers.Dense(1)(drop)
    model = keras.models.Model(inputs=[input1, input2], outputs=out)

    model.compile(loss='mse',
                optimizer='adam',
                metrics=['mape'])
    return model
testModels = [
    [build_model1, "nodense"],
    [build_model2, "nodensemoredense"],
    [build_model3, "nodensemoredensedropout3"],
    [build_model4, "relu"],
    [build_model5, "onelstmrelu"],
    [build_model6, "3lstmrelu"]
]
testModels2 = [
    #[build_model4, "48seqrelu2", 48],
    [build_model7, "48seqrelu5dropout", 48],
    [build_model9, "48seqrelu5dropoutMoreDense", 48],
    [build_model8, "48seqrelu5dropout+512LSTM", 48],
]

def run(args):
    for mt in testModels2:
        import merge
        model = mt[0]
        test = mt[1]
        print(test)
        if len(mt) == 3:
            args.sequence_size = mt[2]
        args.test = test
        merge.main(args, model)