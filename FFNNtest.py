from tensorflow import keras

def model1(input_shape):
    model = keras.Sequential([
        keras.layers.Dense(2048,
            input_shape=input_shape, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1)
    ])
    model.compile(loss='mse',
    optimizer='adam',
    metrics=['mape'])
    return model

def model2(input_shape):
    model = keras.Sequential([
        keras.layers.Dense(2048,
            input_shape=input_shape, activation='relu'),
        keras.layers.Dense(2048, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1)
    ])
    model.compile(loss='mse',
    optimizer='adam',
    metrics=['mape'])
    return model

def model3(input_shape):
    model = keras.Sequential([
        keras.layers.Dense(2048,
            input_shape=input_shape, activation='relu'),
        keras.layers.Dense(2048, activation='relu'),
        keras.layers.Dense(2048, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1)
    ])
    model.compile(loss='mse',
    optimizer='adam',
    metrics=['mape'])
    return model
    
def model4(input_shape):
    model = keras.Sequential([
        keras.layers.Dense(2048,
            input_shape=input_shape, activation='relu'),
        keras.layers.Dense(2048, activation='relu'),
        keras.layers.Dense(2048, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1)
    ])
    model.compile(loss='mse',
    optimizer='adam',
    metrics=['mape'])
    return model
    
testModels = [
    [
        model3,
        "2BigLayers"
    ],
    [
        model2,
        "1BigLayer"
    ],

    [
        model1,
        "3BigLayers"
    ],
    [
        model4,
        "3BigLayersWith5Dropout"
    ],
]
def run(args):
    for mt in testModels:
        import FFNN
        [model, test] = mt
        args.test = test
        FFNN.main(args, model)
