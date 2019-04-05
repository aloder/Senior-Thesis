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
        model1,
        "3BigLayers"
    ],
    [
        model3,
        "2BigLayers"
    ],
    [
        model2,
        "1BigLayer"
    ],


    [
        model4,
        "3BigLayersWith5Dropout"
    ],
]

def model21(input_shape):
    model = keras.Sequential([
        keras.layers.Dense(2048,
            input_shape=input_shape, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1)
    ])
    model.compile(loss='mse',
    optimizer='adam',
    metrics=['mape'])
    return model

def model22(input_shape):
    model = keras.Sequential([
        keras.layers.Dense(2048,
            input_shape=input_shape),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1)
    ])
    model.compile(loss='mse',
    optimizer='adam',
    metrics=['mape'])
    return model

def model23(input_shape):
    model = keras.Sequential([
        keras.layers.Dense(512,
            input_shape=input_shape),
        keras.layers.Dense(512),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1)
    ])
    model.compile(loss='mse',
    optimizer='adam',
    metrics=['mape'])
    return model

    
def model24(input_shape):
    model = keras.Sequential([
        keras.layers.Dense(512,
            input_shape=input_shape, activation='relu' ),
        keras.layers.Dense(512, activation='relu' ),
        keras.layers.Dense(512, activation='relu' ),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation='relu' )
    ])
    model.compile(loss='mse',
    optimizer='adam',
    metrics=['mape'])
    return model

testModels2 = [
    [model21,"1BigLayersDropout2"],
    [model22,"1BigLayersDropout2Sigmoid"],
    [model23,"2512Layers"],
    [model24,"3512Layersrelu"],
]

def model31(input_shape):
    model = keras.Sequential([
        keras.layers.Dense(2048,
            input_shape=input_shape, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1)
    ])
    model.compile(loss='mse',
    optimizer='adam',
    metrics=['mape'])
    return model

def model32(input_shape):
    model = keras.Sequential([
        keras.layers.Dense(1024,
            input_shape=input_shape, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1)
    ])
    model.compile(loss='mse',
    optimizer='adam',
    metrics=['mape'])
    return model

def model33(input_shape):
    model = keras.Sequential([
        keras.layers.Dense(512,
            input_shape=input_shape, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1)
    ])
    model.compile(loss='mse',
    optimizer='adam',
    metrics=['mape'])
    return model

def model34(input_shape):
    model = keras.Sequential([
        keras.layers.Dense(4096,
            input_shape=input_shape, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1)
    ])
    model.compile(loss='mse',
    optimizer='adam',
    metrics=['mape'])
    return model

testModels3 = [
    [model34,"4096"],

]
def run(args):
    for mt in testModels3:
        import FFNN
        [model, test] = mt
        args.test = test
        FFNN.main(args, model)
