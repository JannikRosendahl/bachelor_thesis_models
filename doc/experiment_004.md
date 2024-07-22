# experiment 004 - increase layer neuron count

[commit](https://github.com/JannikRosendahl/bachelor_thesis_models/tree/793c2d9468706370cc7f148b9b70866b896a122b)  
[notebook](https://github.com/JannikRosendahl/bachelor_thesis_models/blob/793c2d9468706370cc7f148b9b70866b896a122b/rnn.ipynb)  
[artifacts](https://github.com/JannikRosendahl/bachelor_thesis_models/tree/793c2d9468706370cc7f148b9b70866b896a122b/saves/rnn/100_stratifiedsplit3)

## therory
In experiment 003 the model was not able to accurately predict the minority classes. Test if increasing the neuron count in the layers will improve the results.

## description
total number of trainable parameters roughly doubled, also increase callback patience

## architecture
```
model = Sequential(layers=[
    Input(shape=(None, no_features)),
    SimpleRNN(96, return_sequences=True),
    SimpleRNN(96, return_sequences=True),
    SimpleRNN(96, return_sequences=False),
    Dense(no_labels, activation='softmax')
])
```
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ simple_rnn (SimpleRNN)          │ (None, None, 96)       │        12,096 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ simple_rnn_1 (SimpleRNN)        │ (None, None, 96)       │        18,528 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ simple_rnn_2 (SimpleRNN)        │ (None, 96)             │        18,528 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 135)            │        13,095 │
└─────────────────────────────────┴────────────────────────┴───────────────┘

 Total params: 62,247 (243.15 KB)

 Trainable params: 62,247 (243.15 KB)

 Non-trainable params: 0 (0.00 B)
```

## parameters
| parameter                    | value                          |
|------------------------------|--------------------------------|
| sequence length              | 100                            |
| data                         | stratified split               |
| train/val/test               | 70/15/15                       |
| len(data) before seq         | 431895                         |
| len(data) after seq          | 757987                         |
| len(data) after oversampling | 1518076                        |
| len(train/val/test)          | 1290902/114330/114330          |
| labels                       | 135                            |
| features                     | event_type (29)                |
| batch size                   | 96                             |
| loss                         | categorical_focal_crossentropy |
| optimizer                    | Adam(learning_rate=0.0001)     |

## training callbacks
| callback          | parameters                               |
|-------------------|------------------------------------------|
| EarlyStopping     | val_loss, patience=7                     |
| ReduceLROnPlateau | loss, factor=.5, patience=3, min_lr=1e-7 |

## result
|              | precision | recall | f1   |
|--------------|-----------|--------|------|
| macro avg    | 0.73      | 0.75   | 0.70 |
| weighted avg | 0.95      | 0.93   | 0.93 |

- training stopped after 18 epochs (better than before)


## conclusion
Macro Average is slighly better, but still not good enough. Increasing the neuron count alone will probably not solve the problem.