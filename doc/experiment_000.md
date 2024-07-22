# experiment 000 - feed forward baseline test

[commit](https://github.com/JannikRosendahl/bachelor_thesis_models/tree/cc2322f290761cb15dd3c2bb7e9af1ed4175682a)  
[notebook](https://github.com/JannikRosendahl/bachelor_thesis_models/blob/cc2322f290761cb15dd3c2bb7e9af1ed4175682a/ffn.ipynb)  
[artifacts](https://github.com/JannikRosendahl/bachelor_thesis_models/tree/cc2322f290761cb15dd3c2bb7e9af1ed4175682a/saves/fnn)

## therory
Before starting with RNNs, test if the data can be learned with a simple feed forward network. If so, the RNNs should be able to learn the data as well.

## description
The sequence data needs to be flattened before feeding it into the network. Flatten sequences to 2D tensor describing where an element describes the count of a event occurrence in the sequence.

## architecture
```
model = Sequential(layers=[
    Input(shape=(no_features,)),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(no_labels, activation='softmax')
])
```
```


┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ dense (Dense)                   │ (None, 64)             │         1,920 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 64)             │         4,160 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (Dense)                 │ (None, 64)             │         4,160 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_3 (Dense)                 │ (None, 135)            │         8,775 │
└─────────────────────────────────┴────────────────────────┴───────────────┘

 Total params: 19,015 (74.28 KB)

 Trainable params: 19,015 (74.28 KB)

 Non-trainable params: 0 (0.00 B)
```

## parameters
| parameter            | value                          |
|----------------------|--------------------------------|
| sequence length      | -                              |
| data                 | flattened                      |
| train/val/test       | 70/15/15                       |
| len(data) before seq | 431895                         |
| len(data) after seq  | -                         |
| labels               | 135                            |
| features             | event_type (29)                |
| batch size           | 96                             |
| loss                 | categorical_focal_crossentropy |
| optimizer            | Adam(learning_rate=0.0001)     |

## training callbacks
| callback          | parameters                               |
|-------------------|------------------------------------------|
| EarlyStopping     | val_loss, patience=5                     |
| ReduceLROnPlateau | loss, factor=.5, patience=2, min_lr=1e-7 |

## result
|              | precision | recall | f1   |
|--------------|-----------|--------|------|
| macro avg    | 0.50      | 0.52   | 0.49 |
| weighted avg | 0.93      | 0.94   | 0.93 |

- training stopped after 51 epochs
- some classes were never predicted


## conclusion
Theory confirmed.  
Prediction is possible, although the macro average is not very good, some classes have very little support.