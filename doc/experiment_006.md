# experiment 006 - bidirectional layers

[commit](https://github.com/JannikRosendahl/bachelor_thesis_models/tree/c3cd7dce34e7891cc0bce8cc467e2f091562f6dd)  
[notebook](https://github.com/JannikRosendahl/bachelor_thesis_models/blob/c3cd7dce34e7891cc0bce8cc467e2f091562f6dd/rnn.ipynb)  
[artifacts](https://github.com/JannikRosendahl/bachelor_thesis_models/tree/c3cd7dce34e7891cc0bce8cc467e2f091562f6dd/saves/rnn/100_bidirectional)

## therory
Test if using bidirectional layers improves metrics.

## description

## architecture
```
model = Sequential(layers=[
    Input(shape=(None, no_features)),
    Bidirectional(SimpleRNN(64, return_sequences=True)),
    Bidirectional(SimpleRNN(64, return_sequences=True)),
    Bidirectional(SimpleRNN(64, return_sequences=False)),
    Dense(no_labels, activation='softmax')
])
```
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ bidirectional_3 (Bidirectional) │ (None, None, 128)      │        12,032 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ bidirectional_4 (Bidirectional) │ (None, None, 128)      │        24,704 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ bidirectional_5 (Bidirectional) │ (None, 128)            │        24,704 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 135)            │        17,415 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 78,855 (308.03 KB)
 Trainable params: 78,855 (308.03 KB)
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
| macro avg    | 0.88      | 0.74   | 0.72 |
| weighted avg | 0.95      | 0.93   | 0.93 |

- manually stopped training after 4 epochs


## conclusion
Some small improvements in macro avg. More features needed.