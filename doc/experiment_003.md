# experiment 003 - stratified split with training set oversampling

[commit](https://github.com/JannikRosendahl/bachelor_thesis_models/tree/c789dae24f5864ec89db4ea88792b232434b3a87)  
[notebook](https://github.com/JannikRosendahl/bachelor_thesis_models/blob/c789dae24f5864ec89db4ea88792b232434b3a87/rnn.ipynb)  
[artifacts](https://github.com/JannikRosendahl/bachelor_thesis_models/tree/c789dae24f5864ec89db4ea88792b232434b3a87/saves/rnn/100_stratifiedsplit2)

## therory
Only oversampling the minority classes in the training set should improve the results. Macro Average should be better than in experiment 001 (as seen in experiment 002) and Weighted Average should be better than in experiment 002.

## description
RNN with stratified split. Oversampling only for training set.

## architecture
```
model = Sequential(layers=[
    Input(shape=(None, no_features)),
    SimpleRNN(64, return_sequences=True),
    SimpleRNN(64, return_sequences=True),
    SimpleRNN(64, return_sequences=False),
    Dense(no_labels, activation='softmax')
])
```
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ simple_rnn (SimpleRNN)          │ (None, None, 64)       │         6,016 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ simple_rnn_1 (SimpleRNN)        │ (None, None, 64)       │         8,256 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ simple_rnn_2 (SimpleRNN)        │ (None, 64)             │         8,256 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 135)            │         8,775 │
└─────────────────────────────────┴────────────────────────┴───────────────┘

 Total params: 31,303 (122.28 KB)

 Trainable params: 31,303 (122.28 KB)

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
| EarlyStopping     | val_loss, patience=5                     |
| ReduceLROnPlateau | loss, factor=.5, patience=2, min_lr=1e-7 |

## result
|              | precision | recall | f1   |
|--------------|-----------|--------|------|
| macro avg    | 0.69      | 0.72   | 0.67 |
| weighted avg | 0.95      | 0.92   | 0.93 |

- training stopped after 10 epochs (much earlier than before!)
- some classes were never predicted, but less than in experiment 002


## conclusion
Macro Average is worse than in experiment 002, weighted average is better.  
Expected macro average to be better. The model may need more time, data or a different architecture to learn the minority classes.
Nevertheless, more features are probably needed to significantly improve the results.
