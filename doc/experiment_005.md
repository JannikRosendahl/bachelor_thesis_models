# experiment 005 - increase layer count

[commit](https://github.com/JannikRosendahl/bachelor_thesis_models/tree/8bfd286d1cbb25620cea0d4ef4432027f72fe769)  
[notebook](https://github.com/JannikRosendahl/bachelor_thesis_models/blob/8bfd286d1cbb25620cea0d4ef4432027f72fe769/rnn.ipynb)  
[artifacts](https://github.com/JannikRosendahl/bachelor_thesis_models/tree/8bfd286d1cbb25620cea0d4ef4432027f72fe769/saves/rnn/100_widernet)

## therory
Experiment 004 showed that increasing neuron count yields only slightly better macro average. Check if increasing layer count will improve the results. If not, the model probably needs more features.

## description
total number of trainable parameters roughly doubled, also increase callback patience

## architecture
```
model = Sequential(layers=[
    Input(shape=(None, no_features)),
    SimpleRNN(64, return_sequences=True),
    SimpleRNN(64, return_sequences=True),
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
│ simple_rnn_15 (SimpleRNN)       │ (None, None, 64)       │         6,016 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ simple_rnn_16 (SimpleRNN)       │ (None, None, 64)       │         8,256 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ simple_rnn_17 (SimpleRNN)       │ (None, None, 64)       │         8,256 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ simple_rnn_18 (SimpleRNN)       │ (None, None, 64)       │         8,256 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ simple_rnn_19 (SimpleRNN)       │ (None, 64)             │         8,256 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_3 (Dense)                 │ (None, 135)            │         8,775 │
└─────────────────────────────────┴────────────────────────┴───────────────┘

 Total params: 47,815 (186.78 KB)

 Trainable params: 47,815 (186.78 KB)

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
| macro avg    | 0.90      | 0.75   | 0.72 |
| weighted avg | 0.95      | 0.93   | 0.93 |

- training stopped after 22 epochs


## conclusion
Macro Average precision significantly (.73 -> .9) improved, but other metrics did not. Increasing layer count will help, but additional features are needed, as expected.