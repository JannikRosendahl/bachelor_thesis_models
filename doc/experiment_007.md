# experiment 007 - only benign data

[commit](https://github.com/JannikRosendahl/bachelor_thesis_models/tree/5a29a63e50727da54aef4d56731559015d0b904a)  
[notebook](https://github.com/JannikRosendahl/bachelor_thesis_models/blob/5a29a63e50727da54aef4d56731559015d0b904a/rnn.ipynb)  
[artifacts](https://github.com/JannikRosendahl/bachelor_thesis_models/tree/5a29a63e50727da54aef4d56731559015d0b904a/saves/rnn/100_benign)

## therory
Training a model only on the subset of benign data generate before 2018-04-06 11:20:00 should be significantly faster and should yield slightly better results. The downside should be that not all labels are present in the data. 

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
| parameter                    | value                                |
|------------------------------|--------------------------------------|
| sequence length              | 100                                  |
| data                         | only data before 2018-04-06 11:20:00 |
| train/val/test               | 70/15/15                             |
| len(data) before seq         | 215150                               |
| len(data) after seq          | 349117                               |
| len(data) after oversampling | 1518076                              |
| len(train/val/test)          | 664670/53066/53067                |
| labels                       | 118                                  |
| features                     | event_type (29)                      |
| batch size                   | 96                                   |
| loss                         | categorical_focal_crossentropy       |
| optimizer                    | Adam(learning_rate=0.0001)           |

## training callbacks
| callback          | parameters                               |
|-------------------|------------------------------------------|
| EarlyStopping     | val_loss, patience=7                     |
| ReduceLROnPlateau | loss, factor=.5, patience=3, min_lr=1e-7 |

## result
|              | precision | recall | f1   |
|--------------|-----------|--------|------|
| macro avg    | 0.91      | 0.79   | 0.77 |
| weighted avg | 0.96      | 0.94   | 0.94 |

- training stopped after 24 epochs


## conclusion
Benign dataset is still quite large and should be sufficient to train a representative model. Classes that are not present anymore had low counts and should not have a significant impact on the model.  
Metrics look as expected.