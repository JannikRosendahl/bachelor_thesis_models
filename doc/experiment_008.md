# experiment 008 - feature: user, predicate file types

[commit](https://github.com/JannikRosendahl/bachelor_thesis_models/tree/c6a7f3d437546fdf196bc0b02c4b6c2ead8d0bda)  
[notebook](https://github.com/JannikRosendahl/bachelor_thesis_models/tree/c6a7f3d437546fdf196bc0b02c4b6c2ead8d0bda/rnn.ipynb)  
[artifacts](https://github.com/JannikRosendahl/bachelor_thesis_models/tree/c6a7f3d437546fdf196bc0b02c4b6c2ead8d0bda/saves/rnn/100_user_filetypes)

## therory
Previous experiments showed that more features were needed to achieve better results. This experiment adds the user (one-hot 17) and the predicate objects file types (2x one-hot 7) to the model.

## description

## architecture
```
model = Sequential(layers=[
    Input(shape=(None, feature_vector_cardinality)),
    SimpleRNN(64, return_sequences=True),
    SimpleRNN(64, return_sequences=True),
    SimpleRNN(64, return_sequences=False),
    Dense(labels_cardinality, activation='softmax')
])
```
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ simple_rnn_11 (SimpleRNN)       │ (None, None, 64)       │         8,000 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ simple_rnn_12 (SimpleRNN)       │ (None, None, 64)       │         8,256 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ simple_rnn_13 (SimpleRNN)       │ (None, 64)             │         8,256 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_3 (Dense)                 │ (None, 81)             │         5,265 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 29,777 (116.32 KB)
 Trainable params: 29,777 (116.32 KB)
 Non-trainable params: 0 (0.00 B)
```

## parameters
| parameter                    | value                                         |
|------------------------------|-----------------------------------------------|
| sequence length              | 100                                           |
| data                         | only data before 2018-04-06 11:20:00          |
| unusable_threshold           | 10                                            |
| rare_threshold               | 50                                            |
| rare_dup_factor              | 25                                            |
| train/val/test               | 70/15/15                                      |
| len(data) before seq         | 215150                                        |
| len(data) after seq          | 349190                                        |
| len(data) after oversampling | 360698                                        |
| len(train/val/test)          | 255980/52359/52359                            |
| labels                       | 81                                            |
| features                     | event_type (29), user (17), predfile{1,2} (7+7) |
| feature_vector_size          | 60                                            |
| batch size                   | 96                                            |
| loss                         | categorical_focal_crossentropy                |
| optimizer                    | Adam(learning_rate=0.0001)                    |

## training callbacks
| callback          | parameters                               |
|-------------------|------------------------------------------|
| EarlyStopping     | val_loss, patience=5                     |
| ReduceLROnPlateau | loss, factor=.5, patience=3, min_lr=1e-7 |

## result
|              | precision | recall | f1   |
|--------------|-----------|--------|------|
| macro avg    | 0.89      | 0.85   | 0.82 |
| weighted avg | 0.98      | 0.98   | 0.98 |

- training stopped after 63 epochs


## conclusion
