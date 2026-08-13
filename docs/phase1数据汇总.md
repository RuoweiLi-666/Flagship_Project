



| Model          | specification                       | Accuracy | Macro-F1 | Balanced Accuracy | Reall(ADL) | Recall (Near Fall) | Recall (Fall) |
| :------------- | ----------------------------------- | -------- | -------- | ----------------- | ---------- | ------------------ | ------------- |
| **1D-CNN**     | --                                  | 0.9483   | 0.9433   | 0.9435            | 0.9667     | 0.9067             | 0.9571        |
| **LSTM**       | 2-layer                             | 0.7250   | 0.7120   | 0.7201            | 0.6042     | 0.6133             | 0.9429        |
|                | 3-layer                             | 0.7600   | 0.7461   | 0.7497            | 0.6833     | 0.6133             | 0.9524        |
| **TCN**        | 2-resblock                          |          |          |                   |            |                    |               |
|                | 4-resblock                          |          |          |                   |            |                    |               |
|                | 6-resblock                          |          |          |                   |            |                    |               |
|                | 8-resblock                          |          |          |                   |            |                    |               |
| **Conv1D-TCN** | 1-conv <br />4-resblk (same follow) |          |          |                   |            |                    |               |
|                | 2-conv                              |          |          |                   |            |                    |               |
|                | 3-conv                              |          |          |                   |            |                    |               |
|                | 2-conv <br />1 dropout              |          |          |                   |            |                    |               |
|                | 2-conv <br />2 dropout              |          |          |                   |            |                    |               |
|                |                                     |          |          |                   |            |                    |               |
|                |                                     |          |          |                   |            |                    |               |