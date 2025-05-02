These are trhe results: 
Evaluating on Validation Set...

=== BERT XGBoost on Validation Set ===
Accuracy: 0.9860

Classification Report:

              precision    recall  f1-score   support

           0     0.9919    0.9965    0.9942     19101
           1     0.9897    0.9438    0.9662       409
           2     0.9725    0.9728    0.9727      3711
           3     0.9858    0.9731    0.9794      1707
           4     0.9724    0.9119    0.9412       193
           5     0.9206    0.8850    0.9025       983

    accuracy                         0.9860     26104
   macro avg     0.9722    0.9472    0.9593     26104
weighted avg     0.9859    0.9860    0.9859     26104

Confusion Matrix:

[[19035     1    32     5     0    28]
 [   13   386     6     3     0     1]
 [   55     0  3610    10     2    34]
 [   21     1    15  1661     0     9]
 [    7     0     5     2   176     3]
 [   60     2    44     4     3   870]]

=== Static XGBoost on Validation Set ===
Accuracy: 0.9794

Classification Report:

              precision    recall  f1-score   support

           0     0.9891    0.9945    0.9918     19101
           1     0.9282    0.9169    0.9225       409
           2     0.9558    0.9617    0.9588      3711
           3     0.9905    0.9777    0.9841      1707
           4     0.8919    0.8549    0.8730       193
           5     0.8900    0.8067    0.8463       983

    accuracy                         0.9794     26104
   macro avg     0.9409    0.9187    0.9294     26104
weighted avg     0.9790    0.9794    0.9791     26104

Confusion Matrix:

[[18995    21    43     5     5    32]
 [   24   375     1     1     1     7]
 [   90     1  3569     3     2    46]
 [   22     1     9  1669     1     5]
 [   18     0     2     0   165     8]
 [   56     6   110     7    11   793]]

Evaluating on Test Set...

=== BERT XGBoost on Test Set ===
Accuracy: 0.9855

Classification Report:

              precision    recall  f1-score   support

           0     0.9917    0.9955    0.9936     19102
           1     0.9898    0.9439    0.9663       410
           2     0.9686    0.9722    0.9704      3711
           3     0.9893    0.9713    0.9802      1707
           4     0.9775    0.9062    0.9405       192
           5     0.9198    0.8983    0.9089       983

    accuracy                         0.9855     26105
   macro avg     0.9728    0.9479    0.9600     26105
weighted avg     0.9854    0.9855    0.9854     26105

Confusion Matrix:

[[19016     0    53     4     0    29]
 [   11   387     4     2     1     5]
 [   59     2  3608     7     2    33]
 [   25     1    15  1658     0     8]
 [   13     1     2     0   174     2]
 [   51     0    43     5     1   883]]

=== Static XGBoost on Test Set ===
Accuracy: 0.9797

Classification Report:

              precision    recall  f1-score   support

           0     0.9892    0.9934    0.9912     19102
           1     0.9233    0.9098    0.9165       410
           2     0.9580    0.9652    0.9616      3711
           3     0.9929    0.9789    0.9858      1707
           4     0.9191    0.8281    0.8712       192
           5     0.8841    0.8301    0.8562       983

    accuracy                         0.9797     26105
   macro avg     0.9444    0.9176    0.9304     26105
weighted avg     0.9795    0.9797    0.9795     26105

Confusion Matrix:

[[18975    24    51     5     4    43]
 [   26   373     1     0     0    10]
 [   78     0  3582     2     4    45]
 [   27     1     2  1671     2     4]
 [   19     1     8     0   159     5]
 [   58     5    95     5     4   816]]





 ![pre-fine-tune-cls png](https://github.com/user-attachments/assets/b8a0c27f-c6c4-4e55-8133-66b64a902f93)
![pca_prefinetune_diff](https://github.com/user-attachments/assets/9ad8878a-15ed-460a-8c06-4e6802fc8b0b)
![pca_finetunecls](https://github.com/user-attachments/assets/9849321d-5199-42b4-a562-8f2d674ea20f)
![pca_finetune_diff](https://github.com/user-attachments/assets/cf2f0a4c-036c-4b7f-bc41-ce3253437deb)
![234](https://github.com/user-attachments/assets/2475090d-bbc3-4865-bf49-b619021d3f0c)

