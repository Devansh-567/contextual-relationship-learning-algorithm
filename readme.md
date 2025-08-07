# Contextual Relationship Learning Algorithm (CHLA)

A machine learning algorithm that learns and evolves relationships between multiple contexts (e.g., users, time, product categories) using a hypergraph structure. CHLA is designed for real-time, adaptive systems such as recommendation engines and context-aware decision-making.

---

## 📌 Overview

**Contextual Hypergraph Learning Algorithm (CHLA)** is a novel approach for modeling complex, dynamic relationships across multiple dimensions of context. Instead of relying on static rules or simple matrix factorization, CHLA adapts in real-time to evolving patterns using a hypergraph-based structure.

---

## 🚀 Key Features

- **Multi-context Learning**: Handles relations between any combinations of contexts (e.g., User–Time, Product–User).
- **Dynamic Relation Updates**: Continuously evolves the strength of relationships over training epochs.
- **Real-time Adaptability**: Suitable for environments where data and user behavior shift over time.
- **Lightweight Implementation**: Easy to run and extend — no external ML libraries required.
- **Readable Output**: Prints contextual relation scores during each epoch.

---

## 🧠 Use Cases

- Personalized Recommendation Systems
- Context-Aware Decision Engines
- Adaptive Pricing or Promotions
- User Behavior Modeling in Dynamic Environments
- Real-Time Social or Interaction Networks

---

```python
contexts = ['User A', 'User B', 'Morning', 'Evening', 'Electronics', 'Clothing']
Epoch 0, Context ('User A', 'User B'): Updated Relation = 0.8651550461182852
Epoch 0, Context ('Morning', 'User A'): Updated Relation = 0.440922603408043
Epoch 0, Context ('Evening', 'User A'): Updated Relation = 0.8309472086798552
Epoch 0, Context ('Electronics', 'User A'): Updated Relation = 0.5935735225190442
Epoch 0, Context ('Clothing', 'User A'): Updated Relation = 0.8497774674468686
Epoch 0, Context ('Morning', 'User B'): Updated Relation = 0.39141952730616975
Epoch 0, Context ('Evening', 'User B'): Updated Relation = 0.5688117766256019
Epoch 0, Context ('Electronics', 'User B'): Updated Relation = 0.7480888941955813
Epoch 0, Context ('Clothing', 'User B'): Updated Relation = 0.7476175176149149
Epoch 0, Context ('Evening', 'Morning'): Updated Relation = 0.419175804804129
Epoch 0, Context ('Electronics', 'Morning'): Updated Relation = 0.8447862212168546
Epoch 0, Context ('Clothing', 'Morning'): Updated Relation = 0.24491060464264464
Epoch 0, Context ('Electronics', 'Evening'): Updated Relation = 0.29602851695529414
Epoch 0, Context ('Clothing', 'Evening'): Updated Relation = 0.8801841194041925
Epoch 0, Context ('Clothing', 'Electronics'): Updated Relation = 0.704383068889061
Epoch 1, Context ('User A', 'User B'): Updated Relation = 0.861899067514162
Epoch 1, Context ('Morning', 'User A'): Updated Relation = 0.43259167300342083
Epoch 1, Context ('Evening', 'User A'): Updated Relation = 0.8378167216752785
Epoch 1, Context ('Electronics', 'User A'): Updated Relation = 0.6022429326448797
Epoch 1, Context ('Clothing', 'User A'): Updated Relation = 0.8427393200908393
Epoch 1, Context ('Morning', 'User B'): Updated Relation = 0.38836161547361087
Epoch 1, Context ('Evening', 'User B'): Updated Relation = 0.5714883873309156
Epoch 1, Context ('Electronics', 'User B'): Updated Relation = 0.7459433974021042
Epoch 1, Context ('Clothing', 'User B'): Updated Relation = 0.7451791854181026
Epoch 1, Context ('Evening', 'Morning'): Updated Relation = 0.42808583733914657
Epoch 1, Context ('Electronics', 'Morning'): Updated Relation = 0.8453870765210174
Epoch 1, Context ('Clothing', 'Morning'): Updated Relation = 0.2424569019458938
Epoch 1, Context ('Electronics', 'Evening'): Updated Relation = 0.29962257314725954
Epoch 1, Context ('Clothing', 'Evening'): Updated Relation = 0.8805183307445195
Epoch 1, Context ('Clothing', 'Electronics'): Updated Relation = 0.7137430720315634
Epoch 2, Context ('User A', 'User B'): Updated Relation = 0.8630603084907054
Epoch 2, Context ('Morning', 'User A'): Updated Relation = 0.4255858588654966
Epoch 2, Context ('Evening', 'User A'): Updated Relation = 0.8421848164817587
Epoch 2, Context ('Electronics', 'User A'): Updated Relation = 0.6012758489349457
Epoch 2, Context ('Clothing', 'User A'): Updated Relation = 0.8366975304262835
Epoch 2, Context ('Morning', 'User B'): Updated Relation = 0.3848762122781809
Epoch 2, Context ('Evening', 'User B'): Updated Relation = 0.5799735873664399
Epoch 2, Context ('Electronics', 'User B'): Updated Relation = 0.7433559674647434
Epoch 2, Context ('Clothing', 'User B'): Updated Relation = 0.7533008998966462
Epoch 2, Context ('Evening', 'Morning'): Updated Relation = 0.42999429888177604
Epoch 2, Context ('Electronics', 'Morning'): Updated Relation = 0.849728616111259
Epoch 2, Context ('Clothing', 'Morning'): Updated Relation = 0.23487654802323107
Epoch 2, Context ('Electronics', 'Evening'): Updated Relation = 0.30051152585395596
Epoch 2, Context ('Clothing', 'Evening'): Updated Relation = 0.8708573832613186
Epoch 2, Context ('Clothing', 'Electronics'): Updated Relation = 0.7042549987850627
Epoch 3, Context ('User A', 'User B'): Updated Relation = 0.8541034386893912
Epoch 3, Context ('Morning', 'User A'): Updated Relation = 0.4252809450162066
Epoch 3, Context ('Evening', 'User A'): Updated Relation = 0.8383713622275838
Epoch 3, Context ('Electronics', 'User A'): Updated Relation = 0.6008179756568776
Epoch 3, Context ('Clothing', 'User A'): Updated Relation = 0.8312282339265338
Epoch 3, Context ('Morning', 'User B'): Updated Relation = 0.379039353902383
Epoch 3, Context ('Evening', 'User B'): Updated Relation = 0.5792941467695948
Epoch 3, Context ('Electronics', 'User B'): Updated Relation = 0.7405924654959049
Epoch 3, Context ('Clothing', 'User B'): Updated Relation = 0.7618608266139367
Epoch 3, Context ('Evening', 'Morning'): Updated Relation = 0.4391621386402753
Epoch 3, Context ('Electronics', 'Morning'): Updated Relation = 0.8443061628092288
Epoch 3, Context ('Clothing', 'Morning'): Updated Relation = 0.23040646984937246
Epoch 3, Context ('Electronics', 'Evening'): Updated Relation = 0.3061499596688431
Epoch 3, Context ('Clothing', 'Evening'): Updated Relation = 0.8771628489211561
Epoch 3, Context ('Clothing', 'Electronics'): Updated Relation = 0.7003555766503761
Epoch 4, Context ('User A', 'User B'): Updated Relation = 0.8468528742579297
Epoch 4, Context ('Morning', 'User A'): Updated Relation = 0.41904532558366514
Epoch 4, Context ('Evening', 'User A'): Updated Relation = 0.8429945631589859
Epoch 4, Context ('Electronics', 'User A'): Updated Relation = 0.6099239843419092
Epoch 4, Context ('Clothing', 'User A'): Updated Relation = 0.8361229596531802
Epoch 4, Context ('Morning', 'User B'): Updated Relation = 0.381655518320864
Epoch 4, Context ('Evening', 'User B'): Updated Relation = 0.5755296602383004
Epoch 4, Context ('Electronics', 'User B'): Updated Relation = 0.7318615483478358
Epoch 4, Context ('Clothing', 'User B'): Updated Relation = 0.7667151877650277
Epoch 4, Context ('Evening', 'Morning'): Updated Relation = 0.4351222361510509
Epoch 4, Context ('Electronics', 'Morning'): Updated Relation = 0.8530325998333091
Epoch 4, Context ('Clothing', 'Morning'): Updated Relation = 0.23264228808139234
Epoch 4, Context ('Electronics', 'Evening'): Updated Relation = 0.30868466594900607
Epoch 4, Context ('Clothing', 'Evening'): Updated Relation = 0.8853909453615177
Epoch 4, Context ('Clothing', 'Electronics'): Updated Relation = 0.6954505205419266
Epoch 5, Context ('User A', 'User B'): Updated Relation = 0.841012217663016
Epoch 5, Context ('Morning', 'User A'): Updated Relation = 0.42565742830025294
Epoch 5, Context ('Evening', 'User A'): Updated Relation = 0.8451089301867453
Epoch 5, Context ('Electronics', 'User A'): Updated Relation = 0.6001133757500833
Epoch 5, Context ('Clothing', 'User A'): Updated Relation = 0.8352503855076191
Epoch 5, Context ('Morning', 'User B'): Updated Relation = 0.37587993137222986
Epoch 5, Context ('Evening', 'User B'): Updated Relation = 0.5853995225815045
Epoch 5, Context ('Electronics', 'User B'): Updated Relation = 0.733068407032253
Epoch 5, Context ('Clothing', 'User B'): Updated Relation = 0.7586728078234615
Epoch 5, Context ('Evening', 'Morning'): Updated Relation = 0.4385946269516262
Epoch 5, Context ('Electronics', 'Morning'): Updated Relation = 0.8446765466414582
Epoch 5, Context ('Clothing', 'Morning'): Updated Relation = 0.22396415248575322
Epoch 5, Context ('Electronics', 'Evening'): Updated Relation = 0.31514808430916963
Epoch 5, Context ('Clothing', 'Evening'): Updated Relation = 0.8944640381692195
Epoch 5, Context ('Clothing', 'Electronics'): Updated Relation = 0.6923230343888988
Epoch 6, Context ('User A', 'User B'): Updated Relation = 0.849398703601433
Epoch 6, Context ('Morning', 'User A'): Updated Relation = 0.4309299817826394
Epoch 6, Context ('Evening', 'User A'): Updated Relation = 0.8391155045306246
Epoch 6, Context ('Electronics', 'User A'): Updated Relation = 0.6003942052626907
Epoch 6, Context ('Clothing', 'User A'): Updated Relation = 0.8440350755370801
Epoch 6, Context ('Morning', 'User B'): Updated Relation = 0.3854937363855855
Epoch 6, Context ('Evening', 'User B'): Updated Relation = 0.5832004101912455
Epoch 6, Context ('Electronics', 'User B'): Updated Relation = 0.7302301975627368
Epoch 6, Context ('Clothing', 'User B'): Updated Relation = 0.7616327807341211
Epoch 6, Context ('Evening', 'Morning'): Updated Relation = 0.44646614429953074
Epoch 6, Context ('Electronics', 'Morning'): Updated Relation = 0.8521262437433271
Epoch 6, Context ('Clothing', 'Morning'): Updated Relation = 0.21765579936770152
Epoch 6, Context ('Electronics', 'Evening'): Updated Relation = 0.3094694427706882
Epoch 6, Context ('Clothing', 'Evening'): Updated Relation = 0.8909262152637084
Epoch 6, Context ('Clothing', 'Electronics'): Updated Relation = 0.7004497164413841
Epoch 7, Context ('User A', 'User B'): Updated Relation = 0.8451341961300554
Epoch 7, Context ('Morning', 'User A'): Updated Relation = 0.43875674316273394
Epoch 7, Context ('Evening', 'User A'): Updated Relation = 0.8367647441518986
Epoch 7, Context ('Electronics', 'User A'): Updated Relation = 0.599229425240784
Epoch 7, Context ('Clothing', 'User A'): Updated Relation = 0.8489128772032429
Epoch 7, Context ('Morning', 'User B'): Updated Relation = 0.3864317942693465
Epoch 7, Context ('Evening', 'User B'): Updated Relation = 0.5824674623700814
Epoch 7, Context ('Electronics', 'User B'): Updated Relation = 0.7370551758690626
Epoch 7, Context ('Clothing', 'User B'): Updated Relation = 0.7700583607740423
Epoch 7, Context ('Evening', 'Morning'): Updated Relation = 0.44436237594260103
Epoch 7, Context ('Electronics', 'Morning'): Updated Relation = 0.8471005395370764
Epoch 7, Context ('Clothing', 'Morning'): Updated Relation = 0.22743881308459465
Epoch 7, Context ('Electronics', 'Evening'): Updated Relation = 0.3065331761196822
Epoch 7, Context ('Clothing', 'Evening'): Updated Relation = 0.8877514398215273
Epoch 7, Context ('Clothing', 'Electronics'): Updated Relation = 0.6996188199038673
Epoch 8, Context ('User A', 'User B'): Updated Relation = 0.8512598651832605
Epoch 8, Context ('Morning', 'User A'): Updated Relation = 0.43190096140660994
Epoch 8, Context ('Evening', 'User A'): Updated Relation = 0.8400808717502124
Epoch 8, Context ('Electronics', 'User A'): Updated Relation = 0.6086437032784653
Epoch 8, Context ('Clothing', 'User A'): Updated Relation = 0.8574894136726197
Epoch 8, Context ('Morning', 'User B'): Updated Relation = 0.3780558430559213
Epoch 8, Context ('Evening', 'User B'): Updated Relation = 0.5910382618717536
Epoch 8, Context ('Electronics', 'User B'): Updated Relation = 0.745956635299797
Epoch 8, Context ('Clothing', 'User B'): Updated Relation = 0.761905703229806
Epoch 8, Context ('Evening', 'Morning'): Updated Relation = 0.4509167527404583
Epoch 8, Context ('Electronics', 'Morning'): Updated Relation = 0.8386193005932574
Epoch 8, Context ('Clothing', 'Morning'): Updated Relation = 0.2366719705210751
Epoch 8, Context ('Electronics', 'Evening'): Updated Relation = 0.30388990149961503
Epoch 8, Context ('Clothing', 'Evening'): Updated Relation = 0.8878844146695026
Epoch 8, Context ('Clothing', 'Electronics'): Updated Relation = 0.7036385148948182
Epoch 9, Context ('User A', 'User B'): Updated Relation = 0.8610157838983211
Epoch 9, Context ('Morning', 'User A'): Updated Relation = 0.43662233470046413
Epoch 9, Context ('Evening', 'User A'): Updated Relation = 0.8380634634243447
Epoch 9, Context ('Electronics', 'User A'): Updated Relation = 0.6060210864512084
Epoch 9, Context ('Clothing', 'User A'): Updated Relation = 0.8644104292086507
Epoch 9, Context ('Morning', 'User B'): Updated Relation = 0.3865644451426535
Epoch 9, Context ('Evening', 'User B'): Updated Relation = 0.591744633633021
Epoch 9, Context ('Electronics', 'User B'): Updated Relation = 0.7422403292054376
Epoch 9, Context ('Clothing', 'User B'): Updated Relation = 0.7575906782587816
Epoch 9, Context ('Evening', 'Morning'): Updated Relation = 0.4524828504650883
Epoch 9, Context ('Electronics', 'Morning'): Updated Relation = 0.8442205883785794
Epoch 9, Context ('Clothing', 'Morning'): Updated Relation = 0.23069411147329483
Epoch 9, Context ('Electronics', 'Evening'): Updated Relation = 0.30084570716043196
Epoch 9, Context ('Clothing', 'Evening'): Updated Relation = 0.8913664079651779
Epoch 9, Context ('Clothing', 'Electronics'): Updated Relation = 0.7049705459211154
```

---

### 🏆 Best Learned Relation

- **📌 Best Relation: 0.8913664079651779**
- **🔗 Context Pair: ('Clothing', 'Evening')**

---

## 🛠️ Installation

```bash
git clone https://github.com/Devansh-567/contextual-relationship-learning-algorithm.git
cd contextual-relationship-learning-algorithm
python app.py
```
