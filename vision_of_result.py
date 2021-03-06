import keras
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.metrics import confusion_matrix
#
# # === dataset ===
# with np.load('mnist.npz') as f:
#     x_train, y_train = f['x_train'], f['y_train']
#     x_test, y_test = f['x_test'], f['y_test']
#
# x_train = x_train.reshape(60000, 28, 28, 1)
# x_test = x_test.reshape(10000, 28, 28, 1)
# print(x_train.shape)
# print(x_test.shape)
#
# # === model: CNN ===
# model = keras.models.Sequential()
# model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# model.add(keras.layers.MaxPooling2D((2, 2)))
# model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(keras.layers.MaxPooling2D((2, 2)))
# model.add(keras.layers.Flatten())
# model.add(keras.layers.Dense(64, activation='relu'))
# model.add(keras.layers.Dense(10, activation='softmax'))
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
# model.summary()
#
# # === train ===
# model.fit(x=x_train, y=y_train,
#           batch_size=512,
#           epochs=10,
#           validation_data=(x_test, y_test))

# === pred ===
# y_pred = model.predict_classes(x_test)
# print(y_pred)
labels = ['covid', 'normal', 'pneumonia']
y_test = list(map(int, "2 0 0 0 1 0 0 0 2 2 2 0 1 0 2 0 2 2 0 2 2 0 0 0 0 0 0 2 0 0 0 0 1 0 0 0 2 0 2 0 2 0 2 2 2 2 0 2 0 0 0 0 0 2 2 0 2 0 0 1 1 0 2 0 0 1 2 0 2 0 0 2 2 2 0 2 0 1 0 2".split(" ")))
y_pred = list(map(int, "1 0 2 0 1 0 0 1 2 1 1 0 1 0 1 2 1 2 0 2 2 0 0 0 2 2 0 1 1 0 2 0 1 0 0 0 2 2 1 1 2 0 2 2 2 2 0 2 0 0 0 0 0 2 1 0 1 1 1 1 1 2 2 0 1 1 2 2 1 1 0 2 2 2 1 2 0 2 0 1".split(" ")))
# === 混淆矩阵：真实值与预测值的对比 ===
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
con_mat = confusion_matrix(y_test, y_pred)

con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]     # 归一化
con_mat_norm = np.around(con_mat_norm, decimals=2)

# === plot ===
plt.figure(figsize=(4, 4))
sns.heatmap(con_mat_norm, annot=True, cmap='Blues')
plt.yticks([0, 1, 2], labels)
plt.xticks([0, 1, 2], labels)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
# plt.show()
plt.savefig("5_2.jpg")