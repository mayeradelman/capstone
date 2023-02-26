from Data_Pipeline import data_pipeline
import matplotlib.pyplot as plt

# The code assumes it is run from within the same folder as hhd_cleaned+merged
directory = '.\\hhd_cleaned+merged\TRAIN'

X, y = data_pipeline(directory, 175, 2)

print(len(X))
print(len(y))

# X_reshaped = np.array([i.flatten() for i in X_train])

# np.save('X_Data.npy', X_reshaped)
# np.save('Y_Data.npy', y_train)

fig, ax = plt.subplots(nrows=3, ncols=9, sharex=True, sharey=True)
ax = ax.flatten()
for i in range(27):
  img = X[y == i][6]
  ax[i].imshow(img, cmap='Greys')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()