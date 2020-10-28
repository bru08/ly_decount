# %%
import h5py
import matplotlib.pyplot as plt
data = h5py.File("./training.h5", 'r')
# %%
X = data["x"]
y = data["y"]
X.shape
# %%
idx = 5
plt.imshow(X[idx])
plt.title(f"N Lymph: {y[idx]}")

# %%
