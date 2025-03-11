# Usage 🏗️💻

The methodology of **Pyro-NN** is contained within the **`ct_reconstruction`** folder, which is organized into four essential parts:

---

### 1. **🔧 Geometry** 

Defines the scanning parameters and trajectory. 

- **Initialization from parameters**: This is possible if you know all your scanning parameters and if the scanning trajectory was circular. 
- **⚠️ Small Tip**: Sometimes, parameters in the header can be incorrectly filled. Be aware—errors may still occur!

---

### 2. **🛠️ Layers** 

Defines the 2D and 3D forward/backward projectors. 

- **Geometry setup**: To initialize these layers, the **geometry** of the scan must be defined.
- **Input of all layers**: The input is the **Image-Tensor** (depending on the dimensionality) and a **geometry dictionary**, returned when the geometry is initialized.
  
  - **For 2D**: Includes implementations for **Parallel Beam** and **Fan Beam**.
  - **For 3D**: Implements **Cone Beam**.

---

### 3. **🔍 Helpers** 

Provides pre-implemented filters, weights, trajectories, and phantoms. 

- **Implemented Filters**: 
  - Ramp Filters, Ram Lak, Shepp Logan, Cosine, Hamming, Hann 🎛️
- **Implemented Weights**: 
  - Cosine, Parker ⚖️
- **Implemented Trajectories**: 
  - Circular and arbitrary paths 🌀

---

### 4. **⚙️ Cores** 

Contains the kernels and the PyTorch connection.

---

            Each part plays a crucial role in making Pyro-NN an efficient 
            and powerful framework for differentiable reconstruction. 🚀

