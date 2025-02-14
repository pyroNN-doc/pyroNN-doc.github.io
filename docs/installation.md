#  Installation and Setup 🛠️

## Installation 📥

Pyro-NN works with **PyTorch** and **TensorFlow**. This guide focuses on **PyTorch**.  
### Requirements:
* `Microsoft Visual Studio (for self-build) 💻`
* `Microsoft Visual C++ 14.0+`
* `Python package `build` 🐍`
* `CUDA >10.2 🚀`

Once all of these requirements are fulfilled, follow these steps::

* Clone the repo and switch to the `torch` branch:
```bash
git checkout torch
```

* Run the command: 
```bash
python -m build . 
```

* Switch to the newly created sub-directory `dist`


* Build the wheel using the following command: 
```bash
pip install pyronn-(*version_number*).whl
```




!!! example "Note"

            If necessary, you can modify the `pyproject.toml` file to specify a 
            particular torch version. However, be cautious and only make changes 
            if you are confident in what you are doing!


