# Surrogate-MPC-EPlus  
**Adaptive Insulation Framework with Multi-Objective Model Predictive Control for EnergyPlus**

---

## 📌 Overview

This project implements a **Surrogate-Based Model Predictive Control (MPC)** framework for building energy systems using:

- EnergyPlus Python Plugin  
- LSTM Neural Network (TensorFlow)  
- Multi-Objective Optimization (NSGA-II)  
- Real-time co-simulation control
- Adaptive Insulation

The framework replaces computationally expensive EnergyPlus evaluations with a trained **LSTM surrogate model**, enabling near real-time optimization of adaptive building envelope systems (e.g., movable insulation).

---

## 🧠 Concept

Instead of directly optimizing EnergyPlus (which is computationally expensive), this framework:
 
1. Uses a Trained LSTM surrogate model  
3. Uses NSGA-II for multi-objective optimization  
4. Applies optimal control actions during simulation  

This significantly reduces computational cost while maintaining predictive accuracy.

---

## 📦 Requirements

### Recommended Environment
- Python 3.8 (recommended for EnergyPlus compatibility)
- model.h5, scaler.pkl # Trained model with the scaler to inverse normalization - should be inside /python folder
- initial_data.csv # Data for the last 24 hours before your first time-step simulation or before your "prediction horizon" for lagged features - should be inside /python folder

### 🔧 Essential Libraries

Install the core dependencies:

```bash
pip install tensorflow==2.10
pip install numpy==1.26.4
pip install scikit-learn==1.4.2
pip install deap==1.4
pip install pandas matplotlib pymoo plotly seaborn
```
Other supporting libraries may be required depending on your environment, but the above are the essential dependencies.

---

# 🏗 EnergyPlus IDF Configuration

To use this controller, your IDF file must define the following components.

```idf
!- example of one surface with movable insulation repeat for all surface with an adaptive insulation
SurfaceControl:MovableInsulation,
    Outside,                  !- Insulation Type
    South,                    !- Surface Name
    Wall Insulation,          !- Material Name
    SIS sch;                  !- Schedule Name

Schedule:Constant,
    SIS sch,                  !- Name
    ,                         !- Schedule Type Limits Name
    0.5;                      !- Hourly Value

PythonPlugin:SearchPaths,
    Python38Library,         !- Name
    Yes,                     !- Add Current Working Directory to Search Path
    Yes,                     !- Add Input File Directory to Search Path
    Yes,                     !- Add epin Environment Variable to Search Path
    C:\Program Files\Python38\Lib\site-packages;  !- Search Path 1

PythonPlugin:Instance,
    Construction Controller, !- Name
    No,                      !- Run During Warmup Days
    MPC_INS,                 !- Python Module Name
    SetConstructionControlState;  !- Plugin Class Name
```

---

Here is your section properly formatted in clean GitHub-compatible Markdown (professional style, no extra commentary):

# Controller Functionality

The controller:

- Collects real-time zone variables  
- Updates surrogate input history  
- Runs NSGA-II multi-objective optimization  
- Minimizes:
  - Heating energy  
  - Cooling energy  
- Determines optimal insulation state  
- Applies control at each hour while optimizing for whole prediction horizon  

---

# License

## Academic / Non-Commercial Use

This project is licensed under:

- **Creative Commons CC-BY-NC 4.0**, or  
- **GNU AGPLv3**

### License Terms

- Free to use for research and personal projects  
- Free to modify for academic purposes  
- Citation is required in academic publications  
- Commercial use is **not permitted** under CC-BY-NC  
- Under AGPLv3, any modifications shared over a network must also be open source  

Please include proper citation if this framework is used in academic research.

---

# Disclaimer

This software is provided for research purposes only.  
No warranty is provided for commercial or safety-critical applications.


