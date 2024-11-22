# ASF SlowFast Network

## Project Structure
```
OpenSource
├── checkpoints/       # Saved model checkpoints
├── data/             # Data processing and loading
│   └── dataset.py    # Dataset implementation
├── log/              # Training logs
├── models/           # Model architecture
│   ├── base_modules.py    # Basic building blocks
│   └── slowfast_base.py   # SlowFast network implementation
├── result/           # Evaluation results
├── main.py          # Training and evaluation scripts
└── readme.md        # This file
```

## Requirements
Please see `requirements.txt` for a complete list of dependencies.

## Installation
1. Clone this repository:
```bash
git clone https://github.com/MagicalLiHua/ASFSlowFast.git
cd ASFSlowFast
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation

#### Download Example Data
1. Download the example dataset from our shared drive:
   - [Download Link]()
   
2. Extract the downloaded data:
```bash
unzip example_data.zip -d ./data/
```

#### Data Structure
The dataset should be organized as follows:
```
data/
├── datasets/
│   ├── class1/
│   │   ├── 1.npy
│   │   └── 2.npy
│   └── class2/
│       ├── 1.npy
│       └── 2.npy
```


### Training And Evaluation
To train and evaluation the model from scratch:
```bash
python main.py
```

## Model Architecture

The implementation is based on the SlowFast Networks architecture, which uses:
- A Slow pathway capturing spatial semantics
- A Fast pathway capturing motion dynamics
- Lateral connections between pathways

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- The implementation is based on the original SlowFast Networks paper
- Thanks to the PyTorch team for their excellent deep learning framework