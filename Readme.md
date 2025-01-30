# Robust Discriminant Vector Machine for Outlier Detection (DVM-OD)

This repository contains the source code accompanying the paper **"Robust Discriminant Vector Machine for Outlier Detection"**. The project includes implementations of our proposed DVM-OD algorithm, as well as baseline comparisons, synthetic data generation, and experimental workflows.

---

## Project Structure

- **Model/**: Contains the main implementation of the DVM-OD algorithm.
- **Results/**: Stores results and outputs generated during the experiments.
- **Data/**: Contains datasets (both real-world and synthetic) used for training and evaluation.
- **Notebooks/**: Contains the notebook version.
- **Code files:**

  1. **`create_synthetic_data.py`**: Generates synthetic datasets for experimentation. A notebook version (`create_synthetic_data.ipynb`) is also provided for easier experimentation.
  
  2. **`run_dvmod_realworld_data.py`**: Runs experiments with the DVM-OD algorithm on real-world datasets. The notebook equivalent is `run_dvmod_realworld_data.ipynb`.

  3. **`run_dvmod_synthetic_data.py`**: Runs experiments with the DVM-OD algorithm on synthetic datasets. The notebook equivalent is `run_dvmod_synthetic_data.ipynb`.

  4. **`run_baseline_realworld_data.py`**: Evaluates baseline algorithms on real-world datasets. The notebook equivalent is `run_baseline_realworld_data.ipynb`.

  5. **`run_baseline_synthetic_data.py`**: Evaluates baseline algorithms on synthetic datasets. The notebook equivalent is `run_baseline_synthetic_data.ipynb`.

---

## Requirements

- Python 3.8 or higher
- Required libraries:
  - NumPy
  - Pandas
  - Scikit-learn
  - Matplotlib
  - Pyod
  - Other dependencies as specified in `requirements.txt`

To install the required libraries, run:

```bash
pip install -r requirements.txt
```

---

## Usage

### Synthetic Data Generation
To create synthetic datasets, execute:

```bash
python create_synthetic_data.py
```
Alternatively, you can use the Jupyter notebook version (`create_synthetic_data.ipynb`) for an interactive approach.

### Running Experiments

#### 1. DVM-OD Algorithm
- **Real-world Data**:
  ```bash
  python run_dvmod_realworld_data.py
  ```
- **Synthetic Data**:
  ```bash
  python run_dvmod_synthetic_data.py
  ```

#### 2. Baseline Algorithms
- **Real-world Data**:
  ```bash
  python run_baseline_realworld_data.py
  ```
- **Synthetic Data**:
  ```bash
  python run_baseline_synthetic_data.py
  ```

#### Using Notebooks
For all the tasks mentioned above, Jupyter notebook versions are provided for easier visualization and experimentation.

---

## Results

Experimental results, including performance metrics and visualizations, are stored in the `Results/` directory. For detailed analyses, refer to the relevant sections of the paper.

---

## Citation
If you use this code in your research, please cite our paper:

```bibtex
@article{YourPaper,
  title={Robust Discriminant Vector Machine for Outlier Detection},
  author={YourName et al.},
  journal={Journal/Conference Name},
  year={2025},
  url={link_to_paper}
}
```

---

## License
This project is licensed under the [MIT License](LICENSE).

---

## Contact
For questions or feedback, please reach out to [Your Contact Information].
