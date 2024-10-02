# Jupiter Notebook execution guide

This project contains a set of Jupyter notebooks (`.ipynb` files) that perform adversarial attacks. This guide outlines the steps to set up the environment and execute the notebooks.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Running the Notebooks](#running-the-notebooks)
4. [Notebook Descriptions](#notebook-descriptions)

## Prerequisites

Before you can run the notebooks, ensure you have the following installed on your system:

- **Python** (version 3.7 or above) - [Python Installation Guide](https://www.python.org/downloads/)
- **Jupyter Notebook** or **JupyterLab** - [Jupyter Installation Guide](https://jupyter.org/install)
- Recommended: **Git** - [Git Installation Guide](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)

## Installation

1. **Set up a Virtual Environment (optional but recommended):**
    ```bash
    python -m venv ipynb_venv
    source ipynb_venv/bin/activate  # On Windows use `ipynb_venv\Scripts\activate`
    ```
    
2. **Clone the Repository** (if applicable):
    ```bash
    git clone https://github.com/un4rch/openpilot_AdvExamples.git
    cd openpilot_AdvExamples/attacks/
    ```

3. **Install the Required Python Packages:**
    The required dependencies are listed in the `requirements.txt` file. Install them by running:
    ```bash
    pip3 install -r requirements.txt
    ```

## Running the Notebooks

1. **Start Jupyter Notebook**:
    You can launch Jupyter Notebook from the terminal using the following command:
    ```bash
    jupyter notebook
    ```
    This will open the Jupyter Notebook interface in your default web browser by default in: [](localhost:8888).

2. **Navigate to the Folder**:
    In the Jupyter Notebook interface, navigate to the directory containing the `.ipynb` files and click on the notebook you wish to run.

3. **Run the Notebook**:
    Once the notebook is open, you can execute each cell sequentially by selecting the cell and pressing `Shift + Enter`. You can also run all cells at once by selecting **Kernel > Restart & Run All**.

## Notebook Descriptions

- `notebook1.ipynb`: Brief description of what this notebook does.
- `notebook2.ipynb`: Brief description of what this notebook does.
- [Add descriptions for each notebook in your project].
