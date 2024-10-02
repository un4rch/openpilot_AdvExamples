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
    This will open the Jupyter Notebook interface in your default web browser.
    IMPORTANT: Open the link from terminal with the access token. The link looks something like this:

   `[I 2024-10-02 11:48:36.274 ServerApp] http://localhost:8888/tree?token=580daa36dc6f3e1c85f34f073ab48f5b7906a88a67cb3afe`

3. **Navigate to the Folder**:
    In the Jupyter Notebook interface, navigate to the directory containing the `.ipynb` files and click on the notebook you wish to run.

4. **Run the Notebook**:
    Once the notebook is open, you can execute each cell sequentially by selecting the cell and pressing `Shift + Enter`. You can also run all cells at once by selecting **Kernel > Restart & Run All**.

## Notebook Descriptions

- `carlini_wagner_cifar_10.ipynb`: Carlini & Wagner algorithm implementation against CIFAR-10 dataset.
- `attack_whitebox.ipynb`: Openpilot white-box attack implementation against the Supercombo model.
