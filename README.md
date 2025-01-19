# EthioMart-NER-LLM

This project focuses on extracting named entities from Telegram messages, tokenizing the data, and preparing it for further analysis. It utilizes various Python scripts, notebooks, and testing frameworks to facilitate efficient data processing.


## Project Structure


```
├── notebooks
│   ├── 1.0.data-prepocessing-and-exploration.ipynb 
│   ├── 2.0-data-auto-labeling.ipynb
│   ├── README.md                 
│   ├── __init__.py               
│
├── README.md                   
├── requirements.txt            
├── scripts
│   ├── README.md                 
│   ├── __init__.py               
│   ├── preprocess_data.py               
│   ├── scrape.py            
│   ├── visualize.py           
│   ├── label_data.py           
│
├── src            
│   ├── __init__.py               
│
├── tests
│   ├── __init__.py  
│
├── data            
│   ├── processed
│       ├── processed_data.csv
│       ├── labeled_data.conll

```

## Installation

1. Clone the repository:
   ```bash
   git clone <repo_url>
   cd <project_directory>
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv/Scripts/activate`
   pip install -r requirements.txt
   ```

## Contribution

Feel free to fork the repository, make improvements, and submit pull requests.