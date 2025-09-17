# Predicting Box Office Revenue Based on Early Rotten Tomatoes & Metacritic Scores

**Overview:**

This project investigates the correlation between early critical reception (Rotten Tomatoes and Metacritic scores) and a movie's box office revenue within the first two weeks of its release.  The goal is to build a predictive model that can leverage these readily available metrics to forecast box office performance, assisting in more strategic marketing budget allocation.  The analysis involves data cleaning, exploratory data analysis (EDA), model training, and evaluation to determine the predictive power of the chosen model.

**Technologies Used:**

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

**How to Run:**

1. **Install Dependencies:**  Ensure you have Python 3 installed. Then, navigate to the project directory in your terminal and install the required libraries using pip:

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Script:** Execute the main script using:

   ```bash
   python main.py
   ```

**Example Output:**

The script will print key statistical analysis to the console, including model performance metrics (e.g., R-squared, RMSE). Additionally, the script generates several visualization files (e.g., scatter plots showing the relationship between critic scores and box office revenue, and potentially model performance curves).  These plots are saved in the `output` directory.  Example output files include: `sales_trend.png` (example plot filename, may vary based on analysis).


**Directory Structure:**

* `data/`: Contains the input datasets.
* `src/`: Contains the source code for data processing, model training, and visualization.
* `output/`: Contains the generated output files (plots, etc.).
* `models/`: Contains saved models (if applicable).
* `requirements.txt`: Lists the project dependencies.
* `README.md`: This file.


**Contributing:**

Contributions are welcome! Please open an issue or submit a pull request.


**License:**

[Specify your license here, e.g., MIT License]