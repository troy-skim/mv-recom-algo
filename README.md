# Movie Recommendation Algorithm

A project to implement a movie recommendation algorithm. 

## Project Description

This project is a movie recommendation algorithm that recommends movies to users, provided some movie preferences. The algorithm will be accessible via web.

## Dataset and API Acknowledgment

### MovieLens Dataset
This project makes use of the MovieLens dataset. The dataset is provided by GroupLens at the University of Minnesota. You can find the dataset and further information [here](https://grouplens.org/datasets/movielens/).

To run the project, please download the dataset and place it in the `data/raw/` directory or follow instructions under **Download the Dataset**.

**Citation**:  
F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19 (December 2015), 19 pages. DOI: [http://dx.doi.org/10.1145/2827872](http://dx.doi.org/10.1145/2827872)

---

### OMDb API
This project also makes use of the OMDb API to fetch movie metadata and posters. The OMDb API is an open database for movies and television series. You can learn more about it [here](https://www.omdbapi.com/).

**Acknowledgment**:  
The inclusion of OMDb API data in this application does not imply endorsement by OMDb. Please refer to their terms of use for more details: [https://www.omdbapi.com/](https://www.omdbapi.com/).

---

## Download the Dataset

Run the following script to automatically download the dataset:
```bash
python download_data.py
```

## Depenencies
