# **Movie Recommendation Algorithm**

A project to implement a movie recommendation algorithm that provides personalized movie suggestions based on user preferences. The recommendation system is accessible via a web interface.

> Archived student project. Built as an early recommendation-systems project to explore collaborative filtering, content-based filtering, and Flask web apps. Not actively maintained.

---

## **Project Description**

This project implements a movie recommendation system that suggests movies to users based on their interactions (likes and dislikes). The system uses:
- **Collaborative Filtering**: Recommends movies based on what similar users liked.
- **Content-Based Filtering**: Recommends movies similar to those the user has liked based on genres.
- The web interface allows users to:
  - Search for a movie and get recommendations.
  - Interact with random movie lists by liking or disliking movies.
  - View personalized recommendations.

---

## **Dataset and API Acknowledgment**

### **MovieLens Dataset**
This project uses the **MovieLens** dataset provided by **GroupLens** at the **University of Minnesota**. The dataset includes millions of movie ratings by real users and is a standard for movie recommendation research.

You can find the dataset and further information [here](https://grouplens.org/datasets/movielens/).

To run the project, please download the dataset and place it in the `data/raw/` directory or follow instructions under **Download the Dataset**.

**Citation**:  
F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context*. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19 (December 2015), 19 pages. DOI: [http://dx.doi.org/10.1145/2827872](http://dx.doi.org/10.1145/2827872)

---

### **OMDb API**
This project also integrates the **OMDb API** to fetch movie metadata and posters. The OMDb API is an open movie and TV database that provides movie details such as title, year, genre, and poster.

- **Acknowledgment**: The inclusion of OMDb API data in this application does not imply endorsement by OMDb. Please refer to their terms of use for more details: [https://www.omdbapi.com/](https://www.omdbapi.com/).
- You must obtain an API key from [OMDb API](https://www.omdbapi.com/) and add it to your environment variables or `utils.py` as `OMDB_API_KEY`.

---

## **Download the Dataset**

To download the MovieLens dataset automatically, run:
```bash
python download_data.py
```
Ensure the downloaded files are placed in the data/raw/ directory.

---

## **Dependencies**

Install the required packages using:
```bash
pip install -r requirements.txt
```

### **Key Dependencies:**
- **Flask**: For the web application.
- **pandas**: For data manipulation.
- **numpy**: For numerical computations.
- **scikit-learn**: For collaborative filtering.
- **OMDb API requests**: For fetching movie posters.
- **jinja2**: For rendering HTML templates.

---

## **Run the Application**

To start the Flask application, run:
```bash
python app.py
```

## **User Instructions**

1. **Homepage (`/`)**:
   - Search for a movie and get recommendations based on it.
   - Explore random movies and provide feedback (like or dislike).
   - Click "Get Recommendations" to receive personalized recommendations.

2. **Results Page (`/results`)**:
   - Displays recommendations based on the entered movie title.
   - Provides a grid layout of recommended movies with posters.

3. **Personalized Recommendations (`/recommend`)**:
   - Displays a list of movies based on the user's preferences.
   - Ensures disliked movies are excluded and results are shuffled for variety.

---

## **Project Structure**

```
├── app.py                 # Main Flask application
├── module/
│   ├── data_prep.py        # Functions for loading and processing data
│   ├── collaborative.py    # Collaborative filtering functions
│   ├── content_based.py    # Content-based filtering functions
│   ├── utils.py            # Utility functions (e.g., OMDb API requests)
├── templates/
│   ├── base.html           # Base template for all HTML pages
│   ├── index.html          # Homepage template
│   ├── results.html        # Results page template for search-based recommendations
│   └── personalized_results.html  # Page for personalized recommendations
├── static/
│   ├── style.css           # Custom styles for the web application
│   └── images/
│       └── default-poster.jpg  # Default poster for movies without images
├── data/
│   └── raw/                # Folder for MovieLens dataset files
└── requirements.txt        # List of dependencies
```

## **Acknowledgments**

- **MovieLens Dataset**: Provided by GroupLens at the University of Minnesota.
- **OMDb API**: For movie metadata and poster images.

---

## **Limitations**

- User likes and dislikes are stored in process memory and are not persisted per user.
- The OMDb API key must be provided locally.
- The app uses the MovieLens small dataset and is intended for demonstration, not production use.

---

This repository is kept public as an archived learning project.
