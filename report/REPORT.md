# Dataset

MovieLens dataset consists of the following files:

### movies.csv

**Size:** 58,098 x 3

**Data examples:**

| movieId | title                       | genres                                                          |
|---------|-----------------------------|-----------------------------------------------------------------|
| 1       | Toy Story (1995)            | Adventure&#124;Animation&#124;Children&#124;Comedy&#124;Fantasy |
| 144078  | Kleinhoff Hotel (1977)      | (no genres listed)                                              |
| 146854  | Dov'è la Libertà...? (1954) | Comedy                                                          |

**Genres:**

* Action
* Adventure
* Animation
* Children's
* Comedy
* Crime
* Documentary
* Drama
* Fantasy
* Film-Noir
* Horror
* Musical
* Mystery
* Romance
* Sci-Fi
* Thriller
* War
* Western
* (no genres listed)

### ratings.csv

**Size:** 27,753,444 x 4

**Data examples:**

| userId | movieId | rating | timestamp  |
|--------|---------|--------|------------|
| 4      | 296     | 5.0    | 1113767056 |
| 65835  | 134853  | 0.5    | 1468203075 |

Ratings are made on a 5-star scale with half-star increments (0.5 stars - 5.0 stars).

### tags.csv

**Size:** 1,108,997 x 4

**Data examples:**

| userId | movieId | tag                       | timestamp  |
|--------|---------|---------------------------|------------|
| 14     | 110     | epic                      | 1443148538 |
| 50924  | 46976   | Dustin Hoffman            | 1248086227 |
| 152473 | 2551    | twins/inter-related lives | 1264325551 |
| 193831 | 175     | ChloÃ« Sevigny            | 1388598582 |
| 240383 | 106749  | seen more than once       | 1500708928 |
| 271131 | 64993   | 01/10                     | 1264325551 |

### genome-tags.csv

**Size:** 1,128 x 2

**Data examples:**

| tagId | tag                                         |
|-------|---------------------------------------------|
| 14    | 9/11                                        |
| 339   | easily confused with other movie(s) (title) |
| 418   | funny as hell                               |
| 601   | life & death                                |
| 754   | oscar (best music - original score)         |

The tag genome is a data structure that contains tag relevance scores for movies. The structure is a dense matrix: each
movie in the genome has a value for *every* tag in the genome.

As described in [this article][genome-paper] the tag genome encodes how strongly movies exhibit particular properties
represented by tags (atmospheric thought-provoking realistic etc.). The tag genome was computed using a machine learning
algorithm on user-contributed content including tags ratings and textual reviews.

[genome-paper]: http://files.grouplens.org/papers/tag_genome.pdf

### genome-scores.csv

**Size:** 14,862,528 x 3

**Data examples:**

| movieId | tagId | relevance            |
|---------|-------|----------------------|
| 1       | 1     | 0.029000000000000026 |
| 55159   | 603   | 0.657                |

# Goal

Build a recommender system.

# Workflow

## Dataset inspection

Script:

```shell
venv/Scripts/python.exe dataset_overview.py
```

### movies.csv

Brief info:

```text
Movies count: 58098
Movies with unique ID count: 58098
Movies with unique title count: 58020
Min movie ID: 1
Max movie ID: 193886
Genres (20): {'Thriller', 'Romance', 'Musical', 'Adventure', 'Children', 'War', 'Sci-Fi', 'Animation', 'Action', 'IMAX',
'Mystery', '(no genres listed)', 'Drama', 'Comedy', 'Horror', 'Film-Noir', 'Fantasy', 'Crime', 'Documentary', 'Western'}
```

Notes:

* Extra genre: IMAX. There are several movies with it in the dataset.
* Titles repeat, but it won't affect results because it's very rare.

### ratings.csv

Brief info:

```text
Unique scores: [0.5 1.  1.5 2.  2.5 3.  3.5 4.  4.5 5. ]
Number of unique movies: 53889
```

Notes:

* At least 7.2% of movies don't have ratings

### tags.csv

Brief info:

```text
Number of unique tags (case insensitive): 66981
Number of unique movies: 45981
```

Notes:

* At least 20.8% of movies don't have tags.
* Timestamp is irrelevant.

### genome-tags.csv

Brief info:

```text
Number of tags absent in ratings.csv (case insensitive): 0
```

Notes:

* all genome tags are present in ratings.csv if comparing case-insensitively.

### genome-scores.csv

Brief info:

```text
Unique movie number: 13176
Number of scores with tag ids absent in genome-tags.csv: 0
Min relevance: 0.0002499999999999
Max relevance: 1.0
```

Genome score distribution:

![img.png](img.png)

Notes:

* At least 77.3% of movies don't have genome scores.
* All genome scores have string representation
* All genome scores' string representations are present in ratings.csv if comparing case-insensitively.

## Plan

For most movies we have the following info that can help us make recommendations:

* Genre
* Ratings
* Tags

Some tags have genome scores. Also, we can add _average rating_ feature for movies.

For users, we have only:

* Ratings

Additionally, we can add _average genre rating_ feature.

We will try to take advantage of all of this data, but progressively. The general plan is:

1. Build a **collaborative filtering** recommender. This system will learn features based solely on user ratings. These
   features will help us later, hopefully.
2. Build a **content-based filtering** recommender using one-hot encoded genre, average rating and user's average genre
   rating info.
3. Build a **tag embeddings** recommender. We will find pre-trained embeddings on the internet. Then we will represent
   every movie as a normalized weighted sum of its tags' embeddings. To do a recommendation, we will take a weighted sum
   of all movie embeddings, that current user has rated, and find movies that are closest to resulting vector.
4. **Combine the recommenders** listed above into a single content-based recommender.

# References

* Jesse Vig Shilad Sen and John Riedl. 2012. The Tag Genome: Encoding Community Knowledge to Support Novel Interaction.
  ACM Trans. Interact. Intell. Syst. 2 3: 13:1–13:44. <https://doi.org/10.1145/2362394.2362395>
