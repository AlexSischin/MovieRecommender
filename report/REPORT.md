# Dataset

MovieLens dataset consists of the following files:

### movies.csv

**Size:** 58,871 x 3

**Data examples:**

| movieId | title                       | genres                                                          |
|---------|-----------------------------|-----------------------------------------------------------------|
| 1       | Toy Story (1995)            | Adventure&#124;Animation&#124;Children&#124;Comedy&#124;Fantasy |
| 144078  | Kleinhoff Hotel (1977)      | (no genres listed)                                              |
| 146854  | Dov'è la Libertà...? (1954) | Comedy                                                          |

Genres:

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

**Size:** 1,129 x 2

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

Build a recommendation system.

# References

* Jesse Vig Shilad Sen and John Riedl. 2012. The Tag Genome: Encoding Community Knowledge to Support Novel Interaction.
  ACM Trans. Interact. Intell. Syst. 2 3: 13:1–13:44. <https://doi.org/10.1145/2362394.2362395>
