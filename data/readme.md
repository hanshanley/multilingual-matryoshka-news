# Dataset Acquisition

This document outlines how to obtain the external corpora used alongside this project. Each source has its own license and access requirements—review the linked documentation before downloading or redistributing the data. We provide access to the synthetic versions of the SemEval 2022 Task 8 dataset that we created in our paper as well as the similarity labels for all articles [here](https://github.com/hanshanley/multilingual-matryoshka-news/blob/main/data/labels_semeval_2022_task_eight.json). 

## SemEval 2022 Task 8 – Synthetic Dataset
- **Overview:** We provide synthetic translations as well as synthetically created alterations of the original SemEval 2022 Task 8 dataset. Due to the size of this dataset, they must be downloaded from Zenodo. You may find the link here: https://zenodo.org/records/17220308

- **Labels:** Labels for the SemEval 2022 Task 8 dataset can be downloaed here. These labels are taken from the original SemEval 2022 Task 8 dataset. Each URL pair is given a granular similar score from Very Similar(0.75), Somewhat Similar (0.50), Somewhat Dissimilar (0.25) to Very Dissimilar (0.00).

- 

## SemEval 2022 Task 8 – Multilingual News Article Similarity
- **Overview:** Article pairs labelled for graded similarity across English, Spanish, Russian, Turkish, and Farsi news outlets. Details are in the shared task description (Lai et al., 2022) – https://aclanthology.org/2022.semeval-1.155.pdf.
- **Access:** The organisers only distribute metadata (URLs, similarity scores, splits) via the CodaLab competition page: http://www.euagendas.org/semeval2022. You must create a CodaLab account, join the competition, and accept the licence to download the official `.tsv` files (train/dev/test links and labels).
- **Scraping the articles:** Because the original news content cannot be redistributed, you will have to download these article and and the extract the contents. The original task organizers provide an Internet Archive scraper published at https://github.com/euagendas/semeval_8_2022_ia_downloader (PyPI package `semeval_8_2022_ia_downloader`).
  1. Create an isolated environment (`python3 -m venv venv && source venv/bin/activate`).
  2. Install the package: `pip install semeval_8_2022_ia_downloader`.
  3. Run the CLI with the provided metadata file: `python -m semeval_8_2022_ia_downloader.cli --links_file=training_links.tsv --dump_dir=downloads/`.
  4. The script saves each article as both `.html` (Internet Archive snapshot) and `.json` (newspaper3k extraction) under hash-based subdirectories, e.g. `downloads/89/0123456789.html`.
- **Notes:** Expect occasional 404/timeout errors from the Wayback Machine; rerun the downloader on the failed rows. Keep the raw `.tsv` files alongside the scraped content so you can map article IDs back to similarity labels.

## Miranda et al. 2018 – Multilingual Streaming News Clustering
- **Overview:** Clustering benchmark introduced by Miranda et al. (2018) – “Multilingual Clustering of Streaming News” (https://aclanthology.org/D18-1483.pdf). Contains anonymised news-stream segments with gold story IDs for English, Spanish, and German.
- **Access:** Priberam hosts the public release in the `news-clustering` repository: https://github.com/Priberam/news-clustering.
  1. Clone or download the repo to inspect the helper scripts.
  2. Run `download_data.sh` (requires `wget`) or execute the commands manually:
     - `wget -P dataset --user=anonymous --password=anonymous ftp://ftp.priberam.pt/SUMMAPublic/Corpora/Clustering/2018.0/dataset/dataset.dev.json`
     - `wget -P dataset --user=anonymous --password=anonymous ftp://ftp.priberam.pt/SUMMAPublic/Corpora/Clustering/2018.0/dataset/dataset.test.json`
     - `wget -P dataset --user=anonymous --password=anonymous ftp://ftp.priberam.pt/SUMMAPublic/Corpora/Clustering/2018.0/dataset-tok-ner/clustering.dev.json`
     - `wget -P dataset --user=anonymous --password=anonymous ftp://ftp.priberam.pt/SUMMAPublic/Corpora/Clustering/2018.0/dataset-tok-ner/clustering.test.json`

## 20 Newsgroups
- **Overview:** Classic topic-classification corpus with ~18k English Usenet posts across 20 categories. Documentation: https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html.
- **Access via scikit-learn:**
  ```python
  from sklearn.datasets import fetch_20newsgroups
  data = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"), download_if_missing=True)
  ```


