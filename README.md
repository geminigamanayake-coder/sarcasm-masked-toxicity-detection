# Sarcasm-Masked Toxicity Detection in Social Media Comments

## Overview

This research project investigates the detection of toxic language concealed through sarcasm in social media comments discussing American tariffs.

The study compares a transformer-based model, DistilBERT, with traditional machine learning classifiers to examine how effectively different approaches identify context-dependent toxic language.

## Research Objectives

- Investigate the challenges of detecting toxicity expressed through sarcasm.
- Compare transformer-based and traditional machine learning approaches.
- Evaluate model performance using classification metrics, including macro F1-score and class-specific F1-scores.
- Explore the potential of context-aware NLP methods for social media content moderation.

## Dataset

The research dataset contains 297 manually labelled social media comments collected from TikTok, Instagram, and YouTube concerning American tariffs.

The study includes toxicity, sarcasm, and sentiment labels.

Dataset availability is subject to applicable research ethics, privacy, platform terms, and redistribution permissions.

## Models Evaluated

- DistilBERT
- Support Vector Machine (SVM)
- Logistic Regression
- Random Forest
- XGBoost

## Reported Findings

The paper reports a macro F1-score of 0.65 for DistilBERT and 0.65 for XGBoost. DistilBERT reports an F1-score of 0.67 for the sarcastic-toxic subset.

These results should be interpreted in the context of the dataset's limited size and topic-specific scope.

## Technologies

- Python
- Natural Language Processing (NLP)
- scikit-learn
- Hugging Face Transformers
- PyTorch
- XGBoost
- Data preprocessing and model evaluation

## Limitations

The dataset is small and focuses on discussions about American tariffs. The findings may not generalise to other topics, platforms, or types of online communication.

## Publication

**Paper:** Evaluating Transformer-Based and Traditional Machine Learning Models for Sarcasm-Masked Toxicity Detection in Social Media Comments on American Tariffs

**Authors:** GGI Gamanayake, MGKS Gamage, and DD Weerasuriya

**Publication:** [Read the published extended abstract] https://irc.kdu.ac.lk/2026/abstract-books/pdfs/Abstract%20book%20-%20Management,%20Social%20Sciences%20and%20Humanities.pdf 
## Reproducibility

Installation instructions, dataset access details, preprocessing steps, and model execution instructions will be documented based on the available code and permitted data.
