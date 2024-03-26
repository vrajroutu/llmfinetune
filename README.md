# Phi-2 Journal Fine-tuning with QLoRA

This repository contains the code and instructions for fine-tuning the Phi-2 language model using QLoRA (Quantization-Based Low-Rank Adaptation) for generating journal entries based on personal notes.

## Introduction

Phi-2 Journal Fine-tuning with QLoRA is a project aimed at leveraging advanced natural language processing techniques to generate journal entries based on personal notes. By fine-tuning the Phi-2 language model with QLoRA, we aim to create a model that can understand the context of personal notes and generate meaningful journal entries.


## QLora Flow

The LoRA method proposed by Hu et al. replaces to decompose the weight changes, ΔW, into a lower-rank representation. To be precise, it does not require to explicitly compute ΔW. Instead, LoRA learns the decomposed representation of ΔW directly during training which is where the savings are coming from, as shown in the figure below.

![QLora Workflow](QLora.png)

## Table of Contents

1. [Preparing Data](#1-preparing-data)
2. [Setting Up GPU Environment](#2-set-up-gpu-environment)
3. [Loading Dataset](#3-load-dataset)
4. [Loading Base Model](#4-load-base-model)
5. [Tokenization](#5-tokenization)
6. [Setting Up LoRA](#6-set-up-lora)
7. [Running Training](#7-run-training)
8. [Trying the Trained Model](#8-try-the-trained-model)

## 1. Preparing Data

Before starting the fine-tuning process, ensure your data is formatted correctly. The dataset should consist of JSONL files containing input-output pairs or structured data suitable for training. Use the provided script to preprocess your data into the required format.

## 2. Setting Up GPU Environment

Utilize a GPU environment for training the model efficiently. Instructions are provided for setting up the environment using Brev.dev, which offers GPU instances suitable for deep learning tasks.

## 3. Loading Dataset

Load the training and evaluation datasets using the `datasets` library. Ensure the data is formatted properly and define a formatting function to structure training examples as prompts.

## 4. Loading Base Model

Load the Phi-2 base model with 8-bit quantization enabled for memory-efficient training. This step initializes the model for fine-tuning.

## 5. Tokenization

Set up the tokenizer for tokenizing the input data. Define the maximum length for input tensors based on the distribution of dataset lengths. Tokenize the data with padding and truncation as necessary.

## 6. Setting Up LoRA

Prepare the model for fine-tuning with Low-Rank Adaptation (LoRA). Configure the LoRA settings such as rank (`r`) and scaling factor (`alpha`) to control the number of parameters and emphasize new fine-tuned data.

## 7. Running Training

Initiate the training process using the configured model, datasets, and training arguments. Monitor the training progress, evaluate the model, and save checkpoints at specified intervals.

## 8. Trying the Trained Model

Load the trained model from the best performing checkpoint directory. Test the model by providing sample prompts and generating journal entries based on the trained model's output.

---
