# Federated UNet3+ Polyp Detection

This project implements a **federated learning framework for polyp segmentation** using the **UNet3+ architecture** on the **Kvasir-SEG dataset**. The system is designed to simulate decentralized medical data training while preserving privacy across multiple clients.

---

## Project Overview

- Task: Medical Image Segmentation
- Model: UNet3+
- Learning Paradigm: Federated Learning
- Dataset: Kvasir-SEG Polyp Dataset
- Goal: Improve polyp detection while preserving data privacy

---

## Dataset

We use the **Kvasir-SEG dataset**, which contains annotated gastrointestinal polyp images for segmentation tasks.

Dataset link:
https://datasets.simula.no/kvasir-seg/

Dataset includes:
- Endoscopic images
- Pixel-level segmentation masks

---

## Federated Learning Setup

The project simulates multiple clients, where:
- Each client trains locally on its own subset of data
- Model updates are aggregated on a central server
- Data never leaves local clients (privacy-preserving training)

---

## Model Architecture

We use **UNet3+**, which enhances traditional U-Net with:
- Full-scale skip connections
- Multi-level feature fusion
- Improved segmentation of small and complex structures

---

## Project Structure

