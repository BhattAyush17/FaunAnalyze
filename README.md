# Fauna-Network-Classifier

**Fauna-Network-Classifier** is a deep learning-powered image classification tool for reliably distinguishing between cats and dogs. Designed with scalability in mind, it will support broader animal recognition (and human detection) in future upgrades.

<img width="1736" height="919" alt="image" src="https://github.com/user-attachments/assets/015933b1-b616-4fb0-b51d-4c5c77ebc68c" />


---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Improvements](#improvements)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Overview

**Fauna-Network-Classifier** provides a modern, intuitive web interface for image classification using Streamlit. Initially, it offers robust and highly accurate recognition between cat and dog images, leveraging advanced deep learning and transfer learning techniques. The project is structured for easy extensibility to other animals and humans in future releases.

---

## Features

- **Accurate Cat vs Dog Classification**  
  Utilizes improved neural network architectures and transfer learning for reliable predictions.

- **Streamlit-Based UI**  
  Modern, glassy interface with custom fonts, gradients, and professional design.

- **Fun Facts & Feedback**  
  Displays fun pet facts and prediction confidence for each uploaded image.

- **Multi-Image Upload**  
  Supports batch classification of multiple images at once.

- **Scalable Architecture**  
  Easily upgradable to include other animal species and human detection.

- **Session Tracking**  
  Counts and displays the number of cats/dogs detected per session.

---

## Installation

1. **Clone the repository**
    ```bash
    git clone https://github.com/BhattAyush17/Fauna-Network-Classifier.git
    cd Fauna-Network-Classifier
    ```

2. **Install requirements**
    ```bash
    pip install -r requirements.txt
    ```

3. **Download/Place the Model**
    - Place your trained model file (`cat_vs_dog_model.keras`) in the repo root.
    - For future upgrades, see [Model Details](#model-details).

---

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

- **Upload**: Drag and drop or browse to select jpg/png images.
- **View Results**: See predictions, fun facts, and session statistics.
- **Future Upgrades**: For other animals/humans, updated models will be supported.

- <img width="1350" height="898" alt="image" src="https://github.com/user-attachments/assets/98fd2ce7-2fcf-4df6-92d2-48cd6419f4d8" />


---

## Model Details

- **Current Model**:  
  Uses transfer learning (e.g., VGG16, ResNet, MobileNet) fine-tuned on high-quality cat/dog datasets for robust recognition.
- **Data Augmentation**:  
  Rotation, flipping, zoom, and color jitter for better generalization.
- **Prediction Output**:  
  Displays class (cat or dog), confidence, and a related fun fact.

---

## Improvements

**Recent Enhancements:**
- Upgraded to transfer learning for stronger recognition.
- Improved UI: Professional glassy buttons, custom fonts, and responsive layout.
- Added batch image support and session statistics.

**Recognition Accuracy:**  
- Significantly reduced misclassification on varied and challenging cat/dog images.
- Handles diverse backgrounds, angles, and lighting.

---

## Roadmap

- **Next Release:**
  - Add support for other animals (birds, horses, etc.).
  - Human detection/classification.
  - Optional breed identification for cats/dogs.
  - Model selection and confidence visualization.
  - REST API support for external integration.

- **Long-Term:**
  - Expand dataset for multi-species recognition.
  - Mobile and desktop app versions.
  - Automated retraining pipeline.

---

## Contributing

1. Fork this repo.
2. Create your feature branch (`git checkout -b feature/NewFeature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/NewFeature`)
5. Open a pull request!

We welcome improvements to UI, model architecture, dataset expansion, and documentation.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [Streamlit](https://streamlit.io/) for the fast and beautiful web UI.
- [TensorFlow](https://www.tensorflow.org/) for deep learning support.
- [Kaggle Cat & Dog datasets](https://www.kaggle.com/c/dogs-vs-cats/data)
- The open-source community for inspiration and code.

---

**Fauna-Network-Classifier: Reliable pet and animal image classification, ready to scale for the future.**
