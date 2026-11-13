# Third-party notices

This repository integrates or can optionally use the following third-party software and model families:

- PyTorch and Torchvision — tensor runtime, pretrained image encoders, and Faster R-CNN.
- OpenCV — image decoding, color histograms, HOG, DCT, and image overlays.
- Scikit-learn — nearest-neighbor retrieval, KMeans, PCA, and metrics support.
- Streamlit — web interface.
- Hugging Face Transformers — optional CLIP model/processor loading.
- FAISS — optional vector-search backend.
- FastAPI / Uvicorn — optional REST API.
- Pillow, NumPy, Pandas, python-dotenv, tqdm, and pytest — supporting libraries.

The project code is MIT-licensed, but third-party code, pretrained weights, and datasets remain governed by their own licenses and terms. The operator is responsible for checking those terms and ensuring that the image collection may be used for the intended purpose. Torchvision's pretrained-weight documentation specifically warns that model weights may have terms derived from their training datasets.
