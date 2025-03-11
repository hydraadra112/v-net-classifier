import streamlit as st
import torch
from torch import nn
from pathlib import Path
from PIL import Image
from streamlit_image_zoom import image_zoom
from torchvision.transforms import v2
from architecture import DR_Classifierv2
import os
import random

# Labels for classification
idx_labels = {0: 'Mild', 1: 'Moderate', 2: 'No DR', 3: 'Proliferate DR', 4: 'Severe'}
current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model loading
@st.cache_resource
def load_model() -> nn.Module:
    PATH = Path('./dataset/model_v2.pth')
    model = DR_Classifierv2(input_shape=3, output_shape=5, hidden_units=64)
    model.load_state_dict(torch.load(PATH, map_location=current_device))
    return model

# Preprocess images for prediction
def preprocess_image(img: Image) -> torch.Tensor:
    transform = v2.Compose([
        v2.PILToTensor(),
        v2.ToDtype(torch.float32),
        v2.Resize((224, 224)),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    return transform(img).unsqueeze(0).to(current_device)

def predict_class(model: nn.Module, img: torch.Tensor) -> str:
    if img.shape != (img.shape[0], 3, 224, 224):
        raise ValueError('Image is not the expected shape: [batch_size, 3, 224, 224]')
    
    model.eval()
    with torch.inference_mode():
        pred = model(img)
        predicted_class = torch.argmax(pred, dim=1).item()
    return idx_labels[predicted_class]

# Main app
def main():
    st.header('Diabetic Retinopathy Classifier')
    st.caption('Prepared by: John Manuel Carado | BSCS 3-A')
    
    # Tabs for navigation
    pred_tab, model_tab, data_tab = st.tabs(['Prediction', 'About the Model', 'About the Dataset'])
    
    with pred_tab:
        pytorch_model = load_model()
        
        # Upload an image
        uploaded_image = st.file_uploader('Upload a Fundus image for classification!', type=['png', 'jpg'])
        
        selected_image = None
        
        with st.expander('**OR choose from existing fundus images:**'):
            existing_images = {
            "No DR": Path('./dataset/colored_images/No_DR/0ae2dd2e09ea.png'),
            "Moderate": Path('./dataset/colored_images/Moderate/fd48cf452e9d.png'),
            "Proliferate DR": Path('./dataset/colored_images/Proliferate_DR/0e82bcacc475.png')
            }
            
            cols = st.columns(len(existing_images))
            for (label, img_path), col in zip(existing_images.items(), cols):
                with col:
                    st.image(img_path, caption=label)
                    if st.button(f'Select {label}'):
                        selected_image = img_path
        
        # Use uploaded image if provided, otherwise fallback to selected existing image
        image = uploaded_image or selected_image
        
        # Prediction and display
        if image:
            parsed_image = Image.open(image).convert("RGB")
            preprocessed_image = preprocess_image(parsed_image)
            classification = predict_class(pytorch_model, preprocessed_image)
            
            # Show zoomable image and classification
            image_zoom(parsed_image, mode="dragmove", size=(700, 500), keep_aspect_ratio=True, zoom_factor=2.0, increment=0.2)
            st.success(f'**Prediction ->** {classification}')
        else:
            st.warning("Please upload an image or select an existing one for prediction.")
            
            
    with model_tab:
        st.header('Model Performance')
        st.write('The DR images are trained in a basic CNN architecture for a multiclassification task.')
        with st.expander('Click to see PyTorch class architecture'):
            arc = """
                class DR_Classifierv2(nn.Module):
                    def __init__(self,  output_shape: int, input_shape: int = 3, hidden_units: int = 64):
                        super().__init__()

                        self.block1 = nn.Sequential(
                            nn.Conv2d(input_shape, hidden_units, kernel_size=3, padding='same'),
                            nn.LeakyReLU(0.1),
                            nn.BatchNorm2d(hidden_units),
                            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding='same'),
                            nn.LeakyReLU(0.1),
                            nn.BatchNorm2d(hidden_units),
                            nn.MaxPool2d(2),
                            nn.Dropout(0.3)
                        )

                        self.block2 = nn.Sequential(
                            nn.Conv2d(hidden_units, hidden_units * 2, kernel_size=3, padding='same'),
                            nn.LeakyReLU(0.1),
                            nn.BatchNorm2d(hidden_units * 2),
                            nn.Conv2d(hidden_units * 2, hidden_units * 2, kernel_size=3, padding='same'),
                            nn.LeakyReLU(0.1),
                            nn.BatchNorm2d(hidden_units * 2),
                            nn.MaxPool2d(2),
                            nn.Dropout(0.4)
                        )

                        self.block3 = nn.Sequential(
                            nn.Conv2d(hidden_units * 2, hidden_units * 4, kernel_size=3, padding='same'),
                            nn.LeakyReLU(0.1),
                            nn.BatchNorm2d(hidden_units * 4),
                            nn.Conv2d(hidden_units * 4, hidden_units * 4, kernel_size=3, padding='same'),
                            nn.LeakyReLU(0.1),
                            nn.BatchNorm2d(hidden_units * 4),
                            nn.MaxPool2d(2),
                            nn.Dropout(0.4)
                        )

                        self.block4 = nn.Sequential(
                            nn.Conv2d(hidden_units * 4, hidden_units * 8, kernel_size=3, padding='same'),
                            nn.LeakyReLU(0.1),
                            nn.BatchNorm2d(hidden_units * 8),
                            nn.Conv2d(hidden_units  * 8, hidden_units  * 8, kernel_size=3, padding='same'),
                            nn.LeakyReLU(0.1),
                            nn.BatchNorm2d(hidden_units * 8),
                            nn.MaxPool2d(2),
                            nn.Dropout(0.5)
                        )

                        self.adaptiveAvgPool = nn.AdaptiveAvgPool2d(1)

                        self.classifier = nn.Sequential(
                            nn.Flatten(),
                            nn.Linear(hidden_units * 8, 512),
                            nn.LeakyReLU(0.1),
                            nn.BatchNorm1d(512),
                            nn.Dropout(0.6),
                            nn.Linear(512, output_shape)
                        )

                    def forward(self, x: torch.Tensor):
                        x = self.block1(x)
                        x = self.block2(x)
                        x = self.block3(x)
                        x = self.block4(x)
                        x = self.adaptiveAvgPool(x)
                        x = self.classifier(x)
                        return x"""
            st.code(arc)
        
        st.image('modelv2_output.png', caption='Models accuracy and loss curves')
        st.write('Hyperparams:')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.caption('**Epochs**: `n_epochs=30`')
            st.caption('**Learning Rate**: `lr=0.00001`')
            st.caption('**Scheduler**: `ReduceLROnPlateau(optimizer, mode="min", patience=5)`')
            
        with col2:
            st.caption('**Data Loader Batches**: `DataLoader(train_dataset, batch_size=64, shuffle=True)`')
            st.caption('**Loss Function**: `nn.CrossEntropyLoss()`')
            st.caption('**Optimizer**: `optim.Adam(params=model_2.parameters(), lr=0.00001, weight_decay=1e-4)`')
            
    with data_tab:
        col1, col2 = st.columns(2)
        
        with col1:
            dataset_path = Path('./dataset/colored_images/')
            
            labels = next(os.walk(dataset_path))[1]
            random_index = random.randint(1, len(labels))
            random_label = labels[random_index]

            label_dir = dataset_path / random_label

            # Get random image
            random_image = label_dir / random.choice(os.listdir(label_dir))

            parsed_image = Image.open(Path(random_image)).convert("RGB")
            
            st.image(parsed_image, caption='Sample image from the dataset', width=300)
            
    with col2:
        st.header('APTOS 2019 Blindness Detection')
        st.caption('The images consist of retina scan images to detect diabetic retinopathy. The original dataset is available at APTOS 2019 Blindness Detection. These images are resized into 224x224 pixels so that they can be readily used with many pre-trained deep learning models. All of the images are already saved into their respective folders according to the severity/stage of diabetic retinopathy using the train.csv file provided. You will find five directories with the respective images:')
        url = 'https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-2019-data'
        st.caption('For more details about the dataset, visit [Kaggle](%s).' % url)
        

if __name__ == '__main__':
    main()
