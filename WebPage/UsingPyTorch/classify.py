import torchvision
import torch
import torchvision.transforms as transforms
import PIL.Image as Image
import streamlit
classes = [
    'Mantled howler',
    'Patas monkey',
    'Bald uakari',
    'Japanese macaque',
    'Pygmy marmoset',
    'White headed capuchin',
    'Silvery marmoset',
    'Common squirrel monkey',
    'Black headed night monkey',
    'Nilgiri langur'
]
model = torch.load('best_model.pth')
mean = [0.4363, 0.4328, 0.3291]
std = [0.2129, 0.2075, 0.2038]
image_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])

def classify(model, image_tranforms, image, classes):
    model = model.eval()
    image = image_tranforms(image).float()
    image = image.unsqueeze(0)

    output = model(image)
    _, predicted = torch.max(output.data, 1)

    print(classes[predicted.item()])
    return classes[predicted.item()]

uploaded_file = streamlit.file_uploader("Choose a file")
streamlit.subheader('Upload a picture of one of these types of monkeys:')
text = ''
for i in classes:
    text += f"{i}\n"
streamlit.text(text)
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    streamlit.image(image)
    streamlit.subheader(classify(model, image_transforms, image, classes))
