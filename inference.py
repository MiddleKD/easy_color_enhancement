from PIL import Image
from function import handler 

def inference(img_pil, gamma=0.75, factor=1.7):
    result = handler(img_pil, gamma, factor)
    return Image.fromarray(result)

if __name__ == "__main__":
    
    img_pil = Image.open("./Images/cup.jpg")
    dst = inference(img_pil, gamma=0.75, factor=1.7)
    
    dst.show()